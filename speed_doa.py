#!/usr/bin/env python3
"""
doa_compass.py  --  Sound-source-localization "compass" for the ReSpeaker 4-Mic Array.

Pipeline:  4-ch audio  ->  GCC-PHAT per mic pair  ->  SRP-PHAT azimuth  ->  LED ring pointer.

SRP-PHAT = for every candidate azimuth, sum the GCC-PHAT correlation each mic pair
*should* show if the sound came from that angle, then pick the angle with the biggest sum.
It's just GCC-PHAT (which you already planned) combined across all 6 pairs, which is far
more robust than averaging per-pair arcsin bearings.

WHERE TO PUT THIS FILE
    Drop it in your ~/4mics_hat folder (the one containing interfaces/pixels.py) and run it
    from there so it can import the APA102 LED driver:
        cd ~/4mics_hat
        python3 doa_compass.py            # live compass
        python3 doa_compass.py --test-leds # LED calibration sweep (no audio)

TWO THINGS YOU MUST CALIBRATE (see the CONFIG block):
    1. MIC_POSITIONS  -- which audio CHANNEL sits at which physical corner. Use your tap
       test. Get this wrong and the compass points to a rotated/mirrored direction.
    2. LED_OFFSET / LED_REVERSE -- align "0 degrees in the maths" with "the physical LED
       that should light". Use --test-leds to dial this in.
"""

import sys
import time
import numpy as np
import pyaudio

# ============================ CONFIG -- EDIT THESE ============================

# ---- Audio device ----
RESPEAKER_INDEX = 2        # your card came up as card 2; confirm with get_device_index.py
SAMPLE_RATE     = 16000    # the AC108 runs at 16 kHz
CHANNELS        = 4
CHUNK           = 1024     # samples per frame per channel (~64 ms at 16 kHz)
FORMAT          = pyaudio.paInt16

# ---- Physics ----
SPEED_OF_SOUND  = 343.0    # m/s (room temperature)
INTERP          = 8        # sub-sample interpolation for GCC-PHAT (higher = finer, more CPU)

# ---- Microphone geometry (METRES, relative to array centre) ----
# The 4-Mic Array is a square with mics near the corners. R is the corner radius.
# THESE ARE APPROXIMATE -- measure your board and, crucially, set the ORDER so that
# index 0 is the physical corner that audio channel 0 comes from (your tap test).
R = 0.032
MIC_POSITIONS = np.array([
    [ R * np.cos(np.radians( 45)), R * np.sin(np.radians( 45)) ],  # channel 0
    [ R * np.cos(np.radians(135)), R * np.sin(np.radians(135)) ],  # channel 1
    [ R * np.cos(np.radians(225)), R * np.sin(np.radians(225)) ],  # channel 2
    [ R * np.cos(np.radians(315)), R * np.sin(np.radians(315)) ],  # channel 3
])

# ---- LED ring ----
NUM_LED      = 12          # APA102 count on the 4-Mic Array
LED_OFFSET   = 0.0         # degrees; rotate the mapping so 0deg lights the right LED
LED_REVERSE  = False       # flip if the pointer spins the wrong way round the ring
LED_POWER_GPIO = 5         # GPIO5 must be HIGH to power the LEDs on this board

# ---- Display behaviour ----
ANGLE_RES_DEG   = 1.0      # azimuth search resolution
CONF_RATIO      = 1.6      # peak/mean SRP ratio required to show a direction (else LEDs off)
SMOOTH_ALPHA    = 0.35     # 0..1 ; lower = smoother/slower pointer
MAIN_COLOR      = (0, 60, 0)   # (R,G,B) 0-255 for the pointing LED
NEIGH_COLOR     = (0, 12, 0)   # dim colour for the two neighbour LEDs (arc effect)

# =============================================================================


# ---- LED driver import (from the repo's interfaces/ folder) ----
sys.path.insert(0, "interfaces")
try:
    from apa102 import APA102
except ImportError:
    print("ERROR: could not import apa102. Run this from your 4mics_hat folder "
          "(the one with interfaces/apa102.py).")
    sys.exit(1)

# ---- Enable LED power rail on GPIO5 ----
try:
    from gpiozero import LED
    _power = LED(LED_POWER_GPIO)
    _power.on()
except Exception as e:
    _power = None
    print("Note: could not toggle GPIO5 LED power ({}). LEDs may stay dark.".format(e))


# ============================ ALGORITHM CORE ============================

# All 6 unique microphone pairs.
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Maximum possible delay for any pair (diagonal of the array) -> bounds the search window.
_max_dist = max(np.linalg.norm(MIC_POSITIONS[i] - MIC_POSITIONS[j]) for i, j in PAIRS)
MAX_TAU   = _max_dist / SPEED_OF_SOUND

# Fixed frame length used inside gcc_phat (sig + ref lengths).
_N        = 2 * CHUNK
MAX_SHIFT = min(int(INTERP * SAMPLE_RATE * MAX_TAU), int(INTERP * _N / 2))
CC_LEN    = 2 * MAX_SHIFT + 1

# Candidate azimuths (degrees) and a Hann window for framing.
ANGLES    = np.arange(0.0, 360.0, ANGLE_RES_DEG)
WINDOW    = np.hanning(CHUNK)


def _expected_tau(i, j, azimuth_deg):
    """TDOA (seconds) that pair (i,j) should show for a far-field source at this azimuth.
    Convention: source located at azimuth phi -> unit vector u = (cos phi, sin phi).
    A mic closer to the source hears the sound earlier, so tau_ij = -((p_i - p_j).u)/c."""
    u = np.array([np.cos(np.radians(azimuth_deg)), np.sin(np.radians(azimuth_deg))])
    return -np.dot(MIC_POSITIONS[i] - MIC_POSITIONS[j], u) / SPEED_OF_SOUND


# Precompute, for every (angle, pair), the index into the GCC-PHAT array to read.
# Shape: (num_angles, num_pairs).  Done once at startup -- the hot loop is just indexing.
_KIDX = np.zeros((len(ANGLES), len(PAIRS)), dtype=np.int64)
for a, ang in enumerate(ANGLES):
    for p, (i, j) in enumerate(PAIRS):
        tau = _expected_tau(i, j, ang)
        k = int(round(tau * INTERP * SAMPLE_RATE)) + MAX_SHIFT
        _KIDX[a, p] = min(max(k, 0), CC_LEN - 1)

_PAIR_IDX = np.arange(len(PAIRS))[None, :]   # for fancy indexing


def gcc_phat(sig, ref):
    """Return the PHAT-weighted cross-correlation of sig vs ref, as an array indexed
    0..2*MAX_SHIFT where index MAX_SHIFT is zero lag. Peak position => delay of sig vs ref."""
    SIG = np.fft.rfft(sig, n=_N)
    REF = np.fft.rfft(ref, n=_N)
    R = SIG * np.conj(REF)
    R /= np.abs(R) + 1e-12                    # <-- the PHAT weighting (keep phase, drop magnitude)
    cc = np.fft.irfft(R, n=INTERP * _N)
    cc = np.concatenate((cc[-MAX_SHIFT:], cc[:MAX_SHIFT + 1]))
    return cc


def srp_phat(channels):
    """channels: list of 4 mono float arrays. Returns (azimuth_deg, confidence_ratio)."""
    ccs = np.empty((len(PAIRS), CC_LEN))
    for p, (i, j) in enumerate(PAIRS):
        ccs[p] = gcc_phat(channels[i], channels[j])

    # For each candidate angle, sum the correlation each pair shows at its expected delay.
    srp = np.sum(ccs[_PAIR_IDX, _KIDX], axis=1)     # shape: (num_angles,)

    best = int(np.argmax(srp))
    peak = srp[best]
    mean = np.mean(np.abs(srp)) + 1e-12
    return ANGLES[best], peak / mean


# ============================ LED COMPASS ============================

strip = APA102(num_led=NUM_LED)


def angle_to_led(azimuth_deg):
    step = 360.0 / NUM_LED
    idx = int(round((azimuth_deg + LED_OFFSET) / step)) % NUM_LED
    if LED_REVERSE:
        idx = (NUM_LED - idx) % NUM_LED
    return idx


def show_direction(azimuth_deg):
    idx = angle_to_led(azimuth_deg)
    strip.clear_strip()
    strip.set_pixel(idx, *MAIN_COLOR)
    strip.set_pixel((idx - 1) % NUM_LED, *NEIGH_COLOR)
    strip.set_pixel((idx + 1) % NUM_LED, *NEIGH_COLOR)
    strip.show()


def leds_off():
    strip.clear_strip()
    strip.show()


# ============================ MODES ============================

def run_led_test():
    """Sweep a pointer around the ring so you can calibrate LED_OFFSET / LED_REVERSE.
    Watch which physical LED lights for each printed angle, then adjust the config."""
    print("LED calibration sweep. Ctrl-C to stop.")
    print("Adjust LED_OFFSET / LED_REVERSE until the lit LED matches the printed angle.")
    try:
        for ang in range(0, 360, 15):
            print("  angle = {:3d} deg  ->  LED {}".format(ang, angle_to_led(ang)))
            show_direction(ang)
            time.sleep(0.6)
        leds_off()
    except KeyboardInterrupt:
        leds_off()


def run_live():
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=SAMPLE_RATE, format=FORMAT, channels=CHANNELS,
                     input=True, input_device_index=RESPEAKER_INDEX,
                     frames_per_buffer=CHUNK)
    print("Listening... make a sound around the array. Ctrl-C to stop.")

    smooth_vec = np.zeros(2)   # smoothed unit direction (handles 0/360 wraparound cleanly)
    try:
        while True:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            # De-interleave: samples arrive [c0,c1,c2,c3, c0,c1,c2,c3, ...]
            channels = [data[c::CHANNELS] * WINDOW for c in range(CHANNELS)]

            azimuth, conf = srp_phat(channels)

            if conf < CONF_RATIO:
                leds_off()
                continue

            # Smooth on the unit circle so the pointer doesn't jump across the 359->0 seam.
            v = np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))])
            smooth_vec = SMOOTH_ALPHA * v + (1 - SMOOTH_ALPHA) * smooth_vec
            smoothed = np.degrees(np.arctan2(smooth_vec[1], smooth_vec[0])) % 360

            show_direction(smoothed)
            print("azimuth ~ {:5.1f} deg   (confidence {:.2f})   LED {}"
                  .format(smoothed, conf, angle_to_led(smoothed)), end="\r")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        leds_off()
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    if "--test-leds" in sys.argv:
        run_led_test()
    else:
        run_live()
