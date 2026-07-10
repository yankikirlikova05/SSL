#!/usr/bin/env python3
"""
mic_config.py  --  Shared configuration for the ReSpeaker 4-Mic Array.

Both record_one.py and locate_one.py import from here so the microphone
geometry and audio settings only ever live in one place.

Run recording_examples/get_device_index.py to find RESPEAKER_INDEX for your Pi.
"""

import os
import numpy as np

# audio device
RESPEAKER_INDEX = 2          # input device id (run get_device_index.py to confirm)
SAMPLE_RATE     = 16000      # Hz
CHANNELS        = 4          # 4-Mic HAT -> 4 input channels
SAMPLE_WIDTH    = 2          # bytes per sample (16-bit PCM)
CHUNK           = 1024       # frames per audio buffer

RECORD_SECONDS  = 30          # length of each recording

# where record_one.py drops its .wav files (created automatically)
_HERE          = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(_HERE, "lab_recordings")

# array geometry
SPEED_OF_SOUND = 343.0       # m/s (room temperature)
R              = 0.032       # array radius (m)

# 4 mics on a circle at 45 / 135 / 225 / 315 degrees.
MIC_POSITIONS = np.array([
    [ R * np.cos(np.radians( 45)), R * np.sin(np.radians( 45)) ],  # channel 0
    [ R * np.cos(np.radians(135)), R * np.sin(np.radians(135)) ],  # channel 1
    [ R * np.cos(np.radians(225)), R * np.sin(np.radians(225)) ],  # channel 2
    [ R * np.cos(np.radians(315)), R * np.sin(np.radians(315)) ],  # channel 3
])

# localization (GCC-PHAT / SRP-PHAT)
INTERP        = 8      # sub-sample interpolation for GCC-PHAT (higher = finer, more CPU)
ANGLE_RES_DEG = 1.0    # azimuth search resolution
CONF_RATIO    = 2.2    # peak/mean SRP ratio required to trust a direction

# led display mapping (used only to report which LED would light)
NUM_LED     = 12
LED_OFFSET  = 180.0
LED_REVERSE = False


# derived values (precomputed once so importing files stay lean)
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Maximum possible delay for any pair (largest baseline) -> bounds the search window.
_max_dist = max(np.linalg.norm(MIC_POSITIONS[i] - MIC_POSITIONS[j]) for i, j in PAIRS)
MAX_TAU   = _max_dist / SPEED_OF_SOUND

# Fixed frame length used inside gcc_phat (sig + ref lengths).
N         = 2 * CHUNK
MAX_SHIFT = min(int(INTERP * SAMPLE_RATE * MAX_TAU), int(INTERP * N / 2))
CC_LEN    = 2 * MAX_SHIFT + 1

# Candidate azimuths (degrees) and a Hann window for framing.
ANGLES = np.arange(0.0, 360.0, ANGLE_RES_DEG)
WINDOW = np.hanning(CHUNK)


def _expected_tau(i, j, azimuth_deg):
    """TDOA (seconds) that pair (i, j) should show for a far-field source at this azimuth.
    A mic closer to the source hears the sound earlier, so tau_ij = -((p_i - p_j).u)/c."""
    u = np.array([np.cos(np.radians(azimuth_deg)), np.sin(np.radians(azimuth_deg))])
    return -np.dot(MIC_POSITIONS[i] - MIC_POSITIONS[j], u) / SPEED_OF_SOUND


# For every (angle, pair), the index into the GCC-PHAT array to read.  Shape: (angles, pairs).
KIDX = np.zeros((len(ANGLES), len(PAIRS)), dtype=np.int64)
for _a, _ang in enumerate(ANGLES):
    for _p, (_i, _j) in enumerate(PAIRS):
        _tau = _expected_tau(_i, _j, _ang)
        _k = int(round(_tau * INTERP * SAMPLE_RATE)) + MAX_SHIFT
        KIDX[_a, _p] = min(max(_k, 0), CC_LEN - 1)

PAIR_IDX = np.arange(len(PAIRS))[None, :]   # for fancy indexing


def angle_to_led(azimuth_deg):
    """Which LED on the ring corresponds to this azimuth (for reporting only)."""
    step = 360.0 / NUM_LED
    idx = int(round((azimuth_deg + LED_OFFSET) / step)) % NUM_LED
    if LED_REVERSE:
        idx = (NUM_LED - idx) % NUM_LED
    return idx
