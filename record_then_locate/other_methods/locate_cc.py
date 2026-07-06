#!/usr/bin/env python3
"""
locate_cc.py -- offline TDOA sound-source localization, plain cross-correlation (baseline).

Part of a time-domain TDOA comparison study on the ReSpeaker 4-Mic Array.
Every locate_*.py script is byte-for-byte identical EXCEPT the weight()
function (and, for GTCC, the postprocess() step). All shared parameters and
array geometry come from mic_config.py -- the single source of truth.

Run:
    python locate_cc.py path/to/recording.wav [true_angle_deg]
"""

import os
import sys
import csv
import time
import wave
from datetime import datetime

import numpy as np

from mic_config import (
    CHANNELS, SAMPLE_RATE, CHUNK, PAIRS, KIDX, PAIR_IDX, ANGLES,
    MAX_SHIFT, CC_LEN, N, WINDOW, INTERP, CONF_RATIO, angle_to_led,
)

METHOD = "CC"

EPS = 1e-12  # guards every magnitude division against divide-by-zero
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")


# ======================================================================
# THE ONLY PART THAT DIFFERS BETWEEN SCRIPTS
# ======================================================================
def weight(R, SIG, REF):
    """CC -- plain cross-correlation: NO weighting (psi = 1). Baseline against
    which every weighted transform below is compared."""
    return R

def postprocess(cc):
    """No time-domain post-processing (identity). Only GTCC overrides this."""
    return cc
# ======================================================================


def gcc(sig, ref):
    """Weighted generalized cross-correlation for one mic pair. FFT sizes and
    the MAX_SHIFT concatenation window are identical across every script; only
    weight() applied to R = SIG*conj(REF) changes."""
    SIG = np.fft.rfft(sig, n=N)
    REF = np.fft.rfft(ref, n=N)
    R = SIG * np.conj(REF)
    Rw = weight(R, SIG, REF)                     # <-- swappable weighting
    cc = np.fft.irfft(Rw, n=INTERP * N)
    cc = np.concatenate((cc[-MAX_SHIFT:], cc[:MAX_SHIFT + 1]))
    return postprocess(cc)


def srp_phat(frame_channels):
    """SRP-style fusion: sum per-pair GCC values at the precomputed delay
    indices (KIDX) for each candidate angle; return (best_angle, confidence)."""
    ccs = np.empty((len(PAIRS), CC_LEN))
    for p, (i, j) in enumerate(PAIRS):
        ccs[p] = gcc(frame_channels[i], frame_channels[j])

    srp = np.sum(ccs[PAIR_IDX, KIDX], axis=1)
    best = int(np.argmax(srp))
    peak = srp[best]
    mean = np.mean(np.abs(srp)) + EPS
    return ANGLES[best], peak / mean


def read_wav(path):
    """Read an interleaved 16-bit WAV -> (float64 samples, num_channels)."""
    wf = wave.open(path, "rb")
    n_ch = wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    return data, n_ch


def circular_mean(angles_deg):
    """Aggregation choice (SAME in every script): circular mean of the
    confident per-frame azimuths -- correct across the 0/360 wraparound."""
    r = np.radians(np.asarray(angles_deg, dtype=np.float64))
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0


def angular_error(a, b):
    """Absolute angular difference wrapped to 0..180 degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python %s path/to/recording.wav [true_angle_deg]"
                 % os.path.basename(sys.argv[0]))

    wav_path = sys.argv[1]
    true_angle = float(sys.argv[2]) if len(sys.argv) > 2 else None

    # --- file reading (NOT timed) ---
    data, n_ch = read_wav(wav_path)
    if n_ch != CHANNELS:
        sys.exit("expected %d channels, got %d in %s" % (CHANNELS, n_ch, wav_path))
    channels = [data[c::CHANNELS] for c in range(CHANNELS)]
    usable = min((len(c) for c in channels), default=0)
    n_frames = usable // CHUNK
    if n_frames < 1:
        sys.exit("file too short: need >= %d samples/channel, got %d" % (CHUNK, usable))

    # --- timed region: localization compute ONLY ---
    t0 = time.perf_counter()
    confident = []
    total_frames = 0
    for f in range(n_frames):
        s = f * CHUNK
        frame = [c[s:s + CHUNK] * WINDOW for c in channels]
        az, conf = srp_phat(frame)
        total_frames += 1
        if conf >= CONF_RATIO:
            confident.append(az)
    t1 = time.perf_counter()
    # -----------------------------------------------

    proc_ms = (t1 - t0) * 1000.0
    ms_per_frame = proc_ms / total_frames if total_frames else 0.0

    have_est = len(confident) > 0
    est = circular_mean(confident) if have_est else float("nan")
    led = angle_to_led(est) if have_est else ""

    err = ""
    if true_angle is not None and have_est:
        err = angular_error(est, true_angle)

    # --- CSV: required columns in exact order, error_deg appended last ---
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow([
                "timestamp", "wav_file", "method", "estimated_angle_deg",
                "confident_frames", "total_frames", "proc_time_ms",
                "ms_per_frame", "led_index", "error_deg",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            os.path.basename(wav_path),
            METHOD,
            "%.1f" % est if have_est else "",
            len(confident),
            total_frames,
            "%.3f" % proc_ms,
            "%.4f" % ms_per_frame,
            led,
            "%.1f" % err if err != "" else "",
        ])

    # --- human-readable summary ---
    if have_est:
        line = ("%-9s %s -> %.1f deg | confident %d/%d | %.1f ms (%.3f ms/frame) | LED %s"
                % (METHOD, os.path.basename(wav_path), est, len(confident),
                   total_frames, proc_ms, ms_per_frame, led))
        if err != "":
            line += " | error %.1f deg" % err
    else:
        line = ("%-9s %s -> NO CONFIDENT ESTIMATE | 0/%d frames | %.1f ms (%.3f ms/frame)"
                % (METHOD, os.path.basename(wav_path), total_frames, proc_ms, ms_per_frame))
    print(line)


if __name__ == "__main__":
    main()
