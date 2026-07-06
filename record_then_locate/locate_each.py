#!/usr/bin/env python3
"""
locate_each.py -- run every TDOA weighting method on the SAME recording and
print each method's estimated direction of arrival, side by side.

It imports the six locate_*.py scripts as modules and reuses each one's own
weighting (the only part that differs between them), so the numbers here match
exactly what running each script standalone would produce. By default it picks
the newest WAV in mic_config.RECORDINGS_DIR ("the latest recorded sound").

Run:
    python locate_each.py                        # newest recording
    python locate_each.py path/to/recording.wav  # a specific file
    python locate_each.py path/to/rec.wav 100     # ...with a known true angle
    python locate_each.py 100                      # newest recording, true angle 100
"""

import os
import sys
import glob
import time

import numpy as np

# This file lives next to mic_config.py; the six method scripts may sit here or
# in an "other_methods/" subfolder. Put both on the path so the imports below
# work either way (and so the method modules can still find mic_config).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, "other_methods")):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

from mic_config import (
    CHANNELS, CHUNK, WINDOW, CONF_RATIO, RECORDINGS_DIR, angle_to_led,
)

# Import each method as a separate module. Each exposes the same helpers
# (srp_phat / circular_mean / angular_error / read_wav / METHOD); only its
# weight()/postprocess() differ.
import locate_cc
import locate_roth
import locate_scot
import locate_phat
import locate_scot_phat
import locate_gtcc

# Order = baseline first, then increasing sophistication / compute.
METHODS = [
    locate_cc,
    locate_roth,
    locate_scot,
    locate_phat,
    locate_scot_phat,
    locate_gtcc,
]


def newest_recording():
    files = glob.glob(os.path.join(RECORDINGS_DIR, "*.wav"))
    if not files:
        sys.exit("No recordings found in %s. Run record_one.py first." % RECORDINGS_DIR)
    return max(files, key=os.path.getmtime)


def parse_args(argv):
    """Positional args are order-insensitive: a path (contains '.wav' / a
    separator / an existing file) is the recording; a bare number is the true
    angle. Anything omitted falls back to (newest recording, no true angle)."""
    wav_path = None
    true_angle = None
    for a in argv:
        if wav_path is None and (os.path.exists(a) or a.lower().endswith(".wav") or os.sep in a):
            wav_path = a
        else:
            try:
                true_angle = float(a)
            except ValueError:
                sys.exit("unrecognised argument: %s" % a)
    if wav_path is None:
        wav_path = newest_recording()
    return wav_path, true_angle


def frames_from(channels):
    """De-interleaved channels -> list of windowed multi-channel frames.
    Identical framing to every locate_*.py script."""
    usable = min((len(c) for c in channels), default=0)
    n_frames = usable // CHUNK
    if n_frames < 1:
        sys.exit("file too short: need >= %d samples/channel, got %d" % (CHUNK, usable))
    frames = []
    for f in range(n_frames):
        s = f * CHUNK
        frames.append([c[s:s + CHUNK] * WINDOW for c in channels])
    return frames


def run_method(mod, frames):
    """Localize every frame with one method's srp_phat, timing compute only.
    Returns (estimate_or_None, n_confident, n_total, proc_ms, ms_per_frame)."""
    t0 = time.perf_counter()
    confident = []
    for frame in frames:
        az, conf = mod.srp_phat(frame)
        if conf >= CONF_RATIO:
            confident.append(az)
    t1 = time.perf_counter()

    proc_ms = (t1 - t0) * 1000.0
    total = len(frames)
    ms_per_frame = proc_ms / total if total else 0.0
    est = mod.circular_mean(confident) if confident else None
    return est, len(confident), total, proc_ms, ms_per_frame


def main():
    wav_path, true_angle = parse_args(sys.argv[1:])

    # Read once (locate_cc.read_wav is identical in every module) -- not timed.
    data, n_ch = locate_cc.read_wav(wav_path)
    if n_ch != CHANNELS:
        sys.exit("expected %d channels, got %d in %s" % (CHANNELS, n_ch, wav_path))
    channels = [data[c::CHANNELS] for c in range(CHANNELS)]
    frames = frames_from(channels)

    print("Recording : %s" % wav_path)
    print("Frames    : %d  (CHUNK=%d, conf threshold=%.2f)" % (len(frames), CHUNK, CONF_RATIO))
    if true_angle is not None:
        print("True angle: %.1f deg" % true_angle)
    print("")

    header = "%-11s %10s %6s %12s %8s %6s" % (
        "METHOD", "AZIMUTH", "LED", "CONFIDENT", "ms/frame", "ERROR")
    print(header)
    print("-" * len(header))

    for mod in METHODS:
        est, n_conf, n_total, proc_ms, ms_pf = run_method(mod, frames)
        if est is None:
            print("%-11s %10s %6s %12s %8.3f %6s"
                  % (mod.METHOD, "--", "--", "0/%d" % n_total, ms_pf, "--"))
            continue

        led = angle_to_led(est)
        err = "--"
        if true_angle is not None:
            err = "%.1f" % mod.angular_error(est, true_angle)
        print("%-11s %9.1f%s %6d %12s %8.3f %6s"
              % (mod.METHOD, est, "°", led, "%d/%d" % (n_conf, n_total), ms_pf, err))


if __name__ == "__main__":
    main()
