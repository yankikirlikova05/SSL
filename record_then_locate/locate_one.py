#!/usr/bin/env python3
"""
locate_one.py  --  Estimate the direction of arrival (DoA) of a sound from a
4-channel recording made by record_one.py.

It runs the same GCC-PHAT -> SRP-PHAT pipeline as doa_compass.py, but offline
over a whole WAV file instead of a live stream, then reports the overall
azimuth estimate.

Usage:
    python locate_one.py                       # use the newest file in recordings/
    python locate_one.py recordings/rec_x.wav  # use a specific recording
"""

import os
import sys
import glob
import wave

import numpy as np

import mic_config as cfg


def newest_recording():
    files = glob.glob(os.path.join(cfg.RECORDINGS_DIR, "*.wav"))
    if not files:
        sys.exit("No recordings found in {}. Run record_one.py first."
                 .format(cfg.RECORDINGS_DIR))
    return max(files, key=os.path.getmtime)


def load_channels(path):
    """Read a WAV and return a list of CHANNELS float64 arrays (de-interleaved)."""
    wf = wave.open(path, "rb")
    n_ch = wf.getnchannels()
    if n_ch != cfg.CHANNELS:
        sys.exit("Expected {}-channel audio, got {} in {}"
                 .format(cfg.CHANNELS, n_ch, path))
    rate = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    wf.close()

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    channels = [data[c::n_ch] for c in range(n_ch)]
    return channels, rate


def gcc_phat(sig, ref):
    """PHAT-weighted cross-correlation of sig vs ref, indexed 0..2*MAX_SHIFT
    (index MAX_SHIFT is zero lag)."""
    SIG = np.fft.rfft(sig, n=cfg.N)
    REF = np.fft.rfft(ref, n=cfg.N)
    R = SIG * np.conj(REF)
    R /= np.abs(R) + 1e-12                     # PHAT weighting: keep phase, drop magnitude
    cc = np.fft.irfft(R, n=cfg.INTERP * cfg.N)
    cc = np.concatenate((cc[-cfg.MAX_SHIFT:], cc[:cfg.MAX_SHIFT + 1]))
    return cc


def srp_phat(frame_channels):
    """Return (azimuth_deg, confidence) for one windowed multi-channel frame."""
    ccs = np.empty((len(cfg.PAIRS), cfg.CC_LEN))
    for p, (i, j) in enumerate(cfg.PAIRS):
        ccs[p] = gcc_phat(frame_channels[i], frame_channels[j])

    srp = np.sum(ccs[cfg.PAIR_IDX, cfg.KIDX], axis=1)

    best = int(np.argmax(srp))
    peak = srp[best]
    mean = np.mean(np.abs(srp)) + 1e-12
    return cfg.ANGLES[best], peak / mean


def locate(channels):
    """Run SRP-PHAT over every CHUNK-length frame and combine the confident
    estimates into a single azimuth (vector-averaged to handle 0/360 wrap)."""
    n = min(len(c) for c in channels)
    n_frames = n // cfg.CHUNK

    accum = np.zeros(2)     # confidence-weighted direction vector
    used = 0
    per_frame = []

    for f in range(n_frames):
        s = f * cfg.CHUNK
        e = s + cfg.CHUNK
        frame = [c[s:e] * cfg.WINDOW for c in channels]

        azimuth, conf = srp_phat(frame)
        per_frame.append((azimuth, conf))
        if conf < cfg.CONF_RATIO:
            continue

        rad = np.radians(azimuth)
        accum += conf * np.array([np.cos(rad), np.sin(rad)])
        used += 1

    if used == 0:
        return None, per_frame

    azimuth = np.degrees(np.arctan2(accum[1], accum[0])) % 360
    return azimuth, per_frame


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else newest_recording()
    print("* analyzing {}".format(path))

    channels, rate = load_channels(path)
    if rate != cfg.SAMPLE_RATE:
        print("  note: file sample rate {} differs from config {}"
              .format(rate, cfg.SAMPLE_RATE))

    azimuth, per_frame = locate(channels)

    confident = [c for _, c in per_frame if c >= cfg.CONF_RATIO]
    print("* frames analyzed : {}".format(len(per_frame)))
    print("* confident frames: {} (conf >= {})".format(len(confident), cfg.CONF_RATIO))

    if azimuth is None:
        print("* no confident direction found (source too quiet / too diffuse).")
        sys.exit(0)

    print("")
    print("  estimated direction of arrival : {:.1f} degrees".format(azimuth))
    print("  average confidence             : {:.2f}".format(np.mean(confident)))
    print("  would light LED                : {}".format(cfg.angle_to_led(azimuth)))
