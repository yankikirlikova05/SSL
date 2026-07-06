#!/usr/bin/env python3
"""
record_one.py  --  Record RECORD_SECONDS of 4-channel audio from the ReSpeaker
4-Mic Array and save it as a WAV into the recordings/ folder.

Usage:
    python record_one.py            # records for mic_config.RECORD_SECONDS
    python record_one.py 8          # override: record for 8 seconds

The saved file is 4-channel interleaved 16-bit PCM, ready for locate_one.py.
"""

import os
import sys
import wave
from datetime import datetime

import pyaudio

import mic_config as cfg


def record(seconds):
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=cfg.SAMPLE_RATE,
        format=pa.get_format_from_width(cfg.SAMPLE_WIDTH),
        channels=cfg.CHANNELS,
        input=True,
        input_device_index=cfg.RESPEAKER_INDEX,
        frames_per_buffer=cfg.CHUNK,
    )

    print("* recording {} s on {} channels...".format(seconds, cfg.CHANNELS))

    frames = []
    for _ in range(0, int(cfg.SAMPLE_RATE / cfg.CHUNK * seconds)):
        frames.append(stream.read(cfg.CHUNK, exception_on_overflow=False))

    print("* done recording")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    return b"".join(frames)


def save(raw_bytes):
    os.makedirs(cfg.RECORDINGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg.RECORDINGS_DIR, "rec_{}.wav".format(stamp))

    wf = wave.open(path, "wb")
    wf.setnchannels(cfg.CHANNELS)
    wf.setsampwidth(cfg.SAMPLE_WIDTH)
    wf.setframerate(cfg.SAMPLE_RATE)
    wf.writeframes(raw_bytes)
    wf.close()
    return path


if __name__ == "__main__":
    seconds = cfg.RECORD_SECONDS
    if len(sys.argv) > 1:
        seconds = float(sys.argv[1])

    raw = record(seconds)
    out = save(raw)
    print("* saved -> {}".format(out))
