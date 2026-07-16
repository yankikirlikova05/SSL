#!/usr/bin/env python3

import os
import sys
import csv
import wave
import threading
from datetime import datetime

import numpy as np

from mic_config import (
    RESPEAKER_INDEX, SAMPLE_RATE, CHANNELS, CHUNK, SAMPLE_WIDTH,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(_HERE, "final_recordings/exp")
LOG_CSV = os.path.join(FINAL_DIR, "recordings_log.csv")


def wait_for_stop(stop_event):
    try:
        import termios
        import tty
        import select

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        stop_event.set()
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (ImportError, Exception):
        # Fallback: block until the user hits ENTER.
        input()
        stop_event.set()


def record():
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=SAMPLE_RATE,
        format=pa.get_format_from_width(SAMPLE_WIDTH),
        channels=CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
        frames_per_buffer=CHUNK,
    )

    start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    stop_event = threading.Event()
    watcher = threading.Thread(target=wait_for_stop, args=(stop_event,), daemon=True)
    watcher.start()

    print("* [%s] recording on %d channels..." % (start_stamp, CHANNELS))
    print("* press SPACE to stop.")

    frames = []
    try:
        while not stop_event.is_set():
            frames.append(stream.read(CHUNK, exception_on_overflow=False))
    except KeyboardInterrupt:
        stop_event.set()

    stream.stop_stream()
    stream.close()
    pa.terminate()

    print("\n* stopped.")
    return b"".join(frames), start_stamp


def _write_wav(path, raw_bytes, n_channels):
    wf = wave.open(path, "wb")
    wf.setnchannels(n_channels)
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(raw_bytes)
    wf.close()


def sanitize(name):
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name.strip())
    while "__" in cleaned:                      # collapse runs of underscores
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or datetime.now().strftime("rec_%Y%m%d_%H%M%S")


def unique_path(base, ext=".wav"):
    path = "%s%s" % (base, ext)
    n = 2
    while os.path.exists(path):
        path = "%s_%d%s" % (base, n, ext)
        n += 1
    return path


def save(raw_bytes, name):
    
    os.makedirs(FINAL_DIR, exist_ok=True)
    name = sanitize(name)

    path = unique_path(os.path.join(FINAL_DIR, name))
    _write_wav(path, raw_bytes, CHANNELS)
    return path, [path]


def log_row(timestamp, name, location, raw_bytes, n_files):
    """Append (timestamp, filename, ...) to the shared CSV log."""
    n_samples = len(raw_bytes) // (SAMPLE_WIDTH * CHANNELS)
    duration = n_samples / float(SAMPLE_RATE)

    write_header = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow([
                "timestamp", "name", "location", "channels",
                "duration_sec", "sample_rate", "num_files",
            ])
        writer.writerow([
            timestamp, name, os.path.relpath(location, _HERE),
            CHANNELS, "%.2f" % duration, SAMPLE_RATE, n_files,
        ])


def main():
    raw, start_stamp = record()

    n_samples = len(raw) // (SAMPLE_WIDTH * CHANNELS)
    if n_samples < 1:
        sys.exit("* nothing recorded -- aborting.")
    print("* captured %.2f s of audio." % (n_samples / float(SAMPLE_RATE)))

    name = input("* filename for this recording: ")
    location, files = save(raw, name)
    safe_name = os.path.basename(location).replace(".wav", "")

    log_row(start_stamp, safe_name, location, raw, len(files))

    print("")
    print("* saved %d file(s) to: %s" % (len(files), location))
    for f in files:
        print("    - %s" % os.path.basename(f))
    print("* logged to: %s" % os.path.relpath(LOG_CSV, _HERE))
    print("* done.")


if __name__ == "__main__":
    main()
