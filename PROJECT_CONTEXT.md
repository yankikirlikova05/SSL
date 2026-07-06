# Project Context — Sound Source Localization (SSL)

A handoff brief describing the current state of this repository, written for an AI
agent that will build on it. Read this first, then read the files it references.

## What this project is

A **sound-source-localization "compass"** for the **ReSpeaker 4-Mic Array HAT** on a
Raspberry Pi. Given audio from the 4 microphones, it estimates the **direction of
arrival (DoA)** of a sound (its azimuth, 0–360°) and points to it using the HAT's
12-LED ring.

The signal-processing pipeline is:

```
4-channel audio  ->  GCC-PHAT (per mic pair)  ->  SRP-PHAT (search all azimuths)  ->  azimuth + LED
```

This repo is a fork of ReSpeaker's `mic_hat` examples. The stock examples
(Alexa/Google Assistant demos, basic recording scripts, LED drivers) are still
present; the new localization work sits on top of them.

## Hardware / runtime assumptions

- **ReSpeaker 4-Mic Array HAT** on a Raspberry Pi (4 input channels, 16 kHz, 16-bit).
- 12-LED APA102 ring driven over SPI; LED power gated on GPIO5.
- Python 3 with `numpy`, `pyaudio`, `spidev`, `gpiozero` (see `requirements.txt`).
- The **live/LED/recording code only runs on the Pi.** The offline analysis code
  (`record_then_locate/locate_one.py`) is pure NumPy and runs anywhere.

## The DSP, briefly (so you don't have to reverse-engineer it)

- **Array geometry:** 4 mics on a circle of radius `R = 0.032 m` at 45/135/225/315°.
- **GCC-PHAT:** for a pair of channels, cross-power spectrum with PHAT weighting
  (`R /= |R|`, i.e. keep phase, discard magnitude — robust to level and reverb),
  inverse-FFT'd at `INTERP=8×` for sub-sample time resolution. The peak lag = the
  time difference of arrival (TDOA) for that pair.
- **SRP-PHAT:** for each candidate azimuth (0–360° in 1° steps), sum the GCC-PHAT
  values that the 6 mic pairs *should* show for a far-field source at that angle.
  The azimuth with the largest sum is the estimate. A precomputed lookup table
  (`KIDX`) maps every (angle, pair) to the exact correlation index, so the hot loop
  is pure array indexing — no per-frame trig.
- **Confidence:** `peak / mean(|SRP|)`. Below `CONF_RATIO = 2.2` the direction is
  treated as untrustworthy (too quiet / too diffuse) and ignored.
- **Smoothing (live only):** the pointer direction is smoothed as a unit vector, not
  a raw angle, so it doesn't glitch across the 0°/360° wraparound.

## File map

### Top level
- `doa_compass.py` — **the main program.** Live real-time compass: reads the mic
  stream, runs GCC-PHAT/SRP-PHAT per audio chunk, and lights the LED ring toward the
  sound. `python doa_compass.py` runs live; `python doa_compass.py --test-leds` runs
  an LED calibration sweep. Requires the Pi + HAT. This file is the reference
  implementation of the DSP.
- `requirements.txt` — `spidev`, `gpiozero`, `pyaudio` (numpy also required).
- `README.md` — **stale.** Still describes the original ReSpeaker Google Assistant
  setup; does not mention the compass. Worth updating.
- `output.wav`, `record_test.py` (empty) — leftover scratch files.

### `interfaces/` — hardware drivers (from the upstream fork, unchanged)
- `apa102.py` — APA102 LED-ring driver (`set_pixel`, `show`, `clear_strip`).
- `pixels.py`, `alexa_led_pattern.py`, `google_home_led_pattern.py` — LED animations.

### `recording_examples/` — stock ReSpeaker examples (reference only)
- `get_device_index.py` — **run this on the Pi to find the mic's input device id.**
- `record.py`, `record_one_channel.py` — basic recording examples.

### `online_service_demos/` — stock Alexa / Google Assistant demos (not part of SSL)

### `record_then_locate/` — offline "record, then analyze" workflow (NEW, complete)
A decoupled alternative to the live compass: capture a clip, then estimate its DoA.
- `mic_config.py` — **single source of truth for all config.** Device index, sample
  rate, `CHANNELS=4`, `RECORD_SECONDS=5`, the `recordings/` path, the mic geometry
  (positions, radius, pairs), and the precomputed SRP-PHAT tables. Both other scripts
  import from here. **When changing mic setup or DSP params, change them here only.**
- `record_one.py` — records `RECORD_SECONDS` of 4-channel audio and saves a
  timestamped 4-channel WAV into `recordings/` (auto-created). Optional CLI arg
  overrides the duration. Requires the Pi.
- `locate_one.py` — loads a recording (newest in `recordings/` by default, or a path
  arg), runs the same GCC-PHAT/SRP-PHAT pipeline as `doa_compass.py` over every
  frame, vector-averages the confident frames, and prints the estimated DoA, the
  confidence, and which LED would light. **Pure NumPy — runs off-Pi.**

## How to run

On the Pi:
```bash
# find the mic device id, put it in mic_config.py / doa_compass.py (RESPEAKER_INDEX)
python recording_examples/get_device_index.py

# live compass with LED pointer
python doa_compass.py
python doa_compass.py --test-leds        # calibrate LED_OFFSET / LED_REVERSE

# record-then-locate workflow
cd record_then_locate
python record_one.py                     # make a sound during the 5 s
python locate_one.py                     # prints the DoA estimate
```

Off-Pi (analysis only): `python record_then_locate/locate_one.py path/to/rec.wav`

## Verified state

- `record_then_locate/` scripts: syntax-checked; `mic_config.py` computes the same
  tables as `doa_compass.py` (CC_LEN=47, KIDX 360×6). `locate_one.py` was validated
  on a synthetic 4-channel recording with a known source at 100° and recovered ~95°
  (within this small array's resolution). The recording/live paths need real hardware.

## Known gaps / good things to build next

1. **`RESPEAKER_INDEX` is hardcoded** (`= 2`) in both `mic_config.py` and
   `doa_compass.py`. Auto-detect the device by name using the logic in
   `get_device_index.py` instead of hardcoding.
2. **Mic channel→angle mapping is assumed** (45/135/225/315°). `LED_OFFSET`/
   `LED_REVERSE` only calibrate the LED display, not the mic mapping. If DoA comes out
   rotated or mirrored, the physical channel order needs verifying. A mic-calibration
   routine (play a sound from a known angle, solve for the offset) would help.
3. **Azimuth only, no elevation** — the array is planar, so this is inherent, but
   front/back ambiguity is worth documenting for users.
4. **README is stale** — should document the compass and the record-then-locate flow.
5. **DSP duplication** — `doa_compass.py` predates `mic_config.py` and defines its own
   copies of the geometry/DSP constants. Consider refactoring `doa_compass.py` to
   import from `mic_config.py` so there's one definition. (Note: `doa_compass.py`
   imports hardware libs at module load, so importing *it* off-Pi fails — the shared
   config should stay hardware-free, as `mic_config.py` currently is.)
6. **No tests** — the synthetic-signal check used during development could be turned
   into a repeatable unit test for the localization math (no hardware needed).

## Conventions to follow

- Keep all configuration and the mic definition in `record_then_locate/mic_config.py`;
  import from it rather than redefining constants.
- Match the existing style: plain functions, NumPy, docstrings explaining the *why*
  of the DSP (see `doa_compass.py` / `locate_one.py` for the tone).
- Anything hardware-touching (`pyaudio`, `gpiozero`, `apa102`) belongs in the scripts,
  not in the shared config, so the math stays runnable off-Pi.
