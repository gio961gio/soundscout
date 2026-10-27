"""Generate a tiny, fully synthetic library to smoke-test SoundScout locally.

These sounds are intentionally synthetic and are not a substitute for evaluating
search quality on real sound effects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48_000


def normalize(x: np.ndarray, peak: float = 0.85) -> np.ndarray:
    maximum = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / maximum)).astype(np.float32) if maximum > 1e-9 else x.astype(np.float32)


def make_library(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    # A clean sustained tone.
    t = np.arange(SR * 2) / SR
    tone = 0.45 * np.sin(2 * np.pi * 440 * t)
    sf.write(output / "01_sustained_tone_synthetic.wav", normalize(tone), SR)

    # Short burst of white noise with a fast exponential decay.
    n = int(SR * 1.2)
    t = np.arange(n) / SR
    noise = rng.normal(0, 1, n) * np.exp(-t * 7)
    sf.write(output / "02_noise_burst_synthetic.wav", normalize(noise), SR)

    # Low, slow rumble from filtered noise and a sub bass component.
    n = SR * 4
    t = np.arange(n) / SR
    raw = rng.normal(0, 1, n)
    lowpass = butter(4, 140, btype="lowpass", fs=SR, output="sos")
    rumble = sosfilt(lowpass, raw) + 0.2 * np.sin(2 * np.pi * 48 * t)
    env = 0.75 + 0.25 * np.sin(2 * np.pi * 0.35 * t)
    sf.write(output / "03_low_rumble_synthetic.wav", normalize(rumble * env), SR)

    # A synthetic metallic ring: inharmonic partials and a long decay.
    n = int(SR * 3.5)
    t = np.arange(n) / SR
    ring = np.zeros_like(t)
    for freq, amp, decay in [(880, 1.0, 2.4), (1320, 0.65, 1.8), (2210, 0.38, 1.3), (3170, 0.22, 0.9)]:
        ring += amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)
    sf.write(output / "04_metallic_ring_synthetic.wav", normalize(ring), SR)

    # Filtered-noise whoosh with a moving amplitude envelope.
    n = int(SR * 3)
    t = np.arange(n) / SR
    raw = rng.normal(0, 1, n)
    band = butter(3, [350, 5000], btype="bandpass", fs=SR, output="sos")
    whoosh = sosfilt(band, raw)
    env = np.sin(np.pi * np.clip(t / 3, 0, 1)) ** 1.8
    sf.write(output / "05_sweeping_whoosh_synthetic.wav", normalize(whoosh * env), SR)

    print(f"Generated 5 synthetic test files in: {output.resolve()}")
    print("Use these only to smoke-test indexing/playback; validate semantic quality on real sounds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="./demo_audio", help="Output folder (default: ./demo_audio)")
    args = parser.parse_args()
    make_library(Path(args.output).expanduser())
