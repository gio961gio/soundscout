"""Generate a rich, reproducible 36-sound evaluation library and ground-truth benchmark.

This dataset provides a legal, self-contained testbed for retrieval metrics (Recall@5, Precision@5, MRR)
across 6 acoustic categories (Impacts, Ambiences, Foley, Tones, Mechanical, Transitions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48_000


def norm(x: np.ndarray, peak: float = 0.88) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 1e-9 else x.astype(np.float32)


def make_dataset(base_dir: Path) -> tuple[Path, Path, Path]:
    library_dir = base_dir / "evaluation_library"
    eval_dir = base_dir / "evaluation"
    library_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    categories = ["Impacts", "Ambiences", "Foley", "Tones", "Mechanical", "Transitions"]
    for cat in categories:
        (library_dir / cat).mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict] = []

    # --- 1. IMPACTS ---
    # IMP_01: Metallic Anvil Impact
    n = int(SR * 3.5)
    t = np.arange(n) / SR
    sig = rng.normal(0, 0.4, n) * np.exp(-t * 25)
    for freq, amp, dec in [(520, 0.9, 2.5), (1140, 0.7, 3.2), (1980, 0.5, 4.0), (3420, 0.3, 5.0)]:
        sig += amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * dec)
    p = library_dir / "Impacts" / "metal_anvil_impact.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "IMP_01", "relative_path": "Impacts/metal_anvil_impact.wav",
        "category": "Impacts", "name": "metal_anvil_impact.wav", "duration_seconds": 3.5,
        "description": "Heavy metallic anvil strike with distinct modal inharmonic ringing and exponential decay.",
    })

    # IMP_02: Wooden Thud Heavy
    n = int(SR * 1.5)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * 110 * t) * np.exp(-t * 12) + 0.6 * np.sin(2 * np.pi * 230 * t) * np.exp(-t * 18)
    burst = rng.normal(0, 0.8, n) * np.exp(-t * 40)
    lowpass = butter(3, 350, btype="lowpass", fs=SR, output="sos")
    sig += sosfilt(lowpass, burst)
    p = library_dir / "Impacts" / "wood_thud_heavy.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "IMP_02", "relative_path": "Impacts/wood_thud_heavy.wav",
        "category": "Impacts", "name": "wood_thud_heavy.wav", "duration_seconds": 1.5,
        "description": "Solid heavy wooden thud impact with low damped resonance.",
    })

    # IMP_03: Glass Shatter Ping
    n = int(SR * 2.2)
    t = np.arange(n) / SR
    sig = rng.normal(0, 0.6, n) * np.exp(-t * 30)
    for freq, amp in [(2450, 0.8), (3820, 0.6), (5190, 0.5), (7150, 0.3)]:
        sig += amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * 3.8)
    p = library_dir / "Impacts" / "glass_shatter_ping.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "IMP_03", "relative_path": "Impacts/glass_shatter_ping.wav",
        "category": "Impacts", "name": "glass_shatter_ping.wav", "duration_seconds": 2.2,
        "description": "High bright glass strike and shatter ping with crystalline resonance.",
    })

    # IMP_04: Plastic Snap Click
    n = int(SR * 0.4)
    t = np.arange(n) / SR
    sig = rng.normal(0, 0.9, n) * np.exp(-t * 90) + 0.5 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 60)
    p = library_dir / "Impacts" / "plastic_snap_click.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "IMP_04", "relative_path": "Impacts/plastic_snap_click.wav",
        "category": "Impacts", "name": "plastic_snap_click.wav", "duration_seconds": 0.4,
        "description": "Short sharp plastic snap click with fast attack and dry finish.",
    })

    # IMP_05: Deep Sub Drop Boom
    n = int(SR * 4.0)
    t = np.arange(n) / SR
    freq_sweep = np.linspace(130, 32, n)
    phase = 2 * np.pi * np.cumsum(freq_sweep) / SR
    sig = np.sin(phase) * np.exp(-t * 0.9)
    sig += 0.3 * np.tanh(sig * 3.0)
    p = library_dir / "Impacts" / "deep_sub_drop_boom.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "IMP_05", "relative_path": "Impacts/deep_sub_drop_boom.wav",
        "category": "Impacts", "name": "deep_sub_drop_boom.wav", "duration_seconds": 4.0,
        "description": "Deep sub bass drop boom impact with prolonged low frequency tail.",
    })

    # IMP_06: Boxing Glove Punch
    n = int(SR * 0.8)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * 85 * t) * np.exp(-t * 15)
    noise = rng.normal(0, 0.7, n) * np.exp(-t * 35)
    band = butter(3, [80, 450], btype="bandpass", fs=SR, output="sos")
    sig += sosfilt(band, noise)
    p = library_dir / "Impacts" / "boxing_glove_punch.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "IMP_06", "relative_path": "Impacts/boxing_glove_punch.wav",
        "category": "Impacts", "name": "boxing_glove_punch.wav", "duration_seconds": 0.8,
        "description": "Heavy physical body punch impact with muffled mid-low thump.",
    })

    # --- 2. AMBIENCES ---
    # AMB_01: Rain Steady Downpour
    n = int(SR * 5.0)
    t = np.arange(n) / SR
    sig = rng.normal(0, 0.4, n)
    bp = butter(2, [300, 6000], btype="bandpass", fs=SR, output="sos")
    rain = sosfilt(bp, sig)
    # Poisson drops
    drops = rng.poisson(lam=0.002, size=n).astype(np.float32) * rng.normal(0, 1.2, n)
    lp = butter(2, 4000, btype="lowpass", fs=SR, output="sos")
    rain += sosfilt(lp, drops) * 0.5
    p = library_dir / "Ambiences" / "rain_steady_downpour.wav"
    sf.write(p, norm(rain), SR)
    manifest_records.append({
        "file_id": "AMB_01", "relative_path": "Ambiences/rain_steady_downpour.wav",
        "category": "Ambiences", "name": "rain_steady_downpour.wav", "duration_seconds": 5.0,
        "description": "Continuous steady rain downpour texture on hard ground.",
    })

    # AMB_02: Wind Howling Gusts
    n = int(SR * 5.0)
    t = np.arange(n) / SR
    raw = rng.normal(0, 0.5, n)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    sig = np.zeros(n, dtype=np.float32)
    bp1 = butter(2, [250, 650], btype="bandpass", fs=SR, output="sos")
    bp2 = butter(2, [450, 1100], btype="bandpass", fs=SR, output="sos")
    sig = (sosfilt(bp1, raw) * (1 - lfo) + sosfilt(bp2, raw) * lfo) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t))
    p = library_dir / "Ambiences" / "wind_howling_gusts.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "AMB_02", "relative_path": "Ambiences/wind_howling_gusts.wav",
        "category": "Ambiences", "name": "wind_howling_gusts.wav", "duration_seconds": 5.0,
        "description": "Howling wind gusts blowing across an open desolate landscape.",
    })

    # AMB_03: Interior Room Tone
    n = int(SR * 5.0)
    t = np.arange(n) / SR
    hum = 0.15 * np.sin(2 * np.pi * 50 * t) + 0.08 * np.sin(2 * np.pi * 100 * t)
    noise = rng.normal(0, 0.08, n)
    lp = butter(2, 280, btype="lowpass", fs=SR, output="sos")
    sig = hum + sosfilt(lp, noise)
    p = library_dir / "Ambiences" / "interior_room_tone.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "AMB_03", "relative_path": "Ambiences/interior_room_tone.wav",
        "category": "Ambiences", "name": "interior_room_tone.wav", "duration_seconds": 5.0,
        "description": "Quiet indoor room tone with subtle electrical mains hum.",
    })

    # AMB_04: Forest Night Crickets
    n = int(SR * 5.0)
    t = np.arange(n) / SR
    carrier = np.sin(2 * np.pi * 4600 * t)
    mod = np.clip(np.sin(2 * np.pi * 16 * t), 0, 1) ** 3
    bursts = (np.sin(2 * np.pi * 1.5 * t) > 0.2).astype(float)
    crickets = carrier * mod * bursts * 0.6
    p = library_dir / "Ambiences" / "forest_night_crickets.wav"
    sf.write(p, norm(crickets), SR)
    manifest_records.append({
        "file_id": "AMB_04", "relative_path": "Ambiences/forest_night_crickets.wav",
        "category": "Ambiences", "name": "forest_night_crickets.wav", "duration_seconds": 5.0,
        "description": "Outdoor night forest atmosphere with high rhythmic chirping crickets.",
    })

    # AMB_05: Underwater Muffled Drone
    n = int(SR * 4.5)
    t = np.arange(n) / SR
    noise = rng.normal(0, 0.6, n)
    lp = butter(4, 180, btype="lowpass", fs=SR, output="sos")
    sig = sosfilt(lp, noise) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * t))
    p = library_dir / "Ambiences" / "underwater_muffled_drone.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "AMB_05", "relative_path": "Ambiences/underwater_muffled_drone.wav",
        "category": "Ambiences", "name": "underwater_muffled_drone.wav", "duration_seconds": 4.5,
        "description": "Submerged muffled underwater drone atmosphere with gentle aquatic pressure.",
    })

    # AMB_06: Distant Thunder Rolling
    n = int(SR * 6.0)
    t = np.arange(n) / SR
    raw = rng.normal(0, 0.8, n)
    lp = butter(3, 110, btype="lowpass", fs=SR, output="sos")
    rumble = sosfilt(lp, raw)
    env = np.exp(-((t - 1.2) ** 2) / 0.8) + 0.6 * np.exp(-((t - 2.8) ** 2) / 1.5)
    p = library_dir / "Ambiences" / "distant_thunder_rolling.wav"
    sf.write(p, norm(rumble * env), SR)
    manifest_records.append({
        "file_id": "AMB_06", "relative_path": "Ambiences/distant_thunder_rolling.wav",
        "category": "Ambiences", "name": "distant_thunder_rolling.wav", "duration_seconds": 6.0,
        "description": "Low distant thunder rumble rolling across the sky.",
    })

    # --- 3. FOLEY ---
    # FOL_01: Footsteps Gravel Walk
    n = int(SR * 3.5)
    sig = np.zeros(n, dtype=np.float32)
    step_times = [0.3, 1.0, 1.8, 2.6]
    for st in step_times:
        idx = int(st * SR)
        dur = int(0.25 * SR)
        t_step = np.arange(dur) / SR
        step_noise = rng.normal(0, 0.8, dur) * np.exp(-t_step * 20)
        bp = butter(2, [700, 3800], btype="bandpass", fs=SR, output="sos")
        sig[idx:idx + dur] += sosfilt(bp, step_noise)
    p = library_dir / "Foley" / "footsteps_gravel_walk.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "FOL_01", "relative_path": "Foley/footsteps_gravel_walk.wav",
        "category": "Foley", "name": "footsteps_gravel_walk.wav", "duration_seconds": 3.5,
        "description": "Rhythmic human footsteps walking on loose crunching gravel.",
    })

    # FOL_02: Cloth Jacket Rustle
    n = int(SR * 2.5)
    t = np.arange(n) / SR
    raw = rng.normal(0, 0.5, n)
    bp = butter(2, [800, 3200], btype="bandpass", fs=SR, output="sos")
    env = (np.sin(2 * np.pi * 1.2 * t) ** 2) * (np.sin(2 * np.pi * 0.4 * t) ** 2)
    sig = sosfilt(bp, raw) * env
    p = library_dir / "Foley" / "cloth_jacket_rustle.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "FOL_02", "relative_path": "Foley/cloth_jacket_rustle.wav",
        "category": "Foley", "name": "cloth_jacket_rustle.wav", "duration_seconds": 2.5,
        "description": "Soft cloth jacket fabric movement and rustle.",
    })

    # FOL_03: Door Creak Slow
    n = int(SR * 2.8)
    t = np.arange(n) / SR
    pitch = 280 + 140 * np.sin(2 * np.pi * 0.5 * t)
    pulse = (np.sin(2 * np.pi * pitch * t) > 0.4).astype(np.float32)
    grain = (np.sin(2 * np.pi * 45 * t) > 0).astype(np.float32)
    sig = pulse * grain * np.sin(np.pi * t / 2.8)
    p = library_dir / "Foley" / "door_creak_slow.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "FOL_03", "relative_path": "Foley/door_creak_slow.wav",
        "category": "Foley", "name": "door_creak_slow.wav", "duration_seconds": 2.8,
        "description": "High pitched squeaking and creaking wooden door opening slowly.",
    })

    # FOL_04: Paper Page Turn
    n = int(SR * 1.2)
    t = np.arange(n) / SR
    raw = rng.normal(0, 0.7, n)
    bp = butter(2, [1200, 6500], btype="bandpass", fs=SR, output="sos")
    env = np.sin(np.pi * np.clip(t / 0.8, 0, 1)) ** 3
    sig = sosfilt(bp, raw) * env
    p = library_dir / "Foley" / "paper_page_turn.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "FOL_04", "relative_path": "Foley/paper_page_turn.wav",
        "category": "Foley", "name": "paper_page_turn.wav", "duration_seconds": 1.2,
        "description": "Crisp paper sheet page turn flip foley.",
    })

    # FOL_05: Keys Jangling Metal
    n = int(SR * 2.2)
    t = np.arange(n) / SR
    sig = np.zeros(n, dtype=np.float32)
    for k_time in [0.2, 0.45, 0.7, 1.1, 1.4, 1.7]:
        idx = int(k_time * SR)
        dur = int(0.3 * SR)
        t_k = np.arange(dur) / SR
        k_sig = np.zeros(dur, dtype=np.float32)
        for f in [3200, 4800, 6400, 8100]:
            k_sig += np.sin(2 * np.pi * f * t_k) * np.exp(-t_k * 35)
        sig[idx:idx + dur] += k_sig
    p = library_dir / "Foley" / "keys_jangling_metal.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "FOL_05", "relative_path": "Foley/keys_jangling_metal.wav",
        "category": "Foley", "name": "keys_jangling_metal.wav", "duration_seconds": 2.2,
        "description": "Bunch of metal keys jangling and clinking together.",
    })

    # FOL_06: Zipper Fast Slide
    n = int(SR * 0.9)
    t = np.arange(n) / SR
    f_zip = np.linspace(150, 480, n)
    train = np.sin(2 * np.pi * np.cumsum(f_zip) / SR)
    train = (train > 0.5).astype(np.float32)
    sig = train * np.sin(np.pi * t / 0.9)
    p = library_dir / "Foley" / "zipper_fast_slide.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "FOL_06", "relative_path": "Foley/zipper_fast_slide.wav",
        "category": "Foley", "name": "zipper_fast_slide.wav", "duration_seconds": 0.9,
        "description": "Fast metallic zipper zip slide opening.",
    })

    # --- 4. TONES ---
    # TON_01: Pure Sine Tone 440Hz
    n = int(SR * 3.0)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * 440 * t)
    p = library_dir / "Tones" / "pure_sine_tone_440hz.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TON_01", "relative_path": "Tones/pure_sine_tone_440hz.wav",
        "category": "Tones", "name": "pure_sine_tone_440hz.wav", "duration_seconds": 3.0,
        "description": "Continuous pure sine wave tone at concert reference pitch 440 Hz.",
    })

    # TON_02: Binaural Beating Drone
    n = int(SR * 4.0)
    t = np.arange(n) / SR
    sig = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 224 * t)
    p = library_dir / "Tones" / "binaural_beating_drone.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TON_02", "relative_path": "Tones/binaural_beating_drone.wav",
        "category": "Tones", "name": "binaural_beating_drone.wav", "duration_seconds": 4.0,
        "description": "Slow beating acoustic drone with interference pulsation.",
    })

    # TON_03: Distorted Sawtooth Lead
    n = int(SR * 3.0)
    t = np.arange(n) / SR
    saw = 2 * (t * 110 - np.floor(0.5 + t * 110))
    sig = np.tanh(saw * 3.5)
    p = library_dir / "Tones" / "distorted_sawtooth_lead.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TON_03", "relative_path": "Tones/distorted_sawtooth_lead.wav",
        "category": "Tones", "name": "distorted_sawtooth_lead.wav", "duration_seconds": 3.0,
        "description": "Distorted aggressive analog synthesizer sawtooth harmonic lead tone.",
    })

    # TON_04: Church Organ Chord
    n = int(SR * 4.0)
    t = np.arange(n) / SR
    sig = (
        0.35 * np.sin(2 * np.pi * 261.63 * t) +
        0.30 * np.sin(2 * np.pi * 329.63 * t) +
        0.25 * np.sin(2 * np.pi * 392.00 * t) +
        0.20 * np.sin(2 * np.pi * 523.25 * t)
    )
    p = library_dir / "Tones" / "church_organ_chord.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TON_04", "relative_path": "Tones/church_organ_chord.wav",
        "category": "Tones", "name": "church_organ_chord.wav", "duration_seconds": 4.0,
        "description": "Sustained church pipe organ harmonic triad chord.",
    })

    # TON_05: Sub Bass Pulse 60Hz
    n = int(SR * 4.0)
    t = np.arange(n) / SR
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    sig = np.sin(2 * np.pi * 60 * t) * pulse
    p = library_dir / "Tones" / "sub_bass_pulse_60hz.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TON_05", "relative_path": "Tones/sub_bass_pulse_60hz.wav",
        "category": "Tones", "name": "sub_bass_pulse_60hz.wav", "duration_seconds": 4.0,
        "description": "Pulsing deep 60 Hz electronic sub bass tone.",
    })

    # TON_06: High Piercing Sine Beep
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    beep_carrier = np.sin(2 * np.pi * 3500 * t)
    gate = ((t % 0.5) < 0.25).astype(np.float32)
    sig = beep_carrier * gate
    p = library_dir / "Tones" / "high_piercing_sine_beep.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TON_06", "relative_path": "Tones/high_piercing_sine_beep.wav",
        "category": "Tones", "name": "high_piercing_sine_beep.wav", "duration_seconds": 2.0,
        "description": "High pitched piercing electronic sine beep warning alarm.",
    })

    # --- 5. MECHANICAL ---
    # MEC_01: Clock Ticking Steady
    n = int(SR * 4.0)
    sig = np.zeros(n, dtype=np.float32)
    for tick_t in [0.5, 1.5, 2.5, 3.5]:
        idx = int(tick_t * SR)
        dur = int(0.08 * SR)
        t_tk = np.arange(dur) / SR
        click = np.sin(2 * np.pi * 1400 * t_tk) * np.exp(-t_tk * 80)
        sig[idx:idx + dur] += click
    p = library_dir / "Mechanical" / "clock_ticking_steady.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "MEC_01", "relative_path": "Mechanical/clock_ticking_steady.wav",
        "category": "Mechanical", "name": "clock_ticking_steady.wav", "duration_seconds": 4.0,
        "description": "Steady analog wall clock ticking rhythmic mechanism.",
    })

    # MEC_02: Engine Idle Diesel
    n = int(SR * 4.0)
    t = np.arange(n) / SR
    rpm_pulse = (np.sin(2 * np.pi * 14 * t) > 0.7).astype(np.float32)
    lowpass = butter(2, 220, btype="lowpass", fs=SR, output="sos")
    sig = sosfilt(lowpass, rpm_pulse + 0.3 * np.sin(2 * np.pi * 56 * t))
    p = library_dir / "Mechanical" / "engine_idle_diesel.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "MEC_02", "relative_path": "Mechanical/engine_idle_diesel.wav",
        "category": "Mechanical", "name": "engine_idle_diesel.wav", "duration_seconds": 4.0,
        "description": "Heavy vehicle diesel engine idling chug with low rumble.",
    })

    # MEC_03: Camera Shutter Click
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    c1 = np.sin(2 * np.pi * 900 * t) * np.exp(-t * 90)
    c2 = np.zeros(n, dtype=np.float32)
    idx2 = int(0.12 * SR)
    t2 = np.arange(n - idx2) / SR
    c2[idx2:] = np.sin(2 * np.pi * 1500 * t2) * np.exp(-t2 * 80)
    sig = c1 + c2
    p = library_dir / "Mechanical" / "camera_shutter_click.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "MEC_03", "relative_path": "Mechanical/camera_shutter_click.wav",
        "category": "Mechanical", "name": "camera_shutter_click.wav", "duration_seconds": 0.5,
        "description": "Mechanical photo camera shutter double click transient.",
    })

    # MEC_04: Stepper Motor Chirp
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    sig = (
        np.sin(2 * np.pi * 400 * t) * (t < 0.6) +
        np.sin(2 * np.pi * 800 * t) * ((t >= 0.6) & (t < 1.3)) +
        np.sin(2 * np.pi * 1200 * t) * (t >= 1.3)
    )
    p = library_dir / "Mechanical" / "stepper_motor_chirp.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "MEC_04", "relative_path": "Mechanical/stepper_motor_chirp.wav",
        "category": "Mechanical", "name": "stepper_motor_chirp.wav", "duration_seconds": 2.0,
        "description": "Robotic stepper motor 3D printer movement frequency chirps.",
    })

    # MEC_05: Rotary Telephone Dial
    n = int(SR * 2.5)
    t = np.arange(n) / SR
    pulses = (np.sin(2 * np.pi * 22 * t) > 0.6).astype(np.float32)
    burst = rng.normal(0, 0.4, n) * pulses
    bp = butter(2, [1200, 3500], btype="bandpass", fs=SR, output="sos")
    sig = sosfilt(bp, burst)
    p = library_dir / "Mechanical" / "rotary_telephone_dial.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "MEC_05", "relative_path": "Mechanical/rotary_telephone_dial.wav",
        "category": "Mechanical", "name": "rotary_telephone_dial.wav", "duration_seconds": 2.5,
        "description": "Vintage rotary telephone dial wheel spinning and clicking return ratchet.",
    })

    # MEC_06: Robotic Servo Whine
    n = int(SR * 2.2)
    t = np.arange(n) / SR
    freq_whine = 1600 + 1200 * np.sin(np.pi * t / 2.2)
    sig = np.sin(2 * np.pi * np.cumsum(freq_whine) / SR)
    p = library_dir / "Mechanical" / "robotic_servo_whine.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "MEC_06", "relative_path": "Mechanical/robotic_servo_whine.wav",
        "category": "Mechanical", "name": "robotic_servo_whine.wav", "duration_seconds": 2.2,
        "description": "Electronic robotic servo motor actuator high pitched movement whine.",
    })

    # --- 6. TRANSITIONS ---
    # TRN_01: Whoosh Fast Air Pass
    n = int(SR * 2.5)
    t = np.arange(n) / SR
    noise = rng.normal(0, 0.8, n)
    bp = butter(2, [400, 4200], btype="bandpass", fs=SR, output="sos")
    filtered = sosfilt(bp, noise)
    env = np.sin(np.pi * np.clip(t / 2.5, 0, 1)) ** 2.2
    sig = filtered * env
    p = library_dir / "Transitions" / "whoosh_fast_air_pass.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TRN_01", "relative_path": "Transitions/whoosh_fast_air_pass.wav",
        "category": "Transitions", "name": "whoosh_fast_air_pass.wav", "duration_seconds": 2.5,
        "description": "Fast sweeping dynamic air whoosh flyby transition sound.",
    })

    # TRN_02: Riser Tension Pitch Sweep
    n = int(SR * 4.0)
    t = np.arange(n) / SR
    f_rise = 100 * (18 ** (t / 4.0))  # 100 Hz to 1800 Hz
    phase = 2 * np.pi * np.cumsum(f_rise) / SR
    sig = np.sin(phase) * (t / 4.0) ** 1.5
    p = library_dir / "Transitions" / "riser_tension_pitch_sweep.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TRN_02", "relative_path": "Transitions/riser_tension_pitch_sweep.wav",
        "category": "Transitions", "name": "riser_tension_pitch_sweep.wav", "duration_seconds": 4.0,
        "description": "Exponential rising pitch synth riser building tension.",
    })

    # TRN_03: Laser Blaster Downsweep
    n = int(SR * 0.4)
    t = np.arange(n) / SR
    f_laser = np.linspace(3400, 120, n)
    sig = np.sin(2 * np.pi * np.cumsum(f_laser) / SR) * np.exp(-t * 9)
    p = library_dir / "Transitions" / "laser_blaster_downsweep.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TRN_03", "relative_path": "Transitions/laser_blaster_downsweep.wav",
        "category": "Transitions", "name": "laser_blaster_downsweep.wav", "duration_seconds": 0.4,
        "description": "Sci-fi laser blaster weapon beam shot with fast downward pitch chirp.",
    })

    # TRN_04: Reverse Cymbal Swell
    n = int(SR * 3.0)
    t = np.arange(n) / SR
    noise = rng.normal(0, 0.7, n)
    hp = butter(2, 3500, btype="highpass", fs=SR, output="sos")
    cymb = sosfilt(hp, noise)
    env = np.exp(t * 1.8) / np.exp(3.0 * 1.8)
    sig = cymb * env
    p = library_dir / "Transitions" / "reverse_cymbal_swell.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TRN_04", "relative_path": "Transitions/reverse_cymbal_swell.wav",
        "category": "Transitions", "name": "reverse_cymbal_swell.wav", "duration_seconds": 3.0,
        "description": "Reverse metallic cymbal crash swell riser crescendo.",
    })

    # TRN_05: Glitch Digital Stutter
    n = int(SR * 1.8)
    t = np.arange(n) / SR
    grain = np.sin(2 * np.pi * 440 * t) * (t < 0.1)
    sig = np.tile(grain[:int(0.1 * SR)], 18)[:n]
    p = library_dir / "Transitions" / "glitch_digital_stutter.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TRN_05", "relative_path": "Transitions/glitch_digital_stutter.wav",
        "category": "Transitions", "name": "glitch_digital_stutter.wav", "duration_seconds": 1.8,
        "description": "Fast repeating digital buffer stutter glitch transition.",
    })

    # TRN_06: Downshifter Sub Impact
    n = int(SR * 3.5)
    t = np.arange(n) / SR
    f_sub = np.linspace(220, 28, n)
    sig = np.sin(2 * np.pi * np.cumsum(f_sub) / SR) * np.exp(-t * 1.2)
    p = library_dir / "Transitions" / "downshifter_sub_impact.wav"
    sf.write(p, norm(sig), SR)
    manifest_records.append({
        "file_id": "TRN_06", "relative_path": "Transitions/downshifter_sub_impact.wav",
        "category": "Transitions", "name": "downshifter_sub_impact.wav", "duration_seconds": 3.5,
        "description": "Cinematic heavy downward sub impact boom drop.",
    })

    # Write Manifest
    manifest_file = eval_dir / "manifest.json"
    manifest_file.write_text(json.dumps({
        "version": 1,
        "total_files": len(manifest_records),
        "license": "CC0 / Antigravity Synthetic Audio Benchmark",
        "records": manifest_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Define Ground-Truth Queries (20 queries)
    queries = [
        # Material & Impacts
        {"query_id": "Q01", "query_text": "heavy metallic anvil impact with ringing tail", "query_type": "material", "relevant_file_ids": ["IMP_01"]},
        {"query_id": "Q02", "query_text": "solid heavy wooden thud impact", "query_type": "material", "relevant_file_ids": ["IMP_02"]},
        {"query_id": "Q03", "query_text": "sharp glass shatter ping with high resonance", "query_type": "material", "relevant_file_ids": ["IMP_03"]},
        {"query_id": "Q04", "query_text": "deep sub drop bass boom", "query_type": "material", "relevant_file_ids": ["IMP_05", "TRN_06"]},
        {"query_id": "Q05", "query_text": "heavy physical boxing body punch impact", "query_type": "material", "relevant_file_ids": ["IMP_06"]},

        # Ambience & Weather
        {"query_id": "Q06", "query_text": "steady rain downpour texture on hard ground", "query_type": "ambience", "relevant_file_ids": ["AMB_01"]},
        {"query_id": "Q07", "query_text": "howling wind gusts blowing across open landscape", "query_type": "ambience", "relevant_file_ids": ["AMB_02"]},
        {"query_id": "Q08", "query_text": "quiet interior room tone with electrical hum", "query_type": "ambience", "relevant_file_ids": ["AMB_03"]},
        {"query_id": "Q09", "query_text": "outdoor night crickets chirping rhythmically", "query_type": "ambience", "relevant_file_ids": ["AMB_04"]},
        {"query_id": "Q10", "query_text": "low distant rolling thunder rumbling", "query_type": "ambience", "relevant_file_ids": ["AMB_06"]},

        # Foley & Human
        {"query_id": "Q11", "query_text": "footsteps walking on loose crunch gravel", "query_type": "foley", "relevant_file_ids": ["FOL_01"]},
        {"query_id": "Q12", "query_text": "metal keys jangling and clinking together", "query_type": "foley", "relevant_file_ids": ["FOL_05"]},
        {"query_id": "Q13", "query_text": "slow creaking old wooden door opening", "query_type": "foley", "relevant_file_ids": ["FOL_03"]},
        {"query_id": "Q14", "query_text": "crisp paper sheet page turn flip", "query_type": "foley", "relevant_file_ids": ["FOL_04"]},

        # Tones & Harmonics
        {"query_id": "Q15", "query_text": "pure reference 440 Hz sine wave tone", "query_type": "tone", "relevant_file_ids": ["TON_01"]},
        {"query_id": "Q16", "query_text": "pipe organ church harmonic triad chord", "query_type": "tone", "relevant_file_ids": ["TON_04"]},
        {"query_id": "Q17", "query_text": "high pitched piercing electronic beep alarm", "query_type": "tone", "relevant_file_ids": ["TON_06"]},

        # Mechanical
        {"query_id": "Q18", "query_text": "steady clock ticking rhythm mechanism", "query_type": "mechanical", "relevant_file_ids": ["MEC_01"]},
        {"query_id": "Q19", "query_text": "diesel vehicle engine idling low rumble", "query_type": "mechanical", "relevant_file_ids": ["MEC_02"]},
        {"query_id": "Q20", "query_text": "fast air whoosh flyby movement transition", "query_type": "action", "relevant_file_ids": ["TRN_01"]},
    ]

    queries_file = eval_dir / "queries.json"
    queries_file.write_text(json.dumps({
        "version": 1,
        "total_queries": len(queries),
        "queries": queries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated {len(manifest_records)} audio files in {library_dir}")
    print(f"Generated manifest: {manifest_file}")
    print(f"Generated queries: {queries_file}")
    return library_dir, manifest_file, queries_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=".", help="Base project directory")
    args = parser.parse_args()
    make_dataset(Path(args.base_dir).resolve())
