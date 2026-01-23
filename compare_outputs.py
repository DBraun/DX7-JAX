"""
Compare DX7-JAX outputs to dexed-py outputs.

Renders the same patches with both synthesizers and compares similarity.
"""
from pathlib import Path
import argparse
import importlib
import os
import sys

import numpy as np
import jax
from jax import numpy as jnp, random
from flax import nnx
from scipy.io import wavfile
import pandas as pd
from tqdm.auto import tqdm

# dexed-py
import dexed_py


# CSV column to NNX label mapping (copied from dx7_render.py)
CSV_TO_LABEL = {
    # Global LFO
    'global/lfoAMD': 'DX7/Global/LFO/AMD',
    'global/lfoDelay': 'DX7/Global/LFO/Delay',
    'global/lfoPMD': 'DX7/Global/LFO/PMD',
    'global/lfoPitchModSens': 'DX7/Global/LFO/P Mod Sens',
    'global/lfoSpeed': 'DX7/Global/LFO/Speed',
    'global/lfoSync': 'DX7/Global/LFO/Sync',
    'global/lfoWave': 'DX7/Global/LFO/Wave',
    # Global Main
    'global/feedback': 'DX7/Global/Main/Feedback',
    'global/oscKeySync': 'DX7/Global/Main/Osc Key Sync',
    'global/transpose': 'DX7/Global/Main/Transpose',
    # Global Pitch EG
    **{f'global/pegL{i}': f'DX7/Global/Pitch EG Levels/L{i}' for i in range(1, 5)},
    **{f'global/pegR{i}': f'DX7/Global/Pitch EG Rates/R{i}' for i in range(1, 5)},
}

# Add operator parameters for operators 1-6
for _op in range(1, 7):
    CSV_TO_LABEL.update({
        **{f'op{_op}/egL{i}': f'DX7/Operator {_op}/Amp Env Generator/Levels/L{i}' for i in range(1, 5)},
        **{f'op{_op}/egR{i}': f'DX7/Operator {_op}/Amp Env Generator/Rates/R{i}' for i in range(1, 5)},
        f'op{_op}/ampModSens': f'DX7/Operator {_op}/Level/A Mod Sens',
        f'op{_op}/touchSensitivity': f'DX7/Operator {_op}/Level/Key Vel',
        f'op{_op}/totalLev': f'DX7/Operator {_op}/Level/Level',
        f'op{_op}/rateScaling': f'DX7/Operator {_op}/Level/Rate Scaling',
        f'op{_op}/breakPoint': f'DX7/Operator {_op}/Breakpoint/Breakpoint',
        f'op{_op}/leftDepth': f'DX7/Operator {_op}/Breakpoint/L Depth',
        f'op{_op}/rightDepth': f'DX7/Operator {_op}/Breakpoint/R Depth',
        f'op{_op}/leftCurve': f'DX7/Operator {_op}/Breakpoint/L Curve',
        f'op{_op}/rightCurve': f'DX7/Operator {_op}/Breakpoint/R Curve',
        f'op{_op}/detune': f'DX7/Operator {_op}/Tone/Tune',
        f'op{_op}/freqCoarse': f'DX7/Operator {_op}/Tone/Coarse',
        f'op{_op}/freqFine': f'DX7/Operator {_op}/Tone/Fine',
        f'op{_op}/freqMode': f'DX7/Operator {_op}/Tone/Freq Mode',
    })


def note_to_hz(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def create_jax_inputs(note: int, note_samples: int, total_samples: int) -> jnp.ndarray:
    """Create input tensor (freq, gain, gate) for DX7 JAX synth."""
    inputs = jnp.zeros((3, total_samples))
    inputs = inputs.at[0, :].set(note_to_hz(note))  # freq
    inputs = inputs.at[1, :].set(1.0)               # gain
    inputs = inputs.at[2, :note_samples].set(1.0)   # gate
    return inputs


def csv_row_to_labels(row: dict) -> dict:
    """Convert CSV row to dict keyed by NNX labels."""
    return {
        CSV_TO_LABEL[col]: float(row[col])
        for col in CSV_TO_LABEL
        if col in row
    }


def csv_row_to_dexed_array(row: dict) -> np.ndarray:
    """Convert CSV row to dexed-py 185-element normalized array.

    Based on dexed_py/patch.py Patch.to_array() format:
    - Indices 0-130: continuous parameters [0,1]
    - Indices 131-136: LFO wave one-hot (6 classes)
    - Indices 137-184: operator curves one-hot (6 ops * 2 * 4 = 48)
    """
    params = np.zeros(185, dtype=np.float32)

    # Helper to get value with default
    def get(key, default=0):
        return row.get(key, default)

    # Global LFO (indices 0-5)
    params[0] = get('global/lfoAMD') / 99.0         # AMD
    params[1] = get('global/lfoDelay') / 99.0       # Delay
    params[2] = get('global/lfoPitchModSens') / 7.0 # P Mod Sens
    params[3] = get('global/lfoPMD') / 99.0         # PMD
    params[4] = get('global/lfoSpeed') / 99.0       # Speed
    params[5] = get('global/lfoSync')               # Sync (0 or 1)

    # Global Main (indices 6-8)
    params[6] = get('global/feedback') / 7.0        # Feedback
    params[7] = get('global/oscKeySync')            # Osc Key Sync (0 or 1)
    # Transpose: CSV has -24 to 24, dexed expects 0-48 (24 = center)
    transpose = get('global/transpose', 0) + 24
    params[8] = transpose / 48.0                    # Transpose

    # Pitch EG (indices 9-16)
    for i in range(4):
        params[9 + i] = get(f'global/pegL{i+1}') / 99.0   # Levels
        params[13 + i] = get(f'global/pegR{i+1}') / 99.0  # Rates

    # Per-operator parameters (6 ops, 19 continuous each = 114 params)
    for op_idx in range(6):
        op = op_idx + 1  # 1-indexed
        base = 17 + op_idx * 19

        # Envelope levels (L1-L4)
        for i in range(4):
            params[base + i] = get(f'op{op}/egL{i+1}') / 99.0

        # Envelope rates (R1-R4)
        for i in range(4):
            params[base + 4 + i] = get(f'op{op}/egR{i+1}') / 99.0

        # Other continuous params
        params[base + 8] = get(f'op{op}/breakPoint') / 99.0
        params[base + 9] = get(f'op{op}/leftDepth') / 99.0
        params[base + 10] = get(f'op{op}/rightDepth') / 99.0
        params[base + 11] = get(f'op{op}/ampModSens') / 3.0
        params[base + 12] = get(f'op{op}/freqMode')  # 0 or 1
        params[base + 13] = get(f'op{op}/touchSensitivity') / 7.0  # velocity sens
        params[base + 14] = get(f'op{op}/totalLev') / 99.0  # output level
        params[base + 15] = get(f'op{op}/rateScaling') / 7.0
        params[base + 16] = get(f'op{op}/freqCoarse') / 31.0
        params[base + 17] = get(f'op{op}/freqFine') / 99.0
        # Detune: CSV has -7 to 7, dexed expects 0-14 (7 = center)
        detune = get(f'op{op}/detune', 0) + 7
        params[base + 18] = detune / 14.0

    # Categorical parameters (one-hot encoded)
    cat_idx = 131

    # LFO Wave (6 classes)
    lfo_wave = int(get('global/lfoWave'))
    params[cat_idx + lfo_wave] = 1.0
    cat_idx += 6

    # Operator curves (6 ops * 2 curves * 4 classes = 48 params)
    for op_idx in range(6):
        op = op_idx + 1
        # Left curve (4 classes)
        left_curve = int(get(f'op{op}/leftCurve'))
        params[cat_idx + left_curve] = 1.0
        cat_idx += 4
        # Right curve (4 classes)
        right_curve = int(get(f'op{op}/rightCurve'))
        params[cat_idx + right_curve] = 1.0
        cat_idx += 4

    return params


def compute_metrics(audio_jax: np.ndarray, audio_dexed: np.ndarray) -> dict:
    """Compute similarity metrics between two audio signals."""
    # Ensure same length (truncate to shorter)
    min_len = min(len(audio_jax), len(audio_dexed))
    a1 = audio_jax[:min_len].astype(np.float64)
    a2 = audio_dexed[:min_len].astype(np.float64)

    # RMS error
    rms_error = np.sqrt(np.mean((a1 - a2) ** 2))

    # Peak error
    peak_error = np.max(np.abs(a1 - a2))

    # Correlation coefficient
    if np.std(a1) > 1e-10 and np.std(a2) > 1e-10:
        correlation = np.corrcoef(a1, a2)[0, 1]
    else:
        correlation = 0.0

    # Signal-to-noise ratio (treating dexed as reference)
    signal_power = np.mean(a2 ** 2)
    noise_power = np.mean((a1 - a2) ** 2)
    if noise_power > 1e-10:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')

    # Max amplitude of each
    peak_jax = np.max(np.abs(a1))
    peak_dexed = np.max(np.abs(a2))

    return {
        'rms_error': rms_error,
        'peak_error': peak_error,
        'correlation': correlation,
        'snr_db': snr_db,
        'peak_jax': peak_jax,
        'peak_dexed': peak_dexed,
    }


def compare_outputs(patches_df, algorithm: int, sample_rate: int,
                    note: int, note_dur: float, record_dur: float,
                    limit: int, output_directory: str, save_audio: bool):
    """Compare outputs for a specific algorithm."""

    # Filter by algorithm (0-indexed in CSV, 1-indexed in Faust/dexed)
    df = patches_df[patches_df["global/algorithm"] == algorithm]
    algo_num = algorithm + 1

    if df.empty:
        print(f"No patches found for algorithm {algo_num}")
        return []

    # Limit number of patches
    if limit is not None and limit < len(df):
        df = df.iloc[:limit]

    # Try to import JAX algorithm module
    try:
        algo_module = importlib.import_module(f'algo{algo_num}')
        ModelClass = getattr(algo_module, f'DX7Algo{algo_num}')
    except (ImportError, AttributeError) as e:
        print(f"Cannot load JAX module for algorithm {algo_num}: {e}")
        return []

    # Create JAX model
    jax_model = ModelClass(sample_rate=sample_rate, rngs=nnx.Rngs(0, params=0, rng_stream=0))
    jax_model.eval()

    # Create dexed synth
    dexed_synth = dexed_py.DexedSynth(sample_rate=float(sample_rate), algorithm=algo_num)

    # Create JAX inputs
    note_samples = int(sample_rate * note_dur)
    total_samples = int(sample_rate * record_dur)
    jax_inputs = create_jax_inputs(note, note_samples, total_samples)

    # Setup output directory
    if save_audio:
        output_dir = Path(output_directory) / f'algo{algo_num}'
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Algo {algo_num}")):
        row_dict = row.to_dict()

        # Render with JAX
        label_params = csv_row_to_labels(row_dict)
        jax_params = jax_model.with_defaults(jax_model.params_from_labels(label_params))
        jax_audio = np.array(jax_model(jax_inputs, params=jax_params))
        jax_audio = jax_audio.flatten()  # mono

        # Render with dexed
        dexed_params = csv_row_to_dexed_array(row_dict)
        dexed_synth.load_params(dexed_params, algorithm=algo_num)
        dexed_audio = dexed_synth.render(
            midi_note=note,
            velocity=100,
            note_duration=note_dur,
            render_duration=record_dur
        )

        # Compute metrics
        metrics = compute_metrics(jax_audio, dexed_audio)
        metrics['patch_idx'] = idx
        metrics['patch_name'] = row_dict.get('name', f'patch_{idx}')
        results.append(metrics)

        # Save audio if requested
        if save_audio:
            wavfile.write(str(output_dir / f"jax_{idx:05d}.wav"), sample_rate, jax_audio.astype(np.float32))
            wavfile.write(str(output_dir / f"dexed_{idx:05d}.wav"), sample_rate, dexed_audio.astype(np.float32))

    return results


def print_summary(results: list):
    """Print summary statistics."""
    if not results:
        return

    rms_errors = [r['rms_error'] for r in results]
    correlations = [r['correlation'] for r in results]
    snrs = [r['snr_db'] for r in results if r['snr_db'] != float('inf')]

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Total patches compared: {len(results)}")
    print()
    print("RMS Error:")
    print(f"  Mean: {np.mean(rms_errors):.6f}")
    print(f"  Std:  {np.std(rms_errors):.6f}")
    print(f"  Min:  {np.min(rms_errors):.6f}")
    print(f"  Max:  {np.max(rms_errors):.6f}")
    print()
    print("Correlation:")
    print(f"  Mean: {np.mean(correlations):.4f}")
    print(f"  Std:  {np.std(correlations):.4f}")
    print(f"  Min:  {np.min(correlations):.4f}")
    print(f"  Max:  {np.max(correlations):.4f}")
    if snrs:
        print()
        print("SNR (dB):")
        print(f"  Mean: {np.mean(snrs):.2f}")
        print(f"  Std:  {np.std(snrs):.2f}")
        print(f"  Min:  {np.min(snrs):.2f}")
        print(f"  Max:  {np.max(snrs):.2f}")
    print()

    # Show worst matches
    print("Worst 5 matches (by correlation):")
    sorted_results = sorted(results, key=lambda x: x['correlation'])
    for r in sorted_results[:5]:
        print(f"  {r['patch_name']}: corr={r['correlation']:.4f}, rms={r['rms_error']:.6f}")

    # Show best matches
    print()
    print("Best 5 matches (by correlation):")
    for r in sorted_results[-5:]:
        print(f"  {r['patch_name']}: corr={r['correlation']:.4f}, rms={r['rms_error']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare DX7-JAX outputs to dexed-py outputs.")
    parser.add_argument("--csv", default="dx7_patches.csv", help="Path to patches CSV")
    parser.add_argument("-sr", "--sample-rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--note-dur", type=float, default=2.0, help="Note duration in seconds")
    parser.add_argument("--record-dur", type=float, default=3.0, help="Recording duration in seconds")
    parser.add_argument("--note", type=int, default=48, help="MIDI note number")
    parser.add_argument("--platform", default="cpu", choices=["cpu", "gpu"], help="JAX platform")
    parser.add_argument("-o", "--output-directory", default="comparison_output", help="Output directory for audio")
    parser.add_argument("--algorithm", type=int, default=16, help="Algorithm 1-32 (default: 16)")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of patches to compare")
    parser.add_argument("--save-audio", action="store_true", help="Save audio files for inspection")
    parser.add_argument("--csv-output", type=str, default=None, help="Save results to CSV file")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", args.platform)

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    if not 1 <= args.algorithm <= 32:
        raise ValueError("Algorithm must be 1-32")

    print(f"Comparing outputs for algorithm {args.algorithm}")
    print(f"  Sample rate: {args.sample_rate}")
    print(f"  Note: MIDI {args.note} ({note_to_hz(args.note):.2f} Hz)")
    print(f"  Duration: {args.note_dur}s note, {args.record_dur}s total")
    print(f"  Limit: {args.limit} patches")

    results = compare_outputs(
        df,
        args.algorithm - 1,  # 0-indexed
        args.sample_rate,
        args.note,
        args.note_dur,
        args.record_dur,
        args.limit,
        args.output_directory,
        args.save_audio
    )

    if results:
        print_summary(results)

        if args.csv_output:
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.csv_output, index=False)
            print(f"\nResults saved to {args.csv_output}")
    else:
        print("No results to compare.")


if __name__ == "__main__":
    main()
