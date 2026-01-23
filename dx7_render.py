"""
DX7 Batch Renderer using NNX-based Faust JAX backend.

Renders DX7 patches from CSV to WAV files using pre-generated algorithm modules.
Uses jax.vmap for high-throughput batched rendering.
"""
from pathlib import Path
import argparse
import importlib
import os

import numpy as np
import jax
from jax import numpy as jnp, random
from flax import nnx
from scipy.io import wavfile
import pandas as pd
from tqdm.auto import trange


# CSV column to NNX label mapping
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
for op in range(1, 7):
    CSV_TO_LABEL.update({
        **{f'op{op}/egL{i}': f'DX7/Operator {op}/Amp Env Generator/Levels/L{i}' for i in range(1, 5)},
        **{f'op{op}/egR{i}': f'DX7/Operator {op}/Amp Env Generator/Rates/R{i}' for i in range(1, 5)},
        f'op{op}/ampModSens': f'DX7/Operator {op}/Level/A Mod Sens',
        f'op{op}/touchSensitivity': f'DX7/Operator {op}/Level/Key Vel',
        f'op{op}/totalLev': f'DX7/Operator {op}/Level/Level',
        f'op{op}/rateScaling': f'DX7/Operator {op}/Level/Rate Scaling',
        f'op{op}/breakPoint': f'DX7/Operator {op}/Breakpoint/Breakpoint',
        f'op{op}/leftDepth': f'DX7/Operator {op}/Breakpoint/L Depth',
        f'op{op}/rightDepth': f'DX7/Operator {op}/Breakpoint/R Depth',
        f'op{op}/leftCurve': f'DX7/Operator {op}/Breakpoint/L Curve',
        f'op{op}/rightCurve': f'DX7/Operator {op}/Breakpoint/R Curve',
        f'op{op}/detune': f'DX7/Operator {op}/Tone/Tune',
        f'op{op}/freqCoarse': f'DX7/Operator {op}/Tone/Coarse',
        f'op{op}/freqFine': f'DX7/Operator {op}/Tone/Fine',
        f'op{op}/freqMode': f'DX7/Operator {op}/Tone/Freq Mode',
    })


def note_to_hz(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def create_inputs(note: int, note_samples: int, total_samples: int) -> jnp.ndarray:
    """Create input tensor (freq, gain, gate) for DX7 synth."""
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


def stack_params(rows: list[dict], model) -> dict:
    """Stack parameters from multiple rows into batched param dict for vmap."""
    params_list = [
        model.with_defaults(model.params_from_labels(csv_row_to_labels(row)))
        for row in rows
    ]
    return {
        key: jnp.stack([p[key] for p in params_list])
        for key in params_list[0]
    }


def dx7_render(patches_df, algorithm: int, batch_size: int, sample_rate: int,
               note: int, note_dur: float, record_dur: float,
               do_normalize: bool, output_directory, limit: int = None):
    """Render DX7 patches for a specific algorithm."""
    # Filter by algorithm (0-indexed in CSV, 1-indexed in Faust)
    df = patches_df[patches_df["global/algorithm"] == algorithm]
    df = df.drop(columns=["filename", "name", "global/algorithm"])

    if df.empty:
        return

    # Limit number of presets if specified
    if limit is not None and limit < len(df):
        df = df.iloc[:limit]

    algo_num = algorithm + 1
    try:
        algo_module = importlib.import_module(f'algo{algo_num}')
        ModelClass = getattr(algo_module, f'DX7Algo{algo_num}')
    except (ImportError, AttributeError) as e:
        print(f"Skipping algorithm {algo_num}: {e}")
        return

    # Create model
    model = ModelClass(sample_rate=sample_rate, rngs=nnx.Rngs(0, params=0, rng_stream=0))
    model.eval()

    # Create inputs
    inputs = create_inputs(note, int(sample_rate * note_dur), int(sample_rate * record_dur))

    # Batched render function
    def render_single(params, rng_key):
        return model(inputs, params=params, rngs=rng_key)

    batched_render = jax.jit(jax.vmap(render_single))

    # Setup output
    output_dir = Path(output_directory) / f'algo{algo_num}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle and pad to batch_size multiple
    df = df.sample(frac=1, ignore_index=True)
    total_presets = len(df)
    if len(df) % batch_size != 0:
        padding = batch_size - (len(df) % batch_size)
        df = pd.concat([df, df.iloc[:padding]], ignore_index=True)

    # Process batches
    count = 0
    rng_key = random.key(42)

    for batch_start in trange(0, len(df), batch_size, desc=f"Algo {algo_num}"):
        batch_rows = [row.to_dict() for _, row in df.iloc[batch_start:batch_start + batch_size].iterrows()]
        batch_params = stack_params(batch_rows, model)

        rng_key, *batch_keys = random.split(rng_key, batch_size + 1)
        audio_batch = np.array(batched_render(batch_params, jnp.stack(batch_keys)))

        for i in range(batch_size):
            if count >= total_presets:
                break
            audio = audio_batch[i].T
            if do_normalize and (peak := np.abs(audio).max()) > 0:
                audio = audio / peak
            wavfile.write(str(output_dir / f"render_{count:05d}.wav"), sample_rate, audio.astype(np.float32))
            count += 1


def main():
    parser = argparse.ArgumentParser(description="Render DX7 presets with JAX.")
    parser.add_argument("--csv", default="dx7_patches.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("-sr", "--sample-rate", type=int, default=44100)
    parser.add_argument("--note-dur", type=float, default=2)
    parser.add_argument("--record-dur", type=float, default=3)
    parser.add_argument("--note", type=int, default=48)
    parser.add_argument("--normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--platform", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("-o", "--output-directory", default="output_dx7")
    parser.add_argument("--algorithm", type=int, default=None, help="Algorithm 1-32 (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of presets to render per algorithm")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", args.platform)

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    if args.algorithm is not None:
        if not 1 <= args.algorithm <= 32:
            raise ValueError("Algorithm must be 1-32")
        dx7_render(df, args.algorithm - 1, args.batch_size, args.sample_rate,
                   args.note, args.note_dur, args.record_dur, args.normalize, args.output_directory, args.limit)
    else:
        for algo_idx in trange(32, desc="Algorithm"):
            dx7_render(df, algo_idx, args.batch_size, args.sample_rate,
                       args.note, args.note_dur, args.record_dur, args.normalize, args.output_directory, args.limit)

    print("All done!")


if __name__ == "__main__":
    main()
