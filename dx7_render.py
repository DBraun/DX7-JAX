from functools import partial
from pathlib import Path
import timeit
import argparse
import os
import math

import numpy as np

import jax.numpy as jnp
import jax
from jax import random

import flax.linen as nn

from scipy.io import wavfile

import pandas as pd

from tqdm.auto import trange

from dawdreamer.faust.box import boxFromDSP, boxToSource
from dawdreamer.faust import FaustContext


def note_number_to_hz(note_number):
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))


def note_to_tensor(note, gain, note_dur, total_dur):
    # Return 2D tensor shaped (3, total_dur) where
    # 1st dimension is (freq, gain, gate)
    # and 2nd dimension is audio sample
    tensor = jnp.zeros((3, total_dur))

    # set freq and gain
    freq = note_number_to_hz(note)
    tensor = tensor.at[:2, :].set(
        jnp.array([freq, gain]).reshape(2, 1))

    # set gate
    tensor = tensor.at[2, :note_dur].set(jnp.array([1]))
    return tensor


def transform_k_v(k, v):
    if 'Levels/' in k or 'Rates/' in k:
        v = jnp.interp(v, jnp.array([0, 99]), jnp.array([-1., 1.]))
    elif 'LFO/AMD' in k or 'LFO/PMD' in k or 'LFO/Delay' in k or 'LFO/Speed' in k:
        v = jnp.interp(v, jnp.array([0, 99]), jnp.array([-1., 1.]))
    elif 'LFO/Pitch Mod Sens' in k:
        v = jnp.interp(v, jnp.array([0, 7]), jnp.array([-1., 1.]))
    elif 'LFO/Sync' in k:
        v = jnp.interp(v, jnp.array([0, 1]), jnp.array([-1., 1.]))
    elif 'LFO/Wave' in k:
        v = nn.one_hot(v, num_classes=6)
    elif 'Feedback' in k:
        v = jnp.interp(v, jnp.array([0, 7]), jnp.array([-1., 1.]))
    elif 'Osc Key Sync' in k:
        v = jnp.interp(v, jnp.array([0, 1]), jnp.array([-1., 1.]))
    elif 'Transpose' in k:
        v = jnp.interp(v, jnp.array([-24, 24]), jnp.array([-1., 1.]))

    # non global / Operator (1-6) parameters
    elif 'AmpModSens' in k:
        v = jnp.interp(v, jnp.array([0, 3]), jnp.array([-1., 1.]))
    elif 'KeyVelSens' in k:
        v = jnp.interp(v, jnp.array([0, 7]), jnp.array([-1., 1.]))
    elif 'Other/Level' in k:
        v = jnp.interp(v, jnp.array([0, 99]), jnp.array([-1., 1.]))
    elif 'Other/Freq Mode' in k:
        v = jnp.interp(v, jnp.array([0, 1]), jnp.array([-1., 1.]))
    elif 'Other/Rate Scale' in k:
        v = jnp.interp(v, jnp.array([0, 7]), jnp.array([-1., 1.]))

    elif 'Break Point/Break Point' in k or 'Break Point/Left Depth' in k or 'Break Point/Right Depth' in k:
        v = jnp.interp(v, jnp.array([0, 99]), jnp.array([-1., 1.]))
    elif 'Break Point/Left Curve' in k or 'Break Point/Right Curve' in k:
        v = nn.one_hot(v, num_classes=4)

    elif 'Tone/Detune' in k:
        v = jnp.interp(v, jnp.array([0, 14]), jnp.array([-1., 1.]))
    elif 'Tone/Freq Coarse' in k:
        v = nn.one_hot(v, num_classes=32)
    elif 'Tone/Freq Fine' in k:
        v = nn.one_hot(v, num_classes=100)
    else:
        raise ValueError(f'Unrecognized key {k}')
    return v


def dx7_render(patches_df, algorithm: int, batch_size: int, sample_rate: int, note: int, note_dur: float,
               record_dur: float, do_normalize: bool, output_directory):

    # Note: algorithm is in range [0-31]

    # Note: It is possible to render different algorithms within the same batch. It might make sense to do this
    # if you only have a few presets to render, and they use different algorithms.
    # We assume we'll render a lot of presets, so we compile a different JAX module for each algorithm.
    df = patches_df[patches_df["global/algorithm"] == algorithm]

    df = df.drop(columns=["filename", "name"])
    df = df.drop(columns=["global/algorithm"])

    if not df.shape[0]:
        return
    
    rename_mapper = {
        'global/lfoAMD': '_DX7/Global/LFO/AMD',
        'global/lfoDelay': '_DX7/Global/LFO/Delay',
        'global/lfoPMD': '_DX7/Global/LFO/PMD',
        'global/lfoPitchModSens': '_DX7/Global/LFO/Pitch Mod Sens',
        'global/lfoSpeed': '_DX7/Global/LFO/Speed',
        'global/lfoSync': '_DX7/Global/LFO/Sync',
        'global/lfoWave': '_DX7/Global/LFO/Wave',

        'global/algorithm': '_DX7/Global/Main/Algorithm',
        'global/feedback': '_DX7/Global/Main/Feedback',
        'global/oscKeySync': '_DX7/Global/Main/Osc Key Sync',
        'global/transpose': '_DX7/Global/Main/Transpose',

        'global/pegL1': '_DX7/Global/Pitch Env Generator Levels/L1',
        'global/pegL2': '_DX7/Global/Pitch Env Generator Levels/L2',
        'global/pegL3': '_DX7/Global/Pitch Env Generator Levels/L3',
        'global/pegL4': '_DX7/Global/Pitch Env Generator Levels/L4',
        'global/pegR1': '_DX7/Global/Pitch Env Generator Rates/R1',
        'global/pegR2': '_DX7/Global/Pitch Env Generator Rates/R2',
        'global/pegR3': '_DX7/Global/Pitch Env Generator Rates/R3',
        'global/pegR4': '_DX7/Global/Pitch Env Generator Rates/R4',
    }

    for i in range(1, 7):
        rename_mapper.update(
            {
                f'op{i}/egL1': f'_DX7/Operator {i}/Amp Env Generator/Levels/L1',
                f'op{i}/egL2': f'_DX7/Operator {i}/Amp Env Generator/Levels/L2',
                f'op{i}/egL3': f'_DX7/Operator {i}/Amp Env Generator/Levels/L3',
                f'op{i}/egL4': f'_DX7/Operator {i}/Amp Env Generator/Levels/L4',

                f'op{i}/ampModSens': f'_DX7/Operator {i}/Amp Env Generator/Other/AmpModSens',
                f'op{i}/touchSensitivity': f'_DX7/Operator {i}/Amp Env Generator/Other/KeyVelSens',
                f'op{i}/totalLev': f'_DX7/Operator {i}/Amp Env Generator/Other/Level',
                f'op{i}/freqMode': f'_DX7/Operator {i}/Amp Env Generator/Other/Freq Mode',
                f'op{i}/rateScaling': f'_DX7/Operator {i}/Amp Env Generator/Other/Rate Scale',

                f'op{i}/egR1': f'_DX7/Operator {i}/Amp Env Generator/Rates/R1',
                f'op{i}/egR2': f'_DX7/Operator {i}/Amp Env Generator/Rates/R2',
                f'op{i}/egR3': f'_DX7/Operator {i}/Amp Env Generator/Rates/R3',
                f'op{i}/egR4': f'_DX7/Operator {i}/Amp Env Generator/Rates/R4',

                f'op{i}/breakPoint': f'_DX7/Operator {i}/Break Point/Break Point',
                f'op{i}/leftCurve': f'_DX7/Operator {i}/Break Point/Left Curve',
                f'op{i}/leftDepth': f'_DX7/Operator {i}/Break Point/Left Depth',
                f'op{i}/rightCurve': f'_DX7/Operator {i}/Break Point/Right Curve',
                f'op{i}/rightDepth': f'_DX7/Operator {i}/Break Point/Right Depth',

                f'op{i}/detune': f'_DX7/Operator {i}/Tone/Detune',
                f'op{i}/freqCoarse': f'_DX7/Operator {i}/Tone/Freq Coarse',
                f'op{i}/freqFine': f'_DX7/Operator {i}/Tone/Freq Fine',
            }
        )

    df.rename(columns=rename_mapper, inplace=True)

    # convert from seconds to samples
    record_dur = int(sample_rate*record_dur)
    note_dur = int(sample_rate*note_dur)

    tensor = note_to_tensor(note, 1, note_dur, record_dur)
    tensor = jnp.expand_dims(tensor, axis=0)

    tensor = jnp.tile(tensor, (batch_size, 1, 1))

    with FaustContext():

        dsp_content = f"""
        custom_dx7 = library("custom_dx7.lib");

        replace = !,_;
        process = ["freq":replace, "gain":replace, "gate":replace -> custom_dx7.dx7_algorithm({algorithm+1})] <: _, _;
        """

        box = boxFromDSP(dsp_content)

        module_name = 'FaustVoice'
        jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

    custom_globals = {}
    exec(jax_code, custom_globals)  # security risk!
    MonoVoice = custom_globals[module_name]

    # split parameters but don't split RNGs (DX7 doesn't use RNGs anyway).
    BatchedVoice = nn.vmap(MonoVoice, in_axes=(0, None), variable_axes={'params': 0}, split_rngs={'params': False})

    batched_model = BatchedVoice(sample_rate=sample_rate)

    # optional shuffle
    df = df.sample(frac=1, ignore_index=True)

    # pad df with additional rows to match batch size evenly
    assert df.shape[0]
    total_presets = df.shape[0]
    while df.shape[0] % batch_size != 0:
        df = pd.concat([df, df[:batch_size-(total_presets % batch_size)]])

    jit_inference_fn = jax.jit(batched_model.apply, static_argnums=[2])

    # timeit_number = 3
    # key = random.PRNGKey(0)
    # key, subkey = random.split(key)
    # params = batched_model.init({'params': subkey}, tensor, DURATION)['params']
    # result = timeit.timeit("jit_inference_fn({'params': params}, tensor, DURATION)", number=timeit_number,
    #                        globals=locals()) / timeit_number
    # print(f"Time it: {result:.3f} seconds average.")

    output_directory = Path(output_directory) / f'algo{algorithm+1}'
    os.makedirs(output_directory, exist_ok=True)

    num_batches = math.ceil(total_presets/batch_size)

    count = 0
    for batch_i in trange(num_batches, desc="Items Loop"):

        params = df[batch_i*batch_size:(batch_i+1)*batch_size].to_dict(orient='list')
        params = {k: transform_k_v(k, jnp.array(v)) for k, v in params.items()}

        audio = jit_inference_fn({'params': params}, tensor, record_dur)
        audio = np.array(audio)

        for i in range(batch_size):

            if count < total_presets:

                output_audio = audio[i].T

                if do_normalize:
                    # normalize peak level
                    output_audio = output_audio / np.abs(output_audio).max()

                output_path = str(output_directory / f"render_{str(count).zfill(5)}.wav")

                wavfile.write(output_path, sample_rate, output_audio)

                count += 1


def main():

    parser = argparse.ArgumentParser(description="Render many presets of a DX7 synthesizer with JAX.")
    parser.add_argument("--csv", default="dx7_patches.csv", help="Path to CSV file.")
    parser.add_argument("--batch-size", default=32, help="Batch size")
    parser.add_argument( "-sr", "--sample-rate", default=44100, help="Sample rate")
    parser.add_argument("--note-dur", default=2, help="Note on duration (seconds).")
    parser.add_argument("--record-dur", default=3, help="Recording duration (seconds).")
    parser.add_argument("--note", default=48, help="MIDI note to play.")
    parser.add_argument("--normalize", default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle for normalizing peak level.")
    parser.add_argument('--platform', default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument("-o", "--output-directory", default="output_dx7",
                        help="Output directory for saved files.")
    args = parser.parse_args()

    # Global flag to set a specific platform, must be used at startup.
    jax.config.update('jax_platform_name', args.platform)

    patches_csv = args.csv
    assert os.path.isfile(patches_csv), FileExistsError(f"CSV of patches not found at path: {patches_csv}")

    df = pd.read_csv(patches_csv)

    for algorithm in trange(32, desc="Algorithm"):

        dx7_render(df, algorithm, args.batch_size, args.sample_rate, args.note, args.note_dur, args.record_dur,
                   args.normalize, args.output_directory)

    print('All done!')


if __name__ == "__main__":

    main()
