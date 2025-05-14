import struct
import os.path
import os
import argparse
from pathlib import Path
import pandas as pd

# References:

# https://web.archive.org/web/20010605031928/http://www.yamaha.com/ycaservice/pdf/553.pdf

# DX7s manual is very similar to DX7
# https://homepages.abdn.ac.uk/d.j.benson/pages/dx7/manuals/dx7s-man.pdf
# Go to section 5-4 "Voice Memory Format" page 101/108

# https://github.com/asb2m10/dexed/blob/master/Documentation/sysex-format.txt
# which is basically right, but I think SCL_LEFT_CURVE and SCL_RIGHT_CURVE are swapped incorrectly


def parse0to99(b):
    """
    Normalize a 7-bit byte to the 0-99 range, matching Dexed's `normparm` behavior.
    """
    b = b & 0b1111111
    if b <= 99:
        return b
    # Dexed maps out-of-range values proportionally: (value/255)*99, then truncated
    return int((b / 255.0) * 99)


# Structure to store operator-specific parameters
class Dx7Operator:

    def __repr__(self):
        return str(self.__dict__)

    def __init__(self, data):
        expected_length = 17
        if len(data) != expected_length:
            raise ValueError(f"Expected {expected_length} bytes of data, but got {len(data)}")

        # Unpack the data, excluding the bitfields for now
        op_params = struct.unpack('B' * expected_length, data)

        # Assign the unpacked values to the corresponding class attributes
        self.EG_R1 = parse0to99(op_params[0])
        self.EG_R2 = parse0to99(op_params[1])
        self.EG_R3 = parse0to99(op_params[2])
        self.EG_R4 = parse0to99(op_params[3])
        self.EG_L1 = parse0to99(op_params[4])
        self.EG_L2 = parse0to99(op_params[5])
        self.EG_L3 = parse0to99(op_params[6])
        self.EG_L4 = parse0to99(op_params[7])
        self.BREAK_POINT = parse0to99(op_params[8])
        self.LEFT_DEPTH = parse0to99(op_params[9])
        self.RIGHT_DEPTH = parse0to99(op_params[10])

        # Process the bitfields for SCL_LEFT_CURVE and SCL_RIGHT_CURVE
        self.LEFT_CURVE = op_params[11] & 0b11  # (0-3) (-LIN, -EXP, +EXP, +LIN)
        self.RIGHT_CURVE = (op_params[11] >> 2) & 0b11  # (0-3) (-LIN, -EXP, +EXP, +LIN)

        # Process the bitfields for OSC_RATE_SCALE and OSC_DETUNE
        self.RATE_SCALING = op_params[12] & 0b111  # (0-7)
        self.DETUNE = min(14, (op_params[12] >> 3) & 0b1111)  # (0-14)

        # Process the bitfields for AMP_MOD_SENS and TOUCH_SENSITIVITY
        self.AMP_MOD_SENS = op_params[13] & 0b11  # (0-3)
        self.TOUCH_SENSITIVITY = (op_params[13] >> 2) & 0b111  # (0-7)

        # Assign TOTAL_LEV
        self.TOTAL_LEV = parse0to99(op_params[14])

        # Process the bitfields for FREQ_MODE and FREQ_COARSE
        self.FREQ_MODE = op_params[15] & 0b1  # (0-1) (ratio, fixed)
        self.FREQ_COARSE = (op_params[15] >> 1) & 0b11111  # (0-31)

        # Assign FREQ_FINE
        self.FREQ_FINE = parse0to99(op_params[16])


# Structure to store global parameters
class Dx7Global:

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def format_name(name):
        """
        Function to get rid of problematic characters in a string
        """
        return ''.join(c for c in name.decode('ascii', 'ignore') if c.isalnum())

    def __init__(self, data, name_data):
        expected_length = 16
        if len(data) != expected_length:
            raise ValueError(f"Expected {expected_length} bytes of data, but got {len(data)}")

        # Unpack the data, excluding the bitfields for now
        global_params = struct.unpack('B' * expected_length, data)

        # Assign the unpacked values to the corresponding class attributes
        self.PITCH_EG_R1 = parse0to99(global_params[0])
        self.PITCH_EG_R2 = parse0to99(global_params[1])
        self.PITCH_EG_R3 = parse0to99(global_params[2])
        self.PITCH_EG_R4 = parse0to99(global_params[3])
        self.PITCH_EG_L1 = parse0to99(global_params[4])
        self.PITCH_EG_L2 = parse0to99(global_params[5])
        self.PITCH_EG_L3 = parse0to99(global_params[6])
        self.PITCH_EG_L4 = parse0to99(global_params[7])

        self.ALGORITHM = global_params[8] & 0b11111  # (0-31)

        self.FEEDBACK = global_params[9] & 0b111  # (0-7)
        self.OSC_KEY_SYNC = (global_params[9] >> 3) & 0b1  # (0-1)

        self.LFO_SPEED = parse0to99(global_params[10])
        self.LFO_DELAY = parse0to99(global_params[11])
        self.LF_PT_MOD_DEP = parse0to99(global_params[12])
        self.LF_AM_MOD_DEP = parse0to99(global_params[13])

        # SYNC is the least significant bit of the 14th byte
        self.SYNC = global_params[14] & 0b1  # (0-1)
        # WAVE is the next 3 bits of the 14th byte. The 6 options are
        # (triangle, saw down, saw up, square, sine, sample&hold)
        self.WAVE = min(5, (global_params[14] >> 1) & 0b111)  # (0-5)
        # LF_PT_MOD_SNS is the most significant 4 bits of the 14th byte
        self.LF_PT_MOD_SNS = (global_params[14] >> 4) & 0b111  # (0-7)

        self.TRANSPOSE = min(48, global_params[15] & 0b111111)  # (0-48)

        self.NAME = self.format_name(name_data)  # note: it may be an empty string


def convert_patch(filename, rel_path):
    def get_operator_d(i: int, operator: Dx7Operator, name: str):

        return {
            f'op{i}/ampModSens': operator.AMP_MOD_SENS,
            f'op{i}/egL1': operator.EG_L1,
            f'op{i}/egL2': operator.EG_L2,
            f'op{i}/egL3': operator.EG_L3,
            f'op{i}/egL4': operator.EG_L4,
            f'op{i}/egR1': operator.EG_R1,
            f'op{i}/egR2': operator.EG_R2,
            f'op{i}/egR3': operator.EG_R3,
            f'op{i}/egR4': operator.EG_R4,

            f'op{i}/breakPoint': operator.BREAK_POINT,
            f'op{i}/leftDepth': operator.LEFT_DEPTH,
            f'op{i}/rightDepth': operator.RIGHT_DEPTH,

            f'op{i}/leftCurve': operator.LEFT_CURVE,
            f'op{i}/rightCurve': operator.RIGHT_CURVE,

            f'op{i}/touchSensitivity': operator.TOUCH_SENSITIVITY,
            f'op{i}/totalLev': operator.TOTAL_LEV,
            f'op{i}/detune': operator.DETUNE - 7,  # note the minus 7
            f'op{i}/freqCoarse': operator.FREQ_COARSE,
            f'op{i}/freqFine': operator.FREQ_FINE,
            f'op{i}/freqMode': operator.FREQ_MODE,
            f'op{i}/rateScaling': operator.RATE_SCALING,
        }

    assert os.path.isfile(filename)

    with open(filename, 'rb') as file:
        header = file.read(6)
        n_voices = 32  # preset file always contain 32 voices

        instruments = []

        # Parsing presets and converting them into Faust functions
        for count in range(n_voices):
            # Read operator data for each of the 6 operators
            operators_data = [file.read(17) for _ in range(6)]

            # According to the DX7 manual, the operators are stored in reverse order.
            operators_data.reverse()

            # Check if any operator data block is shorter than expected
            if any(len(data) != 17 for data in operators_data):
                # maybe the file doesn't have 32 voices
                # todo: show warning
                break

            operators = [Dx7Operator(data) for data in operators_data]

            global_params_data = file.read(16)
            if len(global_params_data) != 16:
                raise ValueError(f"Incomplete global parameters at voice {count + 1}")

            name_data = file.read(10)
            global_params = Dx7Global(global_params_data, name_data)

            d = {
                'filename': rel_path,
                'name': global_params.NAME,
                'global/algorithm': global_params.ALGORITHM,  # [0-31]
                'global/feedback': global_params.FEEDBACK,
                'global/transpose': global_params.TRANSPOSE - 24,  # note the minus 24

                'global/lfoDelay': global_params.LFO_DELAY,
                'global/lfoPMD': global_params.LF_PT_MOD_DEP,
                'global/lfoAMD': global_params.LF_AM_MOD_DEP,
                'global/lfoSpeed': global_params.LFO_SPEED,
                'global/lfoSync': global_params.SYNC,
                'global/lfoWave': global_params.WAVE,
                'global/lfoPitchModSens': global_params.LF_PT_MOD_SNS,

                'global/pegR1': global_params.PITCH_EG_R1,
                'global/pegR2': global_params.PITCH_EG_R2,
                'global/pegR3': global_params.PITCH_EG_R3,
                'global/pegR4': global_params.PITCH_EG_R4,
                'global/pegL1': global_params.PITCH_EG_L1,
                'global/pegL2': global_params.PITCH_EG_L2,
                'global/pegL3': global_params.PITCH_EG_L3,
                'global/pegL4': global_params.PITCH_EG_L4,

                'global/oscKeySync': global_params.OSC_KEY_SYNC,
            }

            for i, operator in enumerate(operators):
                d.update(get_operator_d(i + 1, operator, global_params.NAME))

            instruments.append(d)

        return instruments


def main(directory, output_path):

    # Check if the given directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")

    # List to hold all instrument data
    all_instruments_data = []

    # Traverse the directory and process each .syx file
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.syx'):
                syx_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(syx_file_path, directory)

                try:
                    instruments_data = convert_patch(syx_file_path, rel_path)
                except ValueError as e:
                    print(f'Skipped file {syx_file_path}. {e}')
                all_instruments_data.extend(instruments_data)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_instruments_data)
    print('size before dropping duplicates: ', df.shape[0])
    dedup_columns = list(df.columns)
    dedup_columns.remove('filename')
    dedup_columns.remove('name')
    df = df.drop_duplicates(subset=dedup_columns)
    print('size after dropping duplicates: ', df.shape[0])
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Saved data to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Turn a directory of Yamaha DX7 .syx files into a CSV.')
    parser.add_argument('--directory', default='dx7_patches', type=str, help='Directory of .syx files')
    parser.add_argument('-o', '--output', default='dx7_patches.csv', type=str, help='Output CSV path')

    args = parser.parse_args()

    directory = Path(args.directory).absolute()

    main(directory, args.output)
