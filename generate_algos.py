"""
Generate all 32 DX7 algorithm Python modules using DawDreamer.

NOTE: This script requires DawDreamer to be updated with:
1. The new Faust DX7 library (dx.algorithm)
2. The NNX-based JAX backend (replacing Linen)

Until then, use generate_algos.sh with the command-line Faust compiler.
"""
import argparse
from pathlib import Path

from dawdreamer.faust import FaustContext
from dawdreamer.faust.box import boxFromDSP, boxToSource


def generate_algorithm(algo_num: int, output_dir: Path) -> None:
    """
    Generate a single DX7 algorithm module.

    Args:
        algo_num: Algorithm number (1-32)
        output_dir: Directory to write the output file
    """
    dsp_code = f"""
import("stdfaust.lib");
replace = !,_;
process = ["freq":replace, "gain":replace, "gate":replace -> dx.algorithm({algo_num})] <: _;
"""

    class_name = f"DX7Algo{algo_num}"
    output_file = output_dir / f"algo{algo_num}.py"

    with FaustContext():
        box = boxFromDSP(dsp_code)
        jax_code = boxToSource(box, 'jax', class_name, ['-a', 'jax/minimal.py'])

    output_file.write_text(jax_code)
    print(f"Generated {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DX7 algorithm modules using DawDreamer"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for generated files (default: current directory)"
    )
    parser.add_argument(
        "--algorithm",
        type=int,
        default=None,
        help="Generate only a specific algorithm (1-32). If not set, generates all."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.algorithm is not None:
        if not (1 <= args.algorithm <= 32):
            raise ValueError("Algorithm must be between 1 and 32")
        generate_algorithm(args.algorithm, args.output_dir)
    else:
        for algo_num in range(1, 33):
            generate_algorithm(algo_num, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
