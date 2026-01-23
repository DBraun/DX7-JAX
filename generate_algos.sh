#!/bin/bash
#
# Generate all 32 DX7 algorithm Python modules from Faust
#
# Prerequisites:
#   - Faust compiler built at ~/GitHub/faust/build/bin/faust
#   - Faust libraries at ~/GitHub/faust/libraries
#   - DX7 library at ~/GitHub/faust/libraries/dx7/
#
# Usage:
#   ./generate_algos.sh
#

set -e

FAUST_BIN="${FAUST_BIN:-$HOME/GitHub/faust/build/bin/faust}"
FAUST_LIBS="${FAUST_LIBS:-$HOME/GitHub/faust/libraries}"
FAUST_ARCH="${FAUST_ARCH:-$HOME/GitHub/faust/architecture/jax/minimal.py}"

# Check if faust compiler exists
if [ ! -f "$FAUST_BIN" ]; then
    echo "Error: Faust compiler not found at $FAUST_BIN"
    echo "Set FAUST_BIN environment variable to your faust binary location"
    exit 1
fi

echo "Using Faust compiler: $FAUST_BIN"
echo "Using libraries from: $FAUST_LIBS"
echo "Using architecture: $FAUST_ARCH"
echo ""

for i in {1..32}; do
    echo "Generating algorithm $i..."

    # Create DSP file
    cat > "algo${i}.dsp" << EOF
import("stdfaust.lib");
replace = !,_;
process = ["freq":replace, "gain":replace, "gate":replace -> dx.algorithm($i)] <: _;
EOF

    # Generate Python file
    "$FAUST_BIN" "algo${i}.dsp" \
        -I "$FAUST_LIBS" \
        -I "$FAUST_LIBS/dx7/" \
        -lang jax \
        -a "$FAUST_ARCH" \
        -cn "DX7Algo${i}" \
        -o "algo${i}.py"

    # Clean up DSP file (optional - uncomment to keep them)
    # rm "algo${i}.dsp"
done

echo ""
echo "Done! Generated algo1.py through algo32.py"
