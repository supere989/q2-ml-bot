#!/bin/bash
# compile.sh — compile a generated .map file to a playable .bsp
#
# Usage:
#   ./maps/compile.sh generated/mlmap_00000042.map
#   ./maps/compile.sh --all          # compile every .map in generated/
#   ./maps/compile.sh --batch 8      # generate 8 maps then compile all
#
# Output .bsp files go to q2_lithium_merge/baseq2/maps/ so q2ded can load them.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
Q2ROOT="${Q2ROOT:-/home/raymond/q2_lithium_merge}"
Q2TOOL="$SCRIPT_DIR/q2tools/bin/q2tool"
GENERATED="$SCRIPT_DIR/generated"
INSTALL_DIR="$Q2ROOT/baseq2/maps"
MAP_PREFIX="${MAP_PREFIX:-mltrain}"

_compile_one() {
    local MAP="$1"
    local BASE="${MAP%.map}"
    local NAME="$(basename "$BASE")"
    echo "── compiling $NAME ──"

    "$Q2TOOL" -bsp -vis -fast -rad -bounce 0 -basedir "$Q2ROOT/baseq2" "$MAP" 2>&1 | \
        grep -Ev "WARNING|^$|use_qbsp|moddir|basedir|gamedir|<<<|>>>" | tail -8 || {
        echo "  compile failed"
        return 1
    }

    if [ ! -f "$BASE.bsp" ]; then
        echo "  no bsp produced — skipping install"
        return 1
    fi

    mkdir -p "$INSTALL_DIR"
    cp "$BASE.bsp" "$INSTALL_DIR/$NAME.bsp"
    echo "  installed → $INSTALL_DIR/$NAME.bsp"

    if [ -f "$BASE.json" ]; then
        cp "$BASE.json" "$INSTALL_DIR/$NAME.json"
        echo "  hook zones → $INSTALL_DIR/$NAME.json"
    fi

    for SIDECAR in lattice routes; do
        if [ -f "$BASE.$SIDECAR.json" ]; then
            cp "$BASE.$SIDECAR.json" "$INSTALL_DIR/$NAME.$SIDECAR.json"
            echo "  $SIDECAR lattice → $INSTALL_DIR/$NAME.$SIDECAR.json"
        fi
    done
}

case "${1:-}" in
    --all)
        find "$GENERATED" -name "*.map" | sort | while read -r MAP; do
            _compile_one "$MAP"
        done
        ;;
    --batch)
        N="${2:-4}"
        echo "Generating $N maps with prefix $MAP_PREFIX..."
        python3 "$SCRIPT_DIR/generator.py" --count "$N" --prefix "$MAP_PREFIX"
        find "$GENERATED" -name "${MAP_PREFIX}_*.map" | sort | while read -r MAP; do
            _compile_one "$MAP"
        done
        ;;
    *.map)
        _compile_one "$1"
        ;;
    "")
        echo "Usage: $0 <file.map> | --all | --batch [N]"
        echo ""
        echo "Generates and compiles procedural maps for ML training."
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
esac
