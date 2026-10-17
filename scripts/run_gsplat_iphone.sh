#!/usr/bin/env bash
# Phase 5 (surface/texture precursor): train 3D Gaussian Splats on a
# RawRecorder iPhone capture, using the refined COLMAP poses from
# scripts/run_colmap_raw.py (see hand_off.md, "Beware" note under Layout --
# `sparse/refined/` is the useful model out of the several COLMAP experiments
# that live under `<session>/colmap/sparse/`).
#
# gsplat itself lives in a separate checkout (~/work/gsplat) with its own
# conda env, matching this project's per-tool environment isolation
# (README.md "Setup": indoor3d / colmap / depth kept apart so their
# dependencies don't collide). This script does not create that env -- see
# ~/work/gsplat/docs/INSTALL_WIN.md. Run with --check-env to see what's
# missing without doing anything else.
#
# Usage:
#   scripts/run_gsplat_iphone.sh [SESSION_DIR] [-- EXTRA_ARGS...]
#   scripts/run_gsplat_iphone.sh --check-env
#
# SESSION_DIR defaults to the one iPhone capture this project has so far.
# EXTRA_ARGS are forwarded to simple_trainer.py, e.g.:
#   scripts/run_gsplat_iphone.sh -- --max_steps 30000
#
# Env overrides:
#   GSPLAT_REPO       checkout of nerfstudio-project/gsplat (default: ~/work/gsplat)
#   GSPLAT_CONDA_ENV  conda env with gsplat + its example deps (default: gsplat)
#   GSPLAT_STRATEGY   tyro subcommand: default | mcmc (default: default)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SESSION_DIR="${1:-$HOME/data/iphone/RawRecording_2026-08-18_215941}"
[[ "${1:-}" == "--check-env" ]] && { CHECK_ONLY=1; SESSION_DIR="$HOME/data/iphone/RawRecording_2026-08-18_215941"; } || CHECK_ONLY=0
if [[ $# -gt 0 && "$1" != "--check-env" ]]; then shift; fi

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

GSPLAT_REPO="${GSPLAT_REPO:-$HOME/work/gsplat}"
CONDA_ENV="${GSPLAT_CONDA_ENV:-gsplat}"
STRATEGY="${GSPLAT_STRATEGY:-default}"

COLMAP_DIR="$SESSION_DIR/colmap"
SPARSE_MODEL="$COLMAP_DIR/sparse/refined/0"   # the refined, bundle-adjusted model
IMAGES_DIR="$COLMAP_DIR/images"

# gsplat's Parser wants <data_dir>/sparse/0 and <data_dir>/images specifically;
# our data has the reconstruction under sparse/refined/0 instead, alongside
# several other experimental models. Rather than duplicate ~1440p x 539 images
# and a 160k-point model, wrap the two paths it needs in NTFS junctions (no
# admin rights required, unlike symlinks) and point --data_dir there.
READY_DIR="$COLMAP_DIR/gsplat_ready"

# Keep generated artifacts out of the source tree: results live next to the
# recording under the data folder, not in this repo.
SESSION_NAME="$(basename "$SESSION_DIR")"
RESULT_DIR="$SESSION_DIR/gsplat_results/$SESSION_NAME"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

fail() { echo "error: $*" >&2; exit 1; }

check_env() {
    local ok=1
    if ! conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
        echo "missing: conda env '$CONDA_ENV' does not exist."
        echo "  Run scripts/setup_gsplat_env.sh to build it (see that script for why"
        echo "  cu130 wheels, not cu128 -- torch's build-time CUDA-version check needs"
        echo "  its major version to match the local nvcc, and only the sm_120-capable"
        echo "  toolkit on this machine is v13.3)."
        ok=0
    else
        if ! conda run -n "$CONDA_ENV" python -c "import gsplat, pycolmap, cv2, imageio, tqdm" >/dev/null 2>&1; then
            echo "incomplete: conda env '$CONDA_ENV' exists but is missing gsplat and/or its example deps"
            echo "  (gsplat, pycolmap, opencv-python, imageio, tqdm). See docs/INSTALL_WIN.md."
            ok=0
        fi
    fi
    [[ $ok -eq 1 ]] && echo "ok: conda env '$CONDA_ENV' has gsplat + example deps."
    return $((1 - ok))
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    check_env
    exit $?
fi

[[ -d "$GSPLAT_REPO" ]] || fail "gsplat repo not found at $GSPLAT_REPO (set GSPLAT_REPO)."
[[ -d "$SPARSE_MODEL" ]] || fail "refined COLMAP model not found at $SPARSE_MODEL -- run scripts/run_colmap_raw.py first."
[[ -d "$IMAGES_DIR" ]] || fail "images not found at $IMAGES_DIR."
check_env || fail "gsplat env is not ready (see messages above)."

# ---------------------------------------------------------------------------
# Wire up the data_dir gsplat expects (idempotent)
# ---------------------------------------------------------------------------

mkdir -p "$READY_DIR/sparse"

link_junction() {
    local link="$1" target="$2"
    if [[ -e "$link" ]]; then
        return 0
    fi
    MSYS_NO_PATHCONV=1 cmd /c mklink /J "$(cygpath -w "$link")" "$(cygpath -w "$target")" >/dev/null \
        || fail "could not create junction $link -> $target"
}

link_junction "$READY_DIR/sparse/0" "$SPARSE_MODEL"
link_junction "$READY_DIR/images" "$IMAGES_DIR"

mkdir -p "$RESULT_DIR"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

echo "data_dir:   $READY_DIR"
echo "result_dir: $RESULT_DIR"
echo "strategy:   $STRATEGY"

conda run --no-capture-output -n "$CONDA_ENV" python "$GSPLAT_REPO/examples/simple_trainer.py" \
    "$STRATEGY" \
    --data_dir "$(cygpath -w "$READY_DIR")" \
    --result_dir "$(cygpath -w "$RESULT_DIR")" \
    --data_factor 1 \
    "${EXTRA_ARGS[@]}"
