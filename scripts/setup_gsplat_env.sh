#!/usr/bin/env bash
# One-time setup of the "gsplat" conda env, matching this project's
# per-tool environment isolation (README.md "Setup": indoor3d / colmap /
# depth kept apart so their dependencies don't collide -- gsplat gets the
# same treatment rather than going into any of those three).
#
# Idempotent: safe to re-run; each step is skipped if already done.
#
# Two install paths, because two different classes of package need
# different handling:
#
#   1. Pure-Python / prebuilt-wheel deps (pycolmap, viser, opencv, ...) --
#      installed with plain `pip install`.
#   2. CUDA source extensions that compile at install time (fused-ssim,
#      fused-bilagrid, ppisp, and gsplat itself) -- these need nvcc *and*
#      MSVC's cl.exe on PATH, and must NOT let pip build them in an
#      isolated venv (that would silently pull a second, CPU-only torch
#      just to read build-system.requires). So they run with
#      `--no-build-isolation` inside a shell that has first sourced
#      vcvars64.bat and set CUDA_HOME, via cmd.exe (vcvars64.bat is a
#      .bat file; bash can't source it directly).
#
# GPU note (see README.md): the RTX 5070 Ti is sm_120 (Blackwell). Stock
# torch wheels from PyPI (cu118/cu124) enumerate the device but fail at
# kernel launch, so torch/torchvision are pulled from a CUDA-13 wheel index
# instead (see step 2 below for why cu130, not cu128), pinned to the
# versions examples/requirements.txt expects.
#
# Prereqs this script assumes are already installed (both were present
# when this was written): a CUDA toolkit under
# "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<version>", and
# VS Build Tools with vcvars64.bat. Override via CUDA_HOME / VCVARS64 if
# your versions differ, or see ~/work/gsplat/docs/INSTALL_WIN.md.

set -euo pipefail

GSPLAT_REPO="${GSPLAT_REPO:-$HOME/work/gsplat}"
CONDA_ENV="${GSPLAT_CONDA_ENV:-gsplat}"
PY_VERSION="${GSPLAT_PY_VERSION:-3.11}"
TORCH_VERSION="${GSPLAT_TORCH_VERSION:-2.9.1}"       # must match examples/requirements.txt's pin
TORCHVISION_VERSION="${GSPLAT_TORCHVISION_VERSION:-0.24.1}"
# sm_120 (Blackwell, the RTX 5070 Ti) needs CUDA >=12.8's nvcc -- the
# machine's v12.4 toolkit doesn't recognize the architecture ("nvcc fatal:
# Value 'sm_120' is not defined for option 'gpu-architecture'"), so this
# defaults to the newer v13.3 toolkit that's also installed.
CUDA_HOME="${CUDA_HOME:-C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3}"
VCVARS64="${VCVARS64:-C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat}"

CONDA_BASE="$(conda info --base)"
ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV"
PYTHON="$ENV_DIR/python.exe"

echo "== gsplat env setup =="
echo "repo:        $GSPLAT_REPO"
echo "conda env:   $CONDA_ENV ($ENV_DIR)"
echo "torch:       ${TORCH_VERSION}+cu130 / torchvision ${TORCHVISION_VERSION}+cu130"
echo "CUDA_HOME:   $CUDA_HOME"
echo "vcvars64:    $VCVARS64"
echo

[[ -d "$GSPLAT_REPO" ]] || { echo "error: gsplat repo not found at $GSPLAT_REPO" >&2; exit 1; }
[[ -f "$VCVARS64" ]] || { echo "error: vcvars64.bat not found at $VCVARS64 (set VCVARS64)" >&2; exit 1; }
[[ -d "$CUDA_HOME" ]] || { echo "error: CUDA toolkit not found at $CUDA_HOME (set CUDA_HOME)" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Conda env
# ---------------------------------------------------------------------------

if [[ ! -x "$PYTHON" ]]; then
    conda create -n "$CONDA_ENV" python="$PY_VERSION" -y
fi

# ---------------------------------------------------------------------------
# 2. torch / torchvision, pinned + matched to the sm_120 GPU (cu130 index)
# ---------------------------------------------------------------------------

# cu128 (CUDA 12.8) wheels would seem like the obvious match for the GPU
# note above, but torch's own build step later enforces CUDA_HOME's nvcc
# major version == torch's own CUDA major version (hard error, not just a
# warning, in torch.utils.cpp_extension._check_cuda_version). The two CUDA
# toolkits on this machine are v12.4 (nvcc doesn't know sm_120 at all --
# Blackwell support landed in CUDA 12.6+) and v13.3 (nvcc knows sm_120, but
# major-version-mismatches a cu128 torch). So: cu130 wheels + the v13.3
# toolkit, same exact torch/torchvision version, just a different CUDA build.
if ! "$PYTHON" -c "import torch; assert torch.__version__.startswith('${TORCH_VERSION}+cu130')" >/dev/null 2>&1; then
    "$PYTHON" -m pip install --force-reinstall --no-deps \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu130
fi

# ---------------------------------------------------------------------------
# 3. ninja (needed by both the AOT build below and gsplat's lazy JIT fallback)
# ---------------------------------------------------------------------------

"$PYTHON" -c "import ninja" >/dev/null 2>&1 || "$PYTHON" -m pip install ninja
"$PYTHON" -c "import rich" >/dev/null 2>&1 || "$PYTHON" -m pip install rich

# ---------------------------------------------------------------------------
# 4. Pure-Python / prebuilt-wheel example deps
# ---------------------------------------------------------------------------

PURE_DEPS_REQ="$(mktemp)"
grep -v -E "^torch==|^torchvision==|fused-ssim|fused-bilagrid|ppisp" \
    "$GSPLAT_REPO/examples/requirements.txt" > "$PURE_DEPS_REQ"
"$PYTHON" -m pip install -r "$PURE_DEPS_REQ"
rm -f "$PURE_DEPS_REQ"

# ---------------------------------------------------------------------------
# 5. CUDA source extensions: fused-ssim, fused-bilagrid, ppisp, gsplat itself
#    -- built with vcvars64.bat's environment active, via cmd.exe.
# ---------------------------------------------------------------------------

CUDA_BUILD_BAT="$(mktemp --suffix=.bat)"
cat > "$CUDA_BUILD_BAT" <<BATEOF
@echo off
call "$VCVARS64" || exit /b 1
set "CUDA_HOME=$CUDA_HOME"
rem Without this, distutils re-detects/re-activates the VC env on top of
rem vcvars64.bat's, and torch's cpp_extension build aborts with:
rem "It seems that the VC environment is activated but DISTUTILS_USE_SDK
rem is not set."
set "DISTUTILS_USE_SDK=1"
rem CUDA 13's CCCL headers require MSVC's conforming preprocessor. gsplat's
rem own build.py passes /Zc:preprocessor explicitly, but the third-party
rem extensions below (fused-ssim, fused-bilagrid) don't, and fail with
rem "fatal error C1189: MSVC/cl.exe with traditional preprocessor is used."
rem The CL env var is cl.exe's own mechanism for flags applied to every
rem invocation, including the ones nvcc launches internally for host code.
set "CL=/Zc:preprocessor"
rem gsplat/cuda/build.py locates CCCL 3.0's relocated headers (moved under
rem a cccl/ subdir as of CUDA 13.0) via CUDA_HOME/targets/<arch>/include/cccl
rem -- a Linux toolkit layout. Windows installs have no targets/ directory at
rem all; the headers are directly under CUDA_HOME\include\cccl instead, so
rem that detection silently finds nothing here and cuda/std/* includes
rem 404 ("Cannot open include file: 'cuda/std/limits'"). INCLUDE is cl.exe's
rem (and nvcc's, via --use-local-env) own env var for extra header dirs.
set "INCLUDE=%CUDA_HOME%\include\cccl;%INCLUDE%"
rem This machine has 12 cores but only 16GB RAM, often already under
rem pressure from other GPU/WSL work (see hand_off.md). Ninja's default
rem parallelism (one job per core) runs enough cl.exe/nvcc instances at
rem once to exhaust RAM on gsplat's larger template-heavy files ("fatal
rem error C1060: compiler is out of heap space"), so cap it.
set "MAX_JOBS=3"
set "PIP=$(cygpath -w "$PYTHON") -m pip"
%PIP% install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38 || exit /b 1
%PIP% install --no-build-isolation git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16 || exit /b 1
%PIP% install --no-build-isolation "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.2.1" || exit /b 1
%PIP% install --no-build-isolation -e "$(cygpath -w "$GSPLAT_REPO")" || exit /b 1
BATEOF

cmd //c "$(cygpath -w "$CUDA_BUILD_BAT")"
status=$?
rm -f "$CUDA_BUILD_BAT"
[[ $status -eq 0 ]] || { echo "error: CUDA extension build failed (see output above)." >&2; exit 1; }

echo
echo "== done =="
echo "Verify with: scripts/run_gsplat_iphone.sh --check-env"
