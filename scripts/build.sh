#!/bin/bash
# Build step: run the BioCypher pipeline in the mounted repository (/src).
# Writes biocypher-out/build2neo/ and biocypher-log/ into the repository.
# The uv cache and the container's virtualenv are kept in .cache/ so reruns are fast.
set -euo pipefail
cd /src

export HOME=/tmp
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/src/.cache/uv
export UV_PROJECT_ENVIRONMENT=/src/.cache/docker-venv
export PYTHONPATH=/src

pip install --user --quiet --disable-pip-version-check uv
# --frozen: install exactly what uv.lock specifies, never rewrite it
uv sync --frozen --python 3.11 --extra momapy_sbml_kinetic
# CLEAN_OUTPUT=1 (default): replace output from earlier runs (--clean);
# CLEAN_OUTPUT=0: add to it, e.g. when building from several scripts
if [ "${CLEAN_OUTPUT:-1}" = "1" ]; then
  uv run --no-sync python "$PIPELINE_SCRIPT" --clean
else
  uv run --no-sync python "$PIPELINE_SCRIPT"
fi
