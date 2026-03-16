#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e . --no-deps
python -m brain_subspace_paper bootstrap
