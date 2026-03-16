#!/usr/bin/env bash
set -euo pipefail

python -m brain_subspace_paper build-features "$@"
