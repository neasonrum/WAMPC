#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_EXECUTABLE:-python3}" "${PROJECT_ROOT}/evaluation/bench2drive.py" "$@"