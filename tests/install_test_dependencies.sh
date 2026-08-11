#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script installs Python dependencies required to run CV-CUDA tests.
# When running inside a Docker devel image, all dependencies are pre-installed
# and this script is effectively a no-op.
#
# Usage:
#   install_test_dependencies.sh [numpy1|numpy2] [cu12|cu13]
#
# Arguments:
#   numpy1  - Install NumPy 1.x (Python 3.10-3.12 only)
#   numpy2  - Install NumPy 2.x (Python 3.10-3.14, default)
#   cu12    - Install CuPy/CUDA-Python for CUDA 12.x (compatible with CUDA 12.2+)
#   cu13    - Install CuPy/CUDA-Python for CUDA 13.x (compatible with CUDA 13.3+)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEPARATOR_LINE="======================================="

NUMPY_MAJOR="${1:-numpy2}"
TORCH_CUDA="${2:-}"

echo "$SEPARATOR_LINE"
echo "CV-CUDA Test Dependency Installer"
echo "$SEPARATOR_LINE"
echo ""

# Validate numpy argument
case "$NUMPY_MAJOR" in
    numpy1|numpy2) ;;
    *)
        echo "Error: Invalid first argument '$NUMPY_MAJOR'. Use 'numpy1' or 'numpy2'." >&2
        exit 1
        ;;
esac

# Validate torch CUDA suffix if provided
if [[ -n "$TORCH_CUDA" ]]; then
    case "$TORCH_CUDA" in
        cu12|cu13) ;;
        *)
            echo "Error: Invalid second argument '$TORCH_CUDA'. Use 'cu12' (CUDA 12.2+) or 'cu13' (CUDA 13.x)." >&2
            exit 1
            ;;
    esac
fi

CUDA_REQUIREMENTS=""
if [[ -n "$TORCH_CUDA" ]]; then
    if [[ "$NUMPY_MAJOR" == "numpy1" ]]; then
        if [[ "$TORCH_CUDA" != "cu12" ]]; then
            echo "Error: NumPy 1 test dependencies are supported only with cu12." >&2
            exit 1
        fi
        CUDA_REQUIREMENTS="$SCRIPT_DIR/requirements.tests.cu12.numpy1.txt"
    else
        CUDA_REQUIREMENTS="$SCRIPT_DIR/requirements.tests.${TORCH_CUDA}.txt"
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    echo "Error: pip is not available. Please install python3-pip." >&2
    exit 1
fi

# Check if we're in a virtual environment (recommended)
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Note: Not running in a virtual environment."
    echo "For isolated installations, consider using:"
    echo "  python3 -m venv venv_tests"
    echo "  source venv_tests/bin/activate"
    echo ""
fi

# Check that required requirements files have been generated.
REQ_FILES=(
    "$SCRIPT_DIR/requirements.tests.common.txt"
    "$SCRIPT_DIR/requirements.tests.${NUMPY_MAJOR}.txt"
)
[[ -n "$CUDA_REQUIREMENTS" ]] && REQ_FILES+=("$CUDA_REQUIREMENTS")

MISSING=()
for f in "${REQ_FILES[@]}"; do
    [[ -f "$f" ]] || MISSING+=("$f")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Error: the following requirements files are missing:" >&2
    for f in "${MISSING[@]}"; do echo "  $f" >&2; done
    echo "Generate them first by running from the repository root:" >&2
    echo "  bash generate_requirements.sh" >&2
    exit 1
fi

echo "Installing test dependencies..."
python3 -m pip install -r "$SCRIPT_DIR/requirements.tests.common.txt"

echo ""
echo "Installing NumPy ($NUMPY_MAJOR)..."
python3 -m pip install -r "$SCRIPT_DIR/requirements.tests.${NUMPY_MAJOR}.txt"

if [[ -n "$CUDA_REQUIREMENTS" ]]; then
    echo ""
    echo "Installing CuPy/CUDA-Python ($TORCH_CUDA)..."
    python3 -m pip install -r "$CUDA_REQUIREMENTS"
fi

echo ""
echo "$SEPARATOR_LINE"
echo "Test dependency installation complete!"
echo "$SEPARATOR_LINE"
echo ""
if [[ -z "$TORCH_CUDA" ]]; then
    echo "Note: CuPy/CUDA-Python was not installed. To install it, re-run with a CUDA suffix:"
    echo "  $0 $NUMPY_MAJOR cu12    # for CUDA 12.2+"
    if [[ "$NUMPY_MAJOR" == "numpy2" ]]; then
        echo "  $0 $NUMPY_MAJOR cu13    # for CUDA 13.x (13.3+)"
    fi
    echo ""
fi
echo "To run tests, build the project with -DBUILD_TESTS=1 then:"
echo "  build-rel/bin/run_tests.sh"
echo ""
