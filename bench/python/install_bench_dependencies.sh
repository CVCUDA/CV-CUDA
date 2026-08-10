#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script installs dependencies for CV-CUDA nvbench-based Python benchmarks.
# All packages (including cuda-bench from PyPI) are declared in requirements.bench.cu{12,13}.txt.
# When running inside a Docker devel image, all dependencies are pre-installed and
# this script is effectively a no-op.
#
# Usage:
#   install_bench_dependencies.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEPARATOR_LINE="======================================="

echo "$SEPARATOR_LINE"
echo "CV-CUDA Benchmark Dependency Installer"
echo "$SEPARATOR_LINE"
echo ""

# Check Python version compatibility
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Detected Python version: $PYTHON_VERSION"

# Validate Python version (nvbench requires Python 3.10+)
if [[ "$PYTHON_MINOR" -lt 10 ]]; then
    echo "Error: Python $PYTHON_VERSION is not compatible with nvbench (requires Python 3.10+)" >&2
    echo "Please use Python 3.10 or higher" >&2
    exit 1
fi

echo "✓ Python version compatible"
echo ""

# Check CUDA version. Begin by checking if nvcc command exists.
if command -v nvcc >/dev/null 2>&1; then
    # Get CUDA version from nvcc output
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')

    # Extract major version number
    CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d. -f1)

    echo "Detected CUDA version: $CUDA_VERSION (major: $CUDA_MAJOR_VERSION)"

    # Check major version to determine CUDA version
    if [[ "$CUDA_MAJOR_VERSION" -eq 12 ]] || [[ "$CUDA_MAJOR_VERSION" -eq 13 ]]; then
        echo "✓ CUDA $CUDA_MAJOR_VERSION is supported"
    else
        echo "Warning: CUDA $CUDA_MAJOR_VERSION may not be fully supported (expecting CUDA 12 or 13)"
        echo "Proceeding anyway..."
    fi
else
    echo "Error: CUDA is not installed or nvcc is not in PATH" >&2
    echo "Please install CUDA Toolkit 12 or 13" >&2
    exit 1
fi

echo ""

# Check for CUPTI library (required by nvbench)
CUPTI_FOUND=false
for CUPTI_PATH in /usr/local/cuda/lib64/libcupti.so /usr/local/cuda-${CUDA_MAJOR_VERSION}/lib64/libcupti.so /usr/local/cuda-${CUDA_VERSION}/lib64/libcupti.so; do
    if [[ -f "$CUPTI_PATH" ]]; then
        echo "✓ Found CUPTI library: $CUPTI_PATH"
        CUPTI_FOUND=true
        break
    fi
done

if [[ "$CUPTI_FOUND" == false ]]; then
    echo "Warning: CUPTI library not found in standard locations"
    echo "nvbench requires libcupti.so from CUDA Toolkit"
    echo "Please ensure CUDA Toolkit is fully installed"
    echo ""
fi

# Check LD_LIBRARY_PATH
if [[ ":$LD_LIBRARY_PATH:" == *":/usr/local/cuda/lib64:"* ]] || [[ ":$LD_LIBRARY_PATH:" == *":/usr/local/cuda-${CUDA_MAJOR_VERSION}/lib64:"* ]]; then
    echo "✓ LD_LIBRARY_PATH includes CUDA libraries"
else
    echo "⚠ Warning: LD_LIBRARY_PATH may not include CUDA libraries"
    echo "  You may need to set:"
    echo "    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "  or add it to your ~/.bashrc"
    echo ""
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install python3-pip." >&2
    exit 1
fi

# Check if we're in a virtual environment (recommended for CI)
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Note: Not running in a virtual environment."
    echo "For isolated installations, consider using:"
    echo "  python3 -m venv venv_bench"
    echo "  source venv_bench/bin/activate"
    echo ""
fi

# Upgrade pip (only if in virtual environment to avoid PEP 668 issues on Python 3.12+)
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Upgrading pip in virtual environment..."
    python3 -m pip install --upgrade pip
else
    echo "Skipping pip upgrade (not in virtual environment)"
fi

echo ""
echo "Installing Python dependencies for CUDA $CUDA_MAJOR_VERSION..."
echo ""

REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.bench.cu${CUDA_MAJOR_VERSION}.txt"

# Check that required requirements files have been generated.
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "Error: the following requirements file is missing:" >&2
    echo "  $REQUIREMENTS_FILE" >&2
    echo "Generate it first by running from the repository root:" >&2
    echo "  bash generate_requirements.sh" >&2
    exit 1
fi

echo "Installing from: $REQUIREMENTS_FILE"
python3 -m pip install -r "$REQUIREMENTS_FILE"

echo ""
echo "Checking cvcuda..."

if ! python3 -m pip show cvcuda-cu${CUDA_MAJOR_VERSION} >/dev/null 2>&1; then
    echo "Installing cvcuda for CUDA $CUDA_MAJOR_VERSION..."
    python3 -m pip install cvcuda-cu${CUDA_MAJOR_VERSION}
else
    echo "✓ cvcuda-cu${CUDA_MAJOR_VERSION} already installed, skipping."
fi

echo ""
echo "$SEPARATOR_LINE"
echo "✓ Dependency installation complete!"
echo "$SEPARATOR_LINE"
echo ""
echo "You can now run benchmarks:"
echo "  cd $(dirname "$SCRIPT_DIR")/bin"
echo ""
echo "  # Run all benchmarks (C++ and Python)"
echo "  python3 run_bench.py ."
echo ""
echo "  # Run only C++ or Python"
echo "  python3 run_bench.py . --lang cpp"
echo "  python3 run_bench.py . --lang python"
echo ""
echo "Or run individual benchmarks:"
echo ""
echo "  Python benchmarks:"
echo "    python3 bench_resize.py --axis shape=1x1080x1920 --axis dtype=uint8"
echo "    python3 bench_gaussian.py --axis shape=1x1080x1920"
echo ""
echo "  C++ benchmarks:"
echo "    ./bench_resize --axis shape=1x1080x1920 --csv out.csv"
echo "    ./bench_gaussian --axis shape=1x1080x1920 --csv out.csv"
echo ""
