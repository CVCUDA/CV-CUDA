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

# This script installs all the dependencies required to run CV-CUDA samples.
# It detects the CUDA version and installs appropriate packages.

# Check CUDA version. Begin by checking if nvcc command exists.
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d. -f1)
    if [[ "$CUDA_MAJOR_VERSION" -eq 12 ]] || [[ "$CUDA_MAJOR_VERSION" -eq 13 ]]; then
        echo "CUDA $CUDA_MAJOR_VERSION is installed."
    else
        echo "Unknown/Unsupported CUDA version."
        exit 1
    fi
else
    echo "CUDA is not installed."
    exit 1
fi

echo "Installing Python dependencies for CV-CUDA samples..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create virtual environment if it doesn't exist
if [[ ! -d "$SCRIPT_DIR/venv_samples" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv --system-site-packages "$SCRIPT_DIR/venv_samples"
fi

source "$SCRIPT_DIR/venv_samples/bin/activate"
python3 -m pip install --upgrade pip

# Check that the requirements file has been generated.
REQ_FILE="$SCRIPT_DIR/requirements.samples.cu${CUDA_MAJOR_VERSION}.txt"
if [[ ! -f "$REQ_FILE" ]]; then
    echo "Error: $REQ_FILE not found." >&2
    echo "Generate it first by running from the repository root:" >&2
    echo "  bash generate_requirements.sh" >&2
    exit 1
fi

# Install sample dependencies for the detected CUDA version.
# requirements.samples.cu{12,13}.txt includes torch, torchvision, and all other deps.
# If CV-CUDA is already installed (e.g. from a wheel), filter it out to avoid reinstalling.
FILTERED_REQ=$(mktemp "$SCRIPT_DIR/requirements_filtered.XXXXXX.txt")
trap 'rm -f "$FILTERED_REQ"' EXIT
cp "$REQ_FILE" "$FILTERED_REQ"

if python3 -m pip list 2>/dev/null | grep -q "cvcuda-cu${CUDA_MAJOR_VERSION}"; then
    echo "CV-CUDA is already installed, filtering it from requirements..."
    grep -v "^cvcuda-cu" "$FILTERED_REQ" > "$FILTERED_REQ.tmp"
    mv "$FILTERED_REQ.tmp" "$FILTERED_REQ"
fi

python3 -m pip install -r "$FILTERED_REQ"
rm "$FILTERED_REQ"
trap - EXIT

echo ""
echo "Python dependencies installation complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $SCRIPT_DIR/venv_samples/bin/activate"
echo ""
echo "Then you can run samples from the samples directory:"
echo "  python3 operators/label.py"
echo "  python3 applications/classification.py"
echo "  python3 interoperability/pytorch_interop.py"
echo ""
