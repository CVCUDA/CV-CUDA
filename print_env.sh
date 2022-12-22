#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Usage:
# "./print_env.sh" - prints to stdout
# "./print_env.sh > env.txt" - prints to file "env.txt"

print_env() {
echo "**git***"
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
git log --decorate -n 1
echo "**git submodules***"
git submodule status --recursive
else
echo "Not inside a git repository"
fi
echo

echo "***OS Information***"
cat /etc/*-release
uname -a
echo

echo "***GPU Information***"
nvidia-smi
echo

echo "***CPU***"
lscpu
echo

echo "***CMake***"
which cmake && cmake --version
echo

echo "***g++***"
which g++ && g++ --version
echo

echo "***nvcc***"
which nvcc && nvcc --version
echo

echo "***Python***"
which python && python -c "import sys; print('Python {0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))"
echo

echo "***Environment Variables***"

printf '%-32s: %s\n' PATH $PATH

printf '%-32s: %s\n' LD_LIBRARY_PATH $LD_LIBRARY_PATH

printf '%-32s: %s\n' NUMBAPRO_NVVM $NUMBAPRO_NVVM

printf '%-32s: %s\n' NUMBAPRO_LIBDEVICE $NUMBAPRO_LIBDEVICE

printf '%-32s: %s\n' CONDA_PREFIX $CONDA_PREFIX

printf '%-32s: %s\n' PYTHON_PATH $PYTHON_PATH

echo


# Print conda packages if conda exists
if type "conda" &> /dev/null; then
echo '***conda packages***'
which conda && conda list
echo
# Print pip packages if pip exists
elif type "pip" &> /dev/null; then
echo "conda not found"
echo "***pip packages***"
which pip && pip list
echo
else
echo "conda not found"
echo "pip not found"
fi
}

echo "<details><summary>Click here to see environment details</summary><pre>"
echo "     "
print_env | while read -r line; do
    echo "     $line"
done
echo "</pre></details>"
