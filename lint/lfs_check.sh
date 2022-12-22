#!/bin/bash -e

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

# Check if files that should be handled by LFS are being committed as
# LFS objects

lfs_files=$(echo "$@" | xargs git check-attr filter | grep 'filter: lfs$' | sed -e 's@: filter: lfs@@')

binary_files=''

for file in $lfs_files; do
    soft_sha=$(git hash-object -w $file)
    raw_sha=$(git hash-object -w --no-filters $file)

    if [ $soft_sha == $raw_sha ]; then
        binary_files="* $file\n$binary_files"
    fi
done

if [[ "$binary_files" ]]; then
    echo "The following files tracked by git-lfs are being committed as standard git objects:"
    echo -e "$binary_files"
    echo "Revert your changes and commit those with git-lfs installed."
    echo "In repo's root directory, run: sudo apt-get git-lfs && git lfs install"
    exit 1
fi
