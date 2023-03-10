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

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

distro_name=$(lsb_release -is || true)
distro_ver=$(lsb_release -rs || true)

function version_le()
{
    [[ $(echo -e "$1\n$2" | sort -V | head -n1) = "$1" ]] && echo true
}

skip_precommit=0

if ! which pre-commit || ! which shellcheck ; then
    echo 'pre-commit must be fully configured.'
    if [[ "$distro_name" = "Ubuntu" ]]; then
        if [[ $(version_le "$distro_ver" "18.04") ]]; then
            echo "Ubuntu v$distro_ver is too old, you need at least Ubuntu 20.04."
            skip_precommit=1
        elif [[ $(version_le "$distro_ver" "21.10") ]]; then
            echo "Try 'sudo apt-get install -y pip shellcheck && sudo pip install pre-commit'."
            exit 1
        else
            echo "Try 'sudo apt-get install -y pre-commit shellcheck'."
            exit 1
        fi
    else
        echo "For linting support, install pre-commit and shellcheck packaged from your distro"
        skip_precommit=1
    fi
fi

if ! which git-lfs ; then
    echo "git-lfs must be fully configured. Try 'apt-get install git-lfs'."
    exit 1
fi

cd "$SDIR"

# We use LFS
git lfs install

# Fetch all lfs object
git lfs fetch && git lfs checkout

# We use submodules
git submodule update --init

if [ "$skip_precommit" -ne 1 ]; then
    # allow-missing-config is useful when checking out an old commit or a branch that don't have pre-config configuration.
    pre-commit install \
        --allow-missing-config \
        --install-hooks \
        -t pre-commit \
        -t pre-merge-commit \
        -t commit-msg
fi
