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

if [ $# = 0 ]; then
    # No arguments? Lint all code.
    echo "Linting all code in the repository =========================="
    pre-commit run -a
else
    from=$1
    if [ $# = 1 ]; then
        to=HEAD
    elif [ $# = 2 ]; then
        to=$2
    else
        echo "Invalid arguments"
        echo "Usage: $(basename "$0") [ref_from [ref_to]]"
        exit 1
    fi

    echo "Linting files touched from commit $from to $to =============="
    echo "Files to be linted:"
    git diff --stat $from..$to
    if ! pre-commit run --from-ref $from --to-ref $to ; then
        echo "Formatting errors:"
        git diff
        false
    fi
fi
