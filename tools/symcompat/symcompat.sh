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

# Given an executable (or dso) A and a list of DSOs B, find the symbols
# that missing in A and try to find equivalents (maybe from older version) in B.

# It outputs a list of missing symbol in A that could and could not be found in B.

if [ $# -lt 1 ]; then
    echo "Invalid args. Usage: $(basename $0) <target exec> [old lib,...]"
    exit 1
fi

target=$1
shift

oldlibs="$*"

missingsyms="$(ldd -r $target 2>&1 | awk '/^symbol.*/ { print $2 }' | sed s/,//g | sort | uniq)"

misspattern="^("
for s in $missingsyms; do
    misspattern+="|$s"
done
misspattern+=")@@"

total_found="" # symbol name + version tag

for lib in $oldlibs; do
    libsyms="$(readelf -sW $lib | awk '$4 ~ /FUNC/ && $5 ~ /(GLOBAL|WEAK)/ && $6 ~ /DEFAULT/ && $7 !~ /UND/ && $8 ~ /.*@@/ { print $8 }' | sort | uniq)"
    found="$(echo "$libsyms" | egrep "$misspattern" | sed 's/@@/@/g' || true)"
    echo "------ $lib" 1>&2
    if [ "$found" ]; then
        #if [ "$total_found" ]; then
        #    total_found+="\n"
        #fi
        total_found+="\n$found"
    fi
done

total_found="$(echo -e "$total_found" | egrep -v "^[[:space:]]*$" | sort | uniq)"

notfound="$(diff <( echo "$missingsyms" ) <( echo "$total_found" | awk -F '@' '{ print $1 }') || true)"

if [ "$notfound" ]; then
    echo -e "\nSymbols not found:"
    echo "$notfound" | awk '/^</ { print $2 }'
fi

if [ "$total_found" ]; then
    echo -e "\nSymbols to be added:"
    echo -e "$total_found"
fi
