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

function cleanup()
{
    rm -f stub_symbols orig_symbols libtest.so libtest_stub.so
}

trap 'cleanup' EXIT

function run_test()
{
    local name=$1
    local src=$2
    local ver=$3

    echo "test $name"

    local args=''
    if [ -n "$ver" ]; then
        args="-Wl,--version-script=$ver"
    fi

    gcc -o libtest.so -shared -fPIC $args -Wl,-soname=libtest.so -xc - <<<$src
    rm -f libtest_stub.so

    ./mkstub.sh libtest.so
    function list_contents()
    {
        readelf --dyn-syms "$1" | awk '$7 !~ /(UND|^$|Ndx)/ { print $4,$5,$6,$8 }' | sort
    }

    list_contents libtest.so > orig_symbols
    list_contents libtest_stub.so > stub_symbols

    diff orig_symbols stub_symbols
}

run_test versioned "$(cat test_versioned.c)" test_versioned.v
run_test mixed "$(cat test_versioned.c test_noversion.c)" test_versioned.v
run_test "not versioned" "$(cat test_noversion.c)"
