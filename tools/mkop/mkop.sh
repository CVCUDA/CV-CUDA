#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

modify_and_update_template() {
    local file="$1"
    local name="$2"
    local destination="$3"
    local year
    local capName
    local lowName

    year=$(date +%Y)
    capName=$(echo "$name" | sed 's/\([A-Z]\)/_\L\1/g' | tr '[:lower:]' '[:upper:]')
    lowName=$(echo "$name" | tr '[:upper:]' '[:lower:]')
    spacedName=$(echo $name | sed 's/\([a-z]\)\([A-Z]\)/\1 \2/g')
    # Replace all occurrences of the "TAG" string with the provided name
    sed "s/__OPNAME__/$name/g" "$file" > "$destination"
    sed -i 's/\(.*(\s*c\s*)\).*\(NVIDIA.*\)/\1'" $year "'\2/' "$destination"
    sed -i "s/__OPNAMECAP__/$capName/g" "$destination"
    sed -i "s/__OPNAMELOW__/$lowName/g" "$destination"
    sed -i "s/__OPNAMESPACE__/$spacedName/g" "$destination"
}

add_to_cmake() {
    local file="$1"
    local name="$2"
    local line_number

    if ! grep -q "$name" "$file"; then
        line_number=$(grep -n "add_library(" "$file" | head -n 1 | cut -d: -f1)
        line_number=$((line_number+1))
        sed -i "$line_number i\    $name" "$file"
    fi
}

add_to_test_cmake() {
    local file="$1"
    local name="$2"
    local line_number

    if ! grep -q "$name" "$file"; then
        line_number=$(grep -n "add_executable(" "$file" | head -n 1 | cut -d: -f1)
        line_number=$((line_number+1))
        sed -i "$line_number i\    $name" "$file"
    fi
}

add_to_cmake_python() {
    local file="$1"
    local name="$2"
    local line_number

    if ! grep -q "$name" "$file"; then
        line_number=$(grep -n "SOURCES" "$file" | head -n 1 | cut -d: -f1)
        line_number=$((line_number+2))
        sed -i "$line_number i\        $name" "$file"
    fi
}

add_to_python_main() {

    local file="$1"
    local name="$2"

    if ! grep -q "$name" "$file"; then
         sed "s/CV-CUDA Operators/&\n    ExportOp$name(m);/g" -i "$file"
    fi
}

add_to_python_operators() {

    local file="$1"
    local name="$2"
    local line_number

    if ! grep -q "$name" "$file"; then
        line_number=$(grep -n 'void ExportOp' "$file" | tail -1 | cut -d ':' -f 1)
        line_number=$((line_number+1))
        sed -i "$line_number i\void ExportOp$name(py::module &m);" "$file"

    fi
}

# Check if the correct number of arguments have been provided
if [[ $# != 1 && $# != 2 ]]; then
    echo "Create a stub (noop) operator and tests for the operator"
    echo "Usage: $0 <OperatorName> [CVCUDA root]"
    exit 1;
fi

# if not provided assume script is in /cvcuda/tools/mkop
root="../.."

if [ $# -eq 2 ]; then
    root="$2"
fi

# Store the name and destination arguments (first letter Cap)
name=$(echo "$1" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')
namelower=$(echo "$name" | tr '[:upper:]' '[:lower:]')

#public API's for the operator
modify_and_update_template "Public.h" "$name" "$root/src/cvcuda/include/cvcuda/Op$name.h"
modify_and_update_template "Public.hpp" "$name" "$root/src/cvcuda/include/cvcuda/Op$name.hpp"
modify_and_update_template "CImpl.cpp" "$name" "$root/src/cvcuda/Op$name.cpp"

#internal implementation
modify_and_update_template "PrivateImpl.cpp" "$name" "$root/src/cvcuda/priv/Op$name.cpp"
modify_and_update_template "PrivateImpl.hpp" "$name" "$root/src/cvcuda/priv/Op$name.hpp"

#C++ system tests
modify_and_update_template "CppTest.cpp" "$name" "$root/tests/cvcuda/system/TestOp$name.cpp"

#add to makefiles
add_to_cmake "$root/src/cvcuda/priv/CMakeLists.txt"  "Op$name.cpp"
add_to_cmake "$root/src/cvcuda/CMakeLists.txt"  "Op$name.cpp"
add_to_test_cmake "$root/tests/cvcuda/system/CMakeLists.txt"  "TestOp$name.cpp"

#add python stub
modify_and_update_template "PythonWrap.cpp" "$name" "$root/python/mod_cvcuda/Op$name.cpp"
add_to_python_main "$root/python/mod_cvcuda/Main.cpp"  "$name"
add_to_python_operators "$root/python/mod_cvcuda/Operators.hpp" "$name"

#add python makefile
add_to_cmake_python "$root/python/mod_cvcuda/CMakeLists.txt"  "Op$name.cpp"

#add python test
modify_and_update_template "PythonTest.py" "$name" "$root/tests/cvcuda/python/test_op$namelower.py"
