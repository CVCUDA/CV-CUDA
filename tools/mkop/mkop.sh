#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# POSIX character-class names for `tr` case conversion (shared to avoid
# duplicating the literals; SonarQube shelldre:S1192).
readonly TR_LOWER='[:lower:]'
readonly TR_UPPER='[:upper:]'

# Fail loudly + stop the scaffold when a required insertion anchor / version / file is missing,
# so mkop.sh never exits green with incomplete wiring (the helpers are called bare, without
# `set -e`, so a bare `return 1` would be swallowed — `exit` is what actually stops the run).
die() { # NOSONAR shelldre:S7682 — terminal helper: deliberately `exit`s, never returns
    echo "mkop: $*" >&2
    exit 1
}

modify_and_update_template() {
    local file="$1"
    local name="$2"
    local destination="$3"
    local year
    local cap_name
    local low_name
    local spaced_name
    local upper_name

    year=$(date +%Y)
    cap_name=$(echo "$name" | sed 's/\([A-Z]\)/_\L\1/g' | tr "$TR_LOWER" "$TR_UPPER")
    low_name=$(echo "$name" | tr "$TR_UPPER" "$TR_LOWER")
    spaced_name=$(echo "$name" | sed 's/\([a-z]\)\([A-Z]\)/\1 \2/g')
    # Uppercased lowercase name (no separators), matching the bench codegen's
    # BENCH_<TOUPPER(opkey)>_* macro names (see bench/cpp/GenerateBenchConfig.cmake).
    upper_name=$(echo "$low_name" | tr "$TR_LOWER" "$TR_UPPER")
    # Replace all occurrences of the "TAG" string with the provided name
    sed "s/__OPNAME__/$name/g" "$file" > "$destination"
    sed -i 's/\(.*(\s*c\s*)\).*\(NVIDIA.*\)/\1'" $year "'\2/' "$destination"
    sed -i "s/__OPNAMECAP__/$cap_name/g" "$destination"
    sed -i "s/__OPNAMELOW__/$low_name/g" "$destination"
    sed -i "s/__OPNAMESPACE__/$spaced_name/g" "$destination"
    sed -i "s/__OPNAMEUPPER__/$upper_name/g" "$destination"
    return $?
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
    return $?
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
    return $?
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
    return $?
}

add_to_python_main() {

    local file="$1"
    local name="$2"
    local line_number

    # "CV-CUDA Operators" appears twice — once in an include-section comment and once as the
    # indented marker above the ExportOp* registration block. Anchor on the indented marker so
    # the call lands among the other ExportOp* calls (not spliced into the include comment).
    if ! grep -q "ExportOp$name(m);" "$file"; then
        line_number=$(grep -nE '^[[:space:]]+// CV-CUDA Operators' "$file" | head -1 | cut -d: -f1)
        [[ -z "$line_number" ]] && die "'// CV-CUDA Operators' anchor not found in $file"
        line_number=$((line_number+1))
        sed -i "${line_number}i\\        ExportOp$name(m);" "$file"
    fi
    return $?
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
    return $?
}

# Insert "$entry" on the line after the first line matching "$anchor", indented by "$indent".
# Used for the bench `set(bench_sources ...)` and `set(python_bench_scripts ...)` lists.
add_after_set_anchor() {
    local file="$1"
    local anchor="$2"
    local indent="$3"
    local entry="$4"
    local line_number

    if ! grep -qF "$entry" "$file"; then
        line_number=$(grep -n "$anchor" "$file" | head -n 1 | cut -d: -f1)
        [[ -z "$line_number" ]] && die "anchor '$anchor' not found in $file"
        line_number=$((line_number+1))
        sed -i "${line_number}i\\${indent}${entry}" "$file"
    fi
    return $?
}

# Insert the operator's manifest entry into bench/config/bench_params.json (alphabetically),
# preserving the file's 4-space indentation. Idempotent.
add_to_bench_params() {
    local file="$1"
    local op="$2"

    python3 - "$file" "$op" <<'PYEOF'
import json
import sys

path, op = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
ops = data.setdefault("operators", {})
if op not in ops:
    ops[op] = {
        "config": "operators/%s.json" % op,
        "cpp": "bench_%s" % op,
        "python": "bench_%s.py" % op,
    }
    data["operators"] = {k: ops[k] for k in sorted(ops)}
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
PYEOF
    return $?
}

# Insert the operator into bench/config/operator_categories.json (the RGB-benchmark-guidelines
# manifest that bench/tests/test_bench_rgb_guidelines.py::test_r1_every_operator_classified
# requires every on-disk operator config to appear in). Defaults to category "A" (general
# per-pixel/per-channel image op — the common case); the agent must verify and switch to B
# (inherently single-channel) or C (intrinsic channel semantics) if that fits better. Idempotent.
add_to_operator_categories() {
    local file="$1"
    local op="$2"

    [[ -f "$file" ]] || die "operator category manifest not found: $file"
    python3 - "$file" "$op" <<'PYEOF'
import json
import sys

path, op = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
ops = data.setdefault("operators", {})
if op not in ops:
    ops[op] = {"category": "A"}
    data["operators"] = {k: ops[k] for k in sorted(ops)}
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
PYEOF
    return $?
}

# Insert a two-line row into the docs/sphinx/operator_list.rst table, alphabetically by the
# displayed operator name. Idempotent (keyed on the py:func target).
add_to_oplist() {
    local file="$1"
    local disp="$2"
    local pyfn="$3"
    local desc="$4"

    grep -q "cvcuda\.$pyfn\`" "$file" && return 0
    # Only consider real operator rows (those with a :py:func: target); the list-table's
    # header row ("Pre/Post-Processing Operators" | "Definition") has none and must be skipped.
    awk -v disp="$disp" -v pyfn="$pyfn" -v desc="$desc" '
    BEGIN { ins = 0; key = tolower(disp) }
    /^   \* - .*:py:func:/ {
        d = $0; sub(/^   \* - /, "", d); sub(/ \(:py:func.*/, "", d)
        if (ins == 0 && tolower(d) > key) {
            printf "   * - %s (:py:func:`cvcuda.%s`)\n", disp, pyfn
            printf "     - %s\n", desc
            ins = 1
        }
    }
    { print }
    END {
        if (ins == 0) {
            printf "   * - %s (:py:func:`cvcuda.%s`)\n", disp, pyfn
            printf "     - %s\n", desc
        }
    }
    ' "$file" > "$file.mkop.tmp" && mv "$file.mkop.tmp" "$file"
    return $?
}

# Insert the two cvcuda-autofunction directives (fn + fn_into) into
# docs/sphinx/modules/python/operators.rst, alphabetically. Idempotent.
add_to_autofunction() {
    local file="$1"
    local pyfn="$2"

    grep -qE "cvcuda-autofunction:: cvcuda\.$pyfn\$" "$file" && return 0
    awk -v n="$pyfn" '
    BEGIN { ins = 0 }
    /^\.\. cvcuda-autofunction:: cvcuda\./ {
        fn = $0; sub(/^.*cvcuda\./, "", fn); sub(/_into$/, "", fn)
        if (ins == 0 && fn > n) {
            printf ".. cvcuda-autofunction:: cvcuda.%s\n\n", n
            printf ".. cvcuda-autofunction:: cvcuda.%s_into\n\n", n
            ins = 1
        }
    }
    { print }
    END {
        if (ins == 0) {
            printf "\n.. cvcuda-autofunction:: cvcuda.%s\n\n", n
            printf ".. cvcuda-autofunction:: cvcuda.%s_into\n", n
        }
    }
    ' "$file" > "$file.mkop.tmp" && mv "$file.mkop.tmp" "$file"
    return $?
}

# Insert a "New Features" bullet for the operator into the latest release notes. "Latest" is the
# relnote whose version matches CMakeLists.txt's VERSION. Idempotent.
add_to_relnote() {
    local root="$1"
    local name="$2"
    local ver
    local relfile
    local line_number

    # The project VERSION is the indented `VERSION X.Y.Z` line inside project(); anchor on
    # leading whitespace so we don't match `cmake_minimum_required(VERSION 3.20.1)`.
    ver=$(grep -oE '^[[:space:]]+VERSION[[:space:]]+[0-9]+\.[0-9]+\.[0-9]+' "$root/CMakeLists.txt" | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    [[ -z "$ver" ]] && die "project VERSION not found in $root/CMakeLists.txt"
    relfile=$(ls "$root"/docs/sphinx/relnotes/v"$ver"-*.rst 2>/dev/null | head -1)
    [[ -z "$relfile" ]] && relfile=$(ls "$root"/docs/sphinx/relnotes/v"$ver".rst 2>/dev/null | head -1)
    [[ -z "$relfile" ]] && die "no relnote file for version $ver under $root/docs/sphinx/relnotes/"
    local first_bullet
    grep -q "\`\`$name\`\` operator" "$relfile" && return 0
    line_number=$(grep -n "New Features and Enhancements" "$relfile" | head -1 | cut -d: -f1)
    [[ -z "$line_number" ]] && die "'New Features and Enhancements' anchor not found in $relfile"
    # Insert before the first existing nested bullet so the blank line after the section
    # header (required for RST nested-list rendering) is preserved; append a trailing blank.
    first_bullet=$(awk -v h="$line_number" 'NR>h && /^[[:space:]]+\* / {print NR; exit}' "$relfile")
    if [[ -n "$first_bullet" ]]; then
        sed -i "${first_bullet}i\\  * Added the \`\`$name\`\` operator\n" "$relfile"
    else
        sed -i "$((line_number+1))i\\  * Added the \`\`$name\`\` operator" "$relfile"
    fi
    return $?
}

# Check if the correct number of arguments have been provided
if [[ $# != 1 && $# != 2 ]]; then
    echo "Create a stub (noop) operator and tests for the operator"
    echo "Usage: $0 <OperatorName> [CVCUDA root]"
    exit 1;
fi

# if not provided assume script is in /cvcuda/tools/mkop
root="../.."

if [[ $# -eq 2 ]]; then
    root="$2"
fi

# Store the name and destination arguments (first letter Cap)
name=$(echo "$1" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')
namelower=$(echo "$name" | tr "$TR_UPPER" "$TR_LOWER")

#public API's for the operator
modify_and_update_template "Public.h" "$name" "$root/src/cvcuda/include/cvcuda/Op$name.h"
modify_and_update_template "Public.hpp" "$name" "$root/src/cvcuda/include/cvcuda/Op$name.hpp"
modify_and_update_template "CImpl.cpp" "$name" "$root/src/cvcuda/Op$name.cpp"

#internal implementation
modify_and_update_template "PrivateImpl.cpp" "$name" "$root/src/cvcuda/priv/Op$name.cpp"
modify_and_update_template "PrivateImpl.hpp" "$name" "$root/src/cvcuda/priv/Op$name.hpp"

#C++ system tests
modify_and_update_template "CppTest.cpp" "$name" "$root/tests/cvcuda/system/TestOp$name.cpp"

#add to makefiles (insert into the source-list set(...) blocks, not add_library/add_executable)
add_after_set_anchor "$root/src/cvcuda/priv/CMakeLists.txt"      "set(CV_CUDA_PRIV_OP_FILES"  "    " "Op$name.cpp"
add_after_set_anchor "$root/src/cvcuda/CMakeLists.txt"           "set(CV_CUDA_OP_FILES"       "    " "Op$name.cpp"
add_after_set_anchor "$root/tests/cvcuda/system/CMakeLists.txt"  "set(CVCUDA_TEST_SOURCES"    "    " "TestOp$name.cpp"

#add python stub (bindings live under operators/)
modify_and_update_template "PythonWrap.cpp" "$name" "$root/python/mod_cvcuda/operators/Op$name.cpp"
add_to_python_main "$root/python/mod_cvcuda/Main.cpp"  "$name"
add_to_python_operators "$root/python/mod_cvcuda/operators/Operators.hpp" "$name"

#add python makefile
add_to_cmake_python "$root/python/mod_cvcuda/CMakeLists.txt"  "operators/Op$name.cpp"

#add python test
modify_and_update_template "PythonTest.py" "$name" "$root/tests/cvcuda/python/test_op$namelower.py"

#add benchmarks (C++ driver + Python driver + shared config + manifest entry)
modify_and_update_template "Bench.cpp" "$name" "$root/bench/cpp/ops/Bench$name.cpp"
modify_and_update_template "BenchPy.py" "$name" "$root/bench/python/ops/bench_$namelower.py"
modify_and_update_template "BenchConfig.json" "$name" "$root/bench/config/operators/$namelower.json"
add_after_set_anchor "$root/bench/cpp/CMakeLists.txt"     "set(bench_sources"         "    " "ops/Bench$name.cpp"
add_after_set_anchor "$root/bench/python/CMakeLists.txt"  "set(python_bench_scripts"  "    " "ops/bench_$namelower.py"
add_to_bench_params  "$root/bench/config/bench_params.json"  "$namelower"
add_to_operator_categories "$root/bench/config/operator_categories.json" "$namelower"

#add documentation (operator list + python autofunction directives + latest release notes)
add_to_oplist        "$root/docs/sphinx/operator_list.rst"  "$name"  "$namelower"  "TODO(make-op): one-line description of the $name operator."
add_to_autofunction  "$root/docs/sphinx/modules/python/operators.rst"  "$namelower"
add_to_relnote       "$root"  "$name"
