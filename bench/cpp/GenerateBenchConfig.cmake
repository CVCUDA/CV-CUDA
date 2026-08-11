# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to generate C++ benchmark configuration headers from JSON
#
# This script generates Bench<OperatorName>Config.hpp headers containing
# macro definitions for benchmark axis configurations.
#
# Usage:
#   cmake -DJSON_FILE=<path> -DOPERATOR=<name> -DOPERATOR_CAMEL=<CamelName> -DOUTPUT_FILE=<path> -P GenerateBenchConfig.cmake
#
# Parameters:
#   JSON_FILE      - Path to an operator JSON file
#   OPERATOR       - Lowercase operator name (e.g., "resize", "adaptivethreshold")
#   OPERATOR_CAMEL - CamelCase operator name (e.g., "Resize", "AdaptiveThreshold")
#   OUTPUT_FILE    - Output path for generated header file
#
# Output:
#   Generates a header file with format Bench<OperatorName>Config.hpp containing
#   a BENCH_<OPERATOR>_AXES macro that can be used in NVBENCH_BENCH_TYPES declarations.
#
# Aggregation semantics:
#   Each operator JSON holds multiple config entries for the operator (e.g. one
#   per tier × variant). This generator unions per-operator metadata across ALL
#   entries with `benchmark` == OPERATOR so the compiled binary's axis
#   registry is a superset of every value any tier asks for at runtime:
#     - dtypes:       union of dtype lists → nvbench::type_list<...>
#     - string_axes:  per-axis-name union of string values
#     - int64_axes:   per-axis-name union of int64 values
#     - float64_axes: per-axis-name union of float64 values
#     - warmup:       first-match wins (single scalar; no obvious merge semantic)
#
#   Without union behavior, an axis or dtype that lives only in one tier's
#   entry would silently desync from the binary's compile-time registry —
#   nvbench would error at runtime with "Unknown axis" when run_bench.py
#   passes the override CLI flag.

file(READ "${JSON_FILE}" JSON_CONTENT)

set(CONFIG_ROOT "${JSON_CONTENT}")
set(TOP_LEVEL_BENCHMARK "")
string(JSON _CONFIGS_OBJ ERROR_VARIABLE CONFIGS_ERR GET "${JSON_CONTENT}" "configs")
string(JSON _TOP_BENCHMARK ERROR_VARIABLE TOP_BENCHMARK_ERR GET "${JSON_CONTENT}" "benchmark")
if(NOT CONFIGS_ERR AND NOT TOP_BENCHMARK_ERR)
    set(CONFIG_ROOT "${_CONFIGS_OBJ}")
    set(TOP_LEVEL_BENCHMARK "${_TOP_BENCHMARK}")
endif()

if(NOT DEFINED AXIS_ORDER_HELPER)
    set(AXIS_ORDER_HELPER "${CMAKE_CURRENT_LIST_DIR}/../config/axis_order.py")
endif()
find_program(AXIS_ORDER_PYTHON_EXECUTABLE NAMES python3 REQUIRED)
execute_process(
    COMMAND "${AXIS_ORDER_PYTHON_EXECUTABLE}" "${AXIS_ORDER_HELPER}"
        --config "${JSON_FILE}" --operator "${OPERATOR}"
    RESULT_VARIABLE AXIS_ORDER_RESULT
    OUTPUT_VARIABLE AXIS_ORDER_OUTPUT
    ERROR_VARIABLE AXIS_ORDER_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT AXIS_ORDER_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to resolve axis order: ${AXIS_ORDER_ERROR}")
endif()
string(REPLACE "\n" ";" AXIS_ORDER "${AXIS_ORDER_OUTPUT}")

# ---------------------------------------------------------------------------
# Pass 1: scan ALL top-level entries; for each entry whose "benchmark" field
# matches OPERATOR, accumulate dtypes and per-axis values.
#
# CMake has no dict type — we fake one by maintaining ordered "name" lists
# per axis category, plus a per-(name) values list whose variable name is
# constructed from the axis name. This preserves first-seen ordering so the
# generated header is deterministic across runs.
# ---------------------------------------------------------------------------

set(ALL_DTYPES "")
set(STRING_AXIS_NAMES "")
set(INT64_AXIS_NAMES "")
set(FLOAT64_AXIS_NAMES "")
set(WARMUP_ITERATIONS "")
set(FOUND_CONFIG FALSE)

string(JSON NUM_ENTRIES LENGTH "${CONFIG_ROOT}")
if(NUM_ENTRIES GREATER 0)
    math(EXPR NUM_ENTRIES_MINUS_1 "${NUM_ENTRIES} - 1")
    foreach(IDX RANGE 0 ${NUM_ENTRIES_MINUS_1})
        string(JSON ENTRY_KEY MEMBER "${CONFIG_ROOT}" ${IDX})
        string(JSON ENTRY_CONFIG GET "${CONFIG_ROOT}" "${ENTRY_KEY}")
        if(TOP_LEVEL_BENCHMARK STREQUAL "")
            string(JSON BENCHMARK_FIELD ERROR_VARIABLE BM_ERR GET "${ENTRY_CONFIG}" "benchmark")
        else()
            set(BENCHMARK_FIELD "${TOP_LEVEL_BENCHMARK}")
            set(BM_ERR "")
        endif()

        # Skip entries that don't target this operator. Entries with no
        # "benchmark" field in old-shape files are invalid but not fatal here — they're caught
        # at run time by load_config.py's validator.
        if(BM_ERR OR NOT BENCHMARK_FIELD STREQUAL "${OPERATOR}")
            continue()
        endif()

        set(FOUND_CONFIG TRUE)

        # warmup_iterations: first-match wins.
        if(WARMUP_ITERATIONS STREQUAL "")
            string(JSON WARMUP_VAL ERROR_VARIABLE WARMUP_ERR GET "${ENTRY_CONFIG}" "warmup_iterations")
            if(NOT WARMUP_ERR)
                set(WARMUP_ITERATIONS "${WARMUP_VAL}")
            endif()
        endif()

        # dtypes: union.
        string(JSON ENTRY_DTYPES ERROR_VARIABLE DTYPE_ERR GET "${ENTRY_CONFIG}" "dtypes")
        if(NOT DTYPE_ERR)
            string(JSON NUM_ENTRY_DTYPES LENGTH "${ENTRY_DTYPES}")
            if(NUM_ENTRY_DTYPES GREATER 0)
                math(EXPR NUM_ENTRY_DTYPES_MINUS_1 "${NUM_ENTRY_DTYPES} - 1")
                foreach(DIDX RANGE 0 ${NUM_ENTRY_DTYPES_MINUS_1})
                    string(JSON DTYPE GET "${ENTRY_DTYPES}" ${DIDX})
                    list(APPEND ALL_DTYPES "${DTYPE}")
                endforeach()
            endif()
        endif()

        # string_axes / int64_axes / float64_axes: per-axis-name union.
        # We expand the same loop body for each category since CMake lacks
        # closures.
        foreach(CATEGORY string_axes int64_axes float64_axes)
            string(JSON CAT_OBJ ERROR_VARIABLE CAT_ERR GET "${ENTRY_CONFIG}" "${CATEGORY}")
            if(CAT_ERR)
                continue()
            endif()
            string(JSON NUM_AXES LENGTH "${CAT_OBJ}")
            if(NUM_AXES LESS_EQUAL 0)
                continue()
            endif()

            if(CATEGORY STREQUAL "string_axes")
                set(NAMES_VAR STRING_AXIS_NAMES)
                set(PFX STRING_AXIS_VALUES_)
            elseif(CATEGORY STREQUAL "int64_axes")
                set(NAMES_VAR INT64_AXIS_NAMES)
                set(PFX INT64_AXIS_VALUES_)
            else()
                set(NAMES_VAR FLOAT64_AXIS_NAMES)
                set(PFX FLOAT64_AXIS_VALUES_)
            endif()

            math(EXPR NUM_AXES_MINUS_1 "${NUM_AXES} - 1")
            foreach(AIDX RANGE 0 ${NUM_AXES_MINUS_1})
                string(JSON AXIS_NAME MEMBER "${CAT_OBJ}" ${AIDX})
                string(JSON AXIS_VALUES GET "${CAT_OBJ}" "${AXIS_NAME}")

                # First-seen tracking: append to ordered names list iff new.
                set(NAMES_LIST "${${NAMES_VAR}}")
                list(FIND NAMES_LIST "${AXIS_NAME}" _SEEN_IDX)
                if(_SEEN_IDX EQUAL -1)
                    list(APPEND NAMES_LIST "${AXIS_NAME}")
                    set(${NAMES_VAR} "${NAMES_LIST}")
                endif()

                # Append this entry's values to the per-axis values list.
                # We dedupe the merged list at emit time, not here, so the
                # first-seen value ordering is preserved.
                string(JSON NUM_VALUES LENGTH "${AXIS_VALUES}")
                if(NUM_VALUES GREATER 0)
                    math(EXPR NUM_VALUES_MINUS_1 "${NUM_VALUES} - 1")
                    foreach(VIDX RANGE 0 ${NUM_VALUES_MINUS_1})
                        string(JSON VAL GET "${AXIS_VALUES}" ${VIDX})
                        list(APPEND ${PFX}${AXIS_NAME} "${VAL}")
                    endforeach()
                endif()
            endforeach()
        endforeach()
    endforeach()
endif()

if(NOT FOUND_CONFIG)
    message(FATAL_ERROR "Operator '${OPERATOR}' not found in ${JSON_FILE}")
endif()

# Dedupe dtypes (preserves first-seen order; CMake's REMOVE_DUPLICATES is
# stable from 3.0+).
list(REMOVE_DUPLICATES ALL_DTYPES)

# Map dtype strings to C++ types
set(TYPES_LIST "")
foreach(DTYPE IN LISTS ALL_DTYPES)
    if(DTYPE STREQUAL "uint8")
        list(APPEND TYPES_LIST "uint8_t")
    elseif(DTYPE STREQUAL "uint16")
        list(APPEND TYPES_LIST "uint16_t")
    elseif(DTYPE STREQUAL "uint32")
        list(APPEND TYPES_LIST "uint32_t")
    elseif(DTYPE STREQUAL "uint64")
        list(APPEND TYPES_LIST "uint64_t")
    elseif(DTYPE STREQUAL "int8")
        list(APPEND TYPES_LIST "int8_t")
    elseif(DTYPE STREQUAL "int16")
        list(APPEND TYPES_LIST "int16_t")
    elseif(DTYPE STREQUAL "int32")
        list(APPEND TYPES_LIST "int32_t")
    elseif(DTYPE STREQUAL "int64")
        list(APPEND TYPES_LIST "int64_t")
    elseif(DTYPE STREQUAL "float32")
        list(APPEND TYPES_LIST "float")
    elseif(DTYPE STREQUAL "float64")
        list(APPEND TYPES_LIST "double")
    elseif(DTYPE STREQUAL "bool")
        list(APPEND TYPES_LIST "bool")
    elseif(DTYPE STREQUAL "uchar3")
        list(APPEND TYPES_LIST "uchar3")
    elseif(DTYPE STREQUAL "uchar4")
        list(APPEND TYPES_LIST "uchar4")
    elseif(DTYPE STREQUAL "float3")
        list(APPEND TYPES_LIST "float3")
    elseif(DTYPE STREQUAL "float4")
        list(APPEND TYPES_LIST "float4")
    elseif(DTYPE STREQUAL "short2")
        list(APPEND TYPES_LIST "short2")
    elseif(DTYPE STREQUAL "ushort3")
        list(APPEND TYPES_LIST "ushort3")
    elseif(DTYPE STREQUAL "ushort4")
        list(APPEND TYPES_LIST "ushort4")
    elseif(DTYPE STREQUAL "short4")
        list(APPEND TYPES_LIST "short4")
    else()
        message(FATAL_ERROR "Unsupported dtype '${DTYPE}' for operator '${OPERATOR}'. "
            "Extend GenerateBenchConfig.cmake to handle this type.")
    endif()
endforeach()
string(REPLACE ";" ", " TYPES_STR "${TYPES_LIST}")

# ---------------------------------------------------------------------------
# Pass 2: emit per-axis macro lines for each category.
# ---------------------------------------------------------------------------

# Helper to emit "    .add_<kind>_axis(\"name\", {v1, v2, ...}) \\\n" for one axis.
# CMake macros don't return; we accumulate into a caller-named output var.
macro(emit_axis_line OUT_VAR ADDER NAME VALUES_LIST_VAR QUOTE_VALUES)
    set(_VALS "${${VALUES_LIST_VAR}}")
    list(REMOVE_DUPLICATES _VALS)
    set(_FORMATTED "")
    foreach(_V IN LISTS _VALS)
        # Macro args aren't bound as variables; STREQUAL on the substituted
        # text dodges CMake policy CMP0012 boolean-literal warnings.
        if("${QUOTE_VALUES}" STREQUAL "TRUE")
            list(APPEND _FORMATTED "\"${_V}\"")
        else()
            list(APPEND _FORMATTED "${_V}")
        endif()
    endforeach()
    string(REPLACE ";" ", " _STR "${_FORMATTED}")
    string(APPEND ${OUT_VAR} "    .${ADDER}(\"${NAME}\", {${_STR}}) \\\n")
endmacro()

set(ALL_AXES_CODE "")
foreach(NAME IN LISTS AXIS_ORDER)
    list(FIND STRING_AXIS_NAMES "${NAME}" STRING_IDX)
    list(FIND INT64_AXIS_NAMES "${NAME}" INT64_IDX)
    list(FIND FLOAT64_AXIS_NAMES "${NAME}" FLOAT64_IDX)
    if(NOT STRING_IDX EQUAL -1)
        emit_axis_line(ALL_AXES_CODE "add_string_axis" "${NAME}" STRING_AXIS_VALUES_${NAME} TRUE)
    elseif(NOT INT64_IDX EQUAL -1)
        emit_axis_line(ALL_AXES_CODE "add_int64_axis" "${NAME}" INT64_AXIS_VALUES_${NAME} FALSE)
    elseif(NOT FLOAT64_IDX EQUAL -1)
        emit_axis_line(ALL_AXES_CODE "add_float64_axis" "${NAME}" FLOAT64_AXIS_VALUES_${NAME} FALSE)
    else()
        message(FATAL_ERROR "axis_order references undefined axis '${NAME}'")
    endif()
endforeach()

set(DISCOVERED_AXIS_NAMES ${STRING_AXIS_NAMES} ${INT64_AXIS_NAMES} ${FLOAT64_AXIS_NAMES})
list(REMOVE_DUPLICATES DISCOVERED_AXIS_NAMES)
foreach(NAME IN LISTS DISCOVERED_AXIS_NAMES)
    list(FIND AXIS_ORDER "${NAME}" ORDERED_IDX)
    if(ORDERED_IDX EQUAL -1)
        message(FATAL_ERROR "Axis-order helper omitted discovered axis '${NAME}'")
    endif()
endforeach()

# warmup_iterations default = 0 if the operator never set one.
if(WARMUP_ITERATIONS STREQUAL "")
    set(WARMUP_ITERATIONS 0)
endif()

# Convert operator name to uppercase for header guard and macro
string(TOUPPER "${OPERATOR}" OPERATOR_UPPER)

# Remove trailing backslash and newline from the final combined axes code (last 3 chars: space, backslash, newline)
string(LENGTH "${ALL_AXES_CODE}" CODE_LENGTH)
if(CODE_LENGTH GREATER 3)
    math(EXPR TRIM_LENGTH "${CODE_LENGTH} - 3")
    string(SUBSTRING "${ALL_AXES_CODE}" 0 ${TRIM_LENGTH} ALL_AXES_CODE)
endif()

# Generate the header content
set(HEADER_CONTENT "// AUTO-GENERATED FILE - DO NOT EDIT
// Generated from: bench/config/operators/${OPERATOR}.json
// Operator: ${OPERATOR}
// Generation script: bench/cpp/GenerateBenchConfig.cmake
//
// To modify benchmark parameters:
//   1. Edit bench/config/operators/${OPERATOR}.json
//   2. Rebuild the project (cmake --build)
//
// This file provides default axis configurations for the ${OPERATOR_CAMEL} operator benchmark.
// CLI arguments can still override these defaults at runtime.
//
// dtypes and per-axis values are unioned across ALL entries in
// the operator JSON whose `benchmark` field equals \"${OPERATOR}\". This
// keeps the compile-time axis registry a superset of every value any tier
// asks for, so --tier-driven --axis overrides at runtime always bind to
// existing axes.

#ifndef CVCUDA_BENCH_${OPERATOR_UPPER}_CONFIG_HPP
#define CVCUDA_BENCH_${OPERATOR_UPPER}_CONFIG_HPP

#include <nvbench/nvbench.cuh>

// Auto-generated type list from dtypes in the operator config JSON
// Use BENCH_${OPERATOR_UPPER}_TYPES in NVBENCH_BENCH_TYPES to use config-based types
#define BENCH_${OPERATOR_UPPER}_TYPES nvbench::type_list<${TYPES_STR}>

// Number of warmup iterations to run before benchmarking (0 = disabled)
#define BENCH_${OPERATOR_UPPER}_WARMUP_ITERATIONS ${WARMUP_ITERATIONS}

#define BENCH_${OPERATOR_UPPER}_AXES \\
    .set_type_axes_names({\"InOutDataType\"}) \\
${ALL_AXES_CODE}

#endif // CVCUDA_BENCH_${OPERATOR_UPPER}_CONFIG_HPP
")

# Write the generated file
file(WRITE "${OUTPUT_FILE}" "${HEADER_CONTENT}")
