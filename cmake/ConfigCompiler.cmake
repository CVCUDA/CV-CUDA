# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -ggdb")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -ggdb")


if(WARNINGS_AS_ERRORS)
    set(C_WARNING_ERROR_FLAG "-Werror")
    set(CUDA_WARNING_ERROR_FLAG "-Werror all-warnings")
endif()

# Match warning setup with GVS
set(C_WARNING_FLAGS "-Wall -Wno-unknown-pragmas -Wpointer-arith -Wmissing-declarations -Wredundant-decls -Wmultichar -Wno-unused-local-typedefs -Wunused")

# let the compiler help us marking virtual functions with override
set(CXX_WARNING_FLAGS "-Wsuggest-override")

# This is a bogus warning, safe to ignore.
set(CUDA_WARNING_FLAGS "-Wno-tautological-compare")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${C_WARNING_ERROR_FLAG} ${C_WARNING_FLAGS} ${CXX_WARNING_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_WARNING_ERROR_FLAG} ${C_WARNING_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_WARNING_ERROR_FLAG} ${C_WARNING_FLAGS} ${CXX_WARNING_FLAGS} ${CUDA_WARNING_FLAGS}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.4)
    message(FATAL_ERROR "Must use gcc>=9.4 to compile CV-CUDA, you're using ${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"
   # Enable if gcc>=10. With 9.4 in some contexts we hit ICE with LTO:
   # internal compiler error: in add_symbol_to_partition_1, at lto/lto-partition.c:153
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
    set(LTO_ENABLED ON)
else()
    set(LTO_ENABLED OFF)
endif()

if(ENABLE_SANITIZER)
    set(COMPILER_SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fsanitize=leak
        -fsanitize=undefined
        -fno-sanitize-recover=all
        # not properly supported, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64234
        #-static-libasan
        -static-liblsan
        -static-libubsan)
    string(REPLACE ";" " " COMPILER_SANITIZER_FLAGS "${COMPILER_SANITIZER_FLAGS}" )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILER_SANITIZER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMPILER_SANITIZER_FLAGS}")
endif()

if(BUILD_TESTS)
    # Set up the compilers we'll use to check public API compatibility

    # Are they already specified?
    if(PUBLIC_API_COMPILERS)
        # Use them
        set(candidate_compilers ${PUBLIC_API_COMPILERS})
    else()
        # If not, by default, we'll try these.
        set(candidate_compilers gcc-11 gcc-9 gcc-8 clang-11 clang-14)
    endif()

    unset(valid_compilers)

    foreach(comp ${candidate_compilers})
        string(MAKE_C_IDENTIFIER "${comp}" comp_str)
        string(TOUPPER "${comp_str}" COMP_STR)

        find_program(COMPILER_EXEC_${COMP_STR} ${comp})
        if(COMPILER_EXEC_${COMP_STR})
            list(APPEND valid_compilers ${comp})
        else()
            if(PUBLIC_API_COMPILERS)
                message(FATAL_ERROR "Compiler '${comp}' not found")
            else()
                message(WARNING "Compiler '${comp}' not found, skipping public API checks for it")
            endif()
        endif()
    endforeach()
    set(PUBLIC_API_COMPILERS "${valid_compilers}")
endif()

function(add_header_compat_test)
    cmake_parse_arguments(ARG "" "TARGET;SOURCE;DEPENDS;STANDARD" "HEADERS" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "TARGET must be specified")
    endif()

    if(NOT ARG_SOURCE)
        message(FATAL_ERROR "SOURCE must be specified")
    endif()

    if(NOT ARG_DEPENDS)
        message(FATAL_ERROR "DEPENDS must be specified")
    endif()

    if(NOT ARG_STANDARD)
        message(FATAL_ERROR "STANDARD must be specified")
    endif()

    unset(ALL_HEADERS)
    foreach(hdr ${ARG_HEADERS})
        set(ALL_HEADERS "${ALL_HEADERS}#include <${hdr}>\n")
    endforeach()

    # We compile and link the source twice in order to
    # test if headers included from multiple sources causes
    # link errors (multiple definitions). It must not.
    configure_file(${ARG_SOURCE}.in a_${ARG_SOURCE})
    configure_file(${ARG_SOURCE}.in b_${ARG_SOURCE})

    if(${ARG_SOURCE} MATCHES "\.c$")
        set(lang_flag -x c)
    elseif(${ARG_SOURCE} MATCHES "\.cpp$" OR ${ARG_SOURCE} MATCHES "\.cxx$")
        set(lang_flag -x c++)
    else()
        message(FATAL_ERROR "Can't deduce language of '${ARG_SOURCE}'")
    endif()

    unset(${ARG_TARGET}_files)

    foreach(comp ${PUBLIC_API_COMPILERS})
        string(MAKE_C_IDENTIFIER "${comp}" comp_str)
        string(TOUPPER "${comp_str}" COMP_STR)
        if(COMPILER_EXEC_${COMP_STR})
            if(WARNINGS_AS_ERRORS)
                set(extra_flags "-Werror")
            else()
                unset(extra_flags)
            endif()
            set(inc_paths
                "$<FILTER:$<TARGET_GENEX_EVAL:${ARG_DEPENDS},$<TARGET_PROPERTY:${ARG_DEPENDS},INTERFACE_INCLUDE_DIRECTORIES>>,EXCLUDE,'^ *$'>")

            set(bindir ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}/${comp_str})
            file(MAKE_DIRECTORY ${bindir})

            add_custom_command(OUTPUT ${bindir}/${ARG_SOURCE}.d
                COMMAND ${COMPILER_EXEC_${COMP_STR}} ${CMAKE_CURRENT_BINARY_DIR}/a_${ARG_SOURCE} -M -MT ${bindir}/${ARG_SOURCE}.so -MF ${bindir}/${ARG_SOURCE}.d
                        "-I$<JOIN:${inc_paths},;-I>"
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/a_${ARG_SOURCE}
                COMMAND_EXPAND_LISTS
            )
            add_custom_command(OUTPUT ${bindir}/${ARG_SOURCE}.so
                COMMAND ${COMPILER_EXEC_${COMP_STR}}
                        ${lang_flag}
                        -o ${bindir}/${ARG_SOURCE}.so
                        ${extra_flags}
                        -std=${ARG_STANDARD}
                        -Wall -Wextra
                        -fPIC -shared
                        "-I$<JOIN:${inc_paths},;-I>"
                        ${CMAKE_CURRENT_BINARY_DIR}/a_${ARG_SOURCE}
                        ${CMAKE_CURRENT_BINARY_DIR}/b_${ARG_SOURCE}
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/a_${ARG_SOURCE} ${CMAKE_CURRENT_BINARY_DIR}/b_${ARG_SOURCE} ${bindir}/${ARG_SOURCE}.d
                DEPFILE ${bindir}/${ARG_SOURCE}.d
                COMMAND_EXPAND_LISTS
            )
            list(APPEND ${ARG_TARGET}_files ${bindir}/${ARG_SOURCE}.so)
        endif()
    endforeach()

    if(${ARG_TARGET}_files)
        add_custom_target(${ARG_TARGET} ALL DEPENDS ${${ARG_TARGET}_files})
    endif()
endfunction()
