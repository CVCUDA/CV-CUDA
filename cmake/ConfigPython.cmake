# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(ENABLE_SANITIZER)
    message(FATAL_ERROR "CV-CUDA python modules don't work on sanitized builds")
endif()

# Because we python as subproject, we need to create a fake Findnvcv.cmake so
# that find_package will find our local nvcv_types library and headers
set(FINDNVCV_TYPES_CONTENTS
[=[
add_library(nvcv_types SHARED IMPORTED)
target_include_directories(nvcv_types
    INTERFACE
    "$<TARGET_PROPERTY:nvcv_types,INTERFACE_INCLUDE_DIRECTORIES>"
)
add_library(nvcv_types_headers INTERFACE IMPORTED)
target_include_directories(nvcv_types_headers
    INTERFACE
    "$<TARGET_PROPERTY:nvcv_types,INTERFACE_INCLUDE_DIRECTORIES>"
)
]=])

set(FINDCVCUDA_CONTENTS
[=[
add_library(cvcuda SHARED IMPORTED)
target_include_directories(cvcuda
    INTERFACE
    "$<TARGET_PROPERTY:cvcuda,INTERFACE_INCLUDE_DIRECTORIES>"
)
]=])

if(CMAKE_CONFIGURATION_TYPES)
    set(NVCV_CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES})
else()
    set(NVCV_CONFIG_TYPES ${CMAKE_BUILD_TYPE})
endif()

foreach(cfg ${NVCV_CONFIG_TYPES})
    string(TOLOWER ${cfg} cfg_lower)
    set(FINDNVCV_TYPES_CONTENTS
"${FINDNVCV_TYPES_CONTENTS}include(nvcv_types_${cfg_lower})
")
    set(FINDCVCUDA_CONTENTS
"${FINDCVCUDA_CONTENTS}include(cvcuda_${cfg_lower})
")
endforeach()

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/Findnvcv_types.cmake CONTENT "${FINDNVCV_TYPES_CONTENTS}")
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/Findcvcuda.cmake CONTENT "${FINDCVCUDA_CONTENTS}")

list(LENGTH "${NVCV_CONFIG_TYPES}" num_configs)
if(${num_configs} EQUAL 1)
    set(NVCV_BUILD_SUFFIX "")
else()
    set(NVCV_BUILD_SUFFIX "_$<UPPER_CASE:$<CONFIG>>")
endif()

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcv_types_$<LOWER_CASE:$<CONFIG>>.cmake CONTENT
"set_target_properties(nvcv_types PROPERTIES IMPORTED_LOCATION${NVCV_BUILD_SUFFIX} \"$<TARGET_FILE:nvcv_types>\"
                                       IMPORTED_IMPLIB${NVCV_BUILD_SUFFIX} \"$<TARGET_LINKER_FILE:nvcv_types>\")
")

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/cvcuda_$<LOWER_CASE:$<CONFIG>>.cmake CONTENT
"set_target_properties(cvcuda PROPERTIES IMPORTED_LOCATION${NVCV_BUILD_SUFFIX} \"$<TARGET_FILE:cvcuda>\"
                                                 IMPORTED_IMPLIB${NVCV_BUILD_SUFFIX} \"$<TARGET_LINKER_FILE:cvcuda>\")
")

# Warn if PYTHON_VERSIONS is set but BUILD_PYTHON is disabled
if(PYTHON_VERSIONS AND NOT BUILD_PYTHON)
    message(WARNING "PYTHON_VERSIONS is set to '${PYTHON_VERSIONS}' but BUILD_PYTHON is OFF. Python bindings will not be built.")
endif()

# Python versions to build already set?
if(PYTHON_VERSIONS)
    set(USE_DEFAULT_PYTHON false)

    set(AVAILABLE_PYTHON_VERSIONS "")
    foreach(VER ${PYTHON_VERSIONS})
        # If we only search on the /usr/bin and /usr/local/bin, we may not find the correct Python version
        # if this is called with a virtual environment as opposed to the system Python
        # removing the NO_DEFAULT_PATH allows finding the Python in a virtual environment
        # find_program(PYTHON_EXECUTABLE python${VER} PATHS /usr/bin /usr/local/bin NO_DEFAULT_PATH)
        find_program(PYTHON_EXECUTABLE python${VER} /usr/bin /usr/local/bin)
        if (PYTHON_EXECUTABLE)
            execute_process(
                COMMAND ${PYTHON_EXECUTABLE} -c "import sys, sysconfig; print(f'{sys.version_info.major}.{sys.version_info.minor}' + ('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))"
                OUTPUT_VARIABLE PYTHON_VERSION_OUTPUT
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if (PYTHON_VERSION_OUTPUT STREQUAL ${VER})
                list(APPEND AVAILABLE_PYTHON_VERSIONS ${VER})
            else()
                message(WARNING "Python executable ${PYTHON_EXECUTABLE} does not match version ${VER} (${PYTHON_VERSION_OUTPUT}). Skipping.")
            endif()
        else()
            message(WARNING "Python version ${VER} not found. Skipping.")
        endif()
        unset(PYTHON_EXECUTABLE CACHE)
    endforeach()

    if(NOT AVAILABLE_PYTHON_VERSIONS)
        message(FATAL_ERROR "No available Python versions found. Exiting.")
    endif()

    set(PYTHON_VERSIONS ${AVAILABLE_PYTHON_VERSIONS})
    unset(AVAILABLE_PYTHON_VERSIONS)
# If not, gets the default version from FindPython
else()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(PYTHON_VERSIONS ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))"
                    OUTPUT_VARIABLE PYTHON_FREE_THREADED OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYTHON_FREE_THREADED STREQUAL "True")
        string(APPEND PYTHON_VERSIONS "t")
    endif()
    set(USE_DEFAULT_PYTHON true)
endif()
