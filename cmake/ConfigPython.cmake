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

if(ENABLE_SANITIZERS)
    message(FATAL_ERROR "NVCV python modules don't work on sanitized builds")
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

# Python versions to build already set?
if(PYTHON_VERSIONS)
    set(USE_DEFAULT_PYTHON false)
# If not, gets the default version from FindPython
else()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(PYTHON_VERSIONS ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
    set(USE_DEFAULT_PYTHON true)
endif()
