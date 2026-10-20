# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include_guard(GLOBAL)

find_package(dlpack CONFIG QUIET)
if(NOT dlpack_FOUND)
    # Fall back to fetching the pinned release when dlpack isn't installed.
    message(STATUS "dlpack not found via find_package; fetching v1.3 with FetchContent")
    include(FetchContent)
    FetchContent_Declare(dlpack
        GIT_REPOSITORY https://github.com/dmlc/dlpack.git
        GIT_TAG        84d107bf416c6bab9ae68ad285876600d230490d # immutable commit for v1.3
        GIT_SHALLOW    TRUE
    )
    # We only need the headers; skip dlpack's mock and docs targets.
    set(_saved_build_mock "${BUILD_MOCK}")
    set(_saved_build_docs "${BUILD_DOCS}")
    set(BUILD_MOCK OFF)
    set(BUILD_DOCS OFF)
    FetchContent_MakeAvailable(dlpack)
    set(BUILD_MOCK "${_saved_build_mock}")
    set(BUILD_DOCS "${_saved_build_docs}")
    set(_dlpack_fetched TRUE)
endif()

# dlpack's CMake project version is stuck at 0.6 regardless of release,
# so check the actual version from header macros.
if(_dlpack_fetched)
    set(_dlpack_inc "${dlpack_SOURCE_DIR}/include")
else()
    get_target_property(_dlpack_inc dlpack::dlpack INTERFACE_INCLUDE_DIRECTORIES)
endif()
file(STRINGS "${_dlpack_inc}/dlpack/dlpack.h" _dlpack_major REGEX "#define DLPACK_MAJOR_VERSION")
file(STRINGS "${_dlpack_inc}/dlpack/dlpack.h" _dlpack_minor REGEX "#define DLPACK_MINOR_VERSION")
string(REGEX MATCH "[0-9]+" _dlpack_major "${_dlpack_major}")
string(REGEX MATCH "[0-9]+" _dlpack_minor "${_dlpack_minor}")
if(_dlpack_major LESS 1 OR (_dlpack_major EQUAL 1 AND _dlpack_minor LESS 3))
    message(FATAL_ERROR "DLPack >= 1.3 required, found ${_dlpack_major}.${_dlpack_minor}")
endif()
