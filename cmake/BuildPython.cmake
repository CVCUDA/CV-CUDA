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

include(ExternalProject)

# Where our python module installed, it'll end up being in the same
# directory nvcv shared library resides
set(PYPROJ_COMMON_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
                       -Dnvcv_types_ROOT=${CMAKE_CURRENT_BINARY_DIR}/cmake)

if(CMAKE_BUILD_TYPE)
    list(APPEND PYPROJ_COMMON_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

get_target_property(NVCV_TYPES_SOURCE_DIR nvcv_types SOURCE_DIR)

# Needed so that nvcv_types library's build path gets added
# as RPATH to the plugin module. When outer project gets installed,
# it shall overwrite the RPATH with the final installation path.
list(APPEND PYPROJ_COMMON_ARGS
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=true
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=true
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_MODULE_PATH=${CMAKE_CURRENT_BINARY_DIR}/cmake
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    -DNVCV_TYPES_SOURCE_DIR=${NVCV_TYPES_SOURCE_DIR}
    -DPYBIND11_SOURCE_DIR=${PYBIND11_SOURCE_DIR}
    -DWARNINGS_AS_ERRORS=${WARNINGS_AS_ERRORS}
    -DENABLE_COMPAT_OLD_GLIBC=${ENABLE_COMPAT_OLD_GLIBC}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

foreach(VER ${PYTHON_VERSIONS})
    set(BASEDIR ${CMAKE_CURRENT_BINARY_DIR}/python${VER})

    ExternalProject_Add(nvcv_python${VER}
        PREFIX ${BASEDIR}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python
        CMAKE_ARGS ${PYPROJ_COMMON_ARGS} -DPYTHON_VERSION=${VER}
        BINARY_DIR ${BASEDIR}/build
        TMP_DIR ${BASEDIR}/tmp
        STAMP_DIR ${BASEDIR}/stamp
        BUILD_ALWAYS true
        DEPENDS nvcv_types cvcuda
        INSTALL_COMMAND ""
    )
endforeach()
