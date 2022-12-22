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

set(CMAKE_DEBUG_POSTFIX "_d")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(GNUInstallDirs)

set(CMAKE_INSTALL_LIBDIR "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

# Executables try to find libnvvpi library relative to themselves.
set(CMAKE_BUILD_RPATH_USE_ORIGIN true)

# Whether assert dumps expose code
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(DEFAULT_EXPOSE_CODE OFF)
else()
    set(DEFAULT_EXPOSE_CODE ON)
endif()

option(EXPOSE_CODE "Expose in resulting binaries parts of our code" ${DEFAULT_EXPOSE_CODE})
option(WARNINGS_AS_ERRORS "Treat compilation warnings as errors" OFF)
option(ENABLE_COMPAT_OLD_GLIBC "Generates binaries that work with old distros, with old glibc" ON)

# Needed to get cuda version
find_package(CUDAToolkit REQUIRED)

# Are we inside a git repo and it has submodules enabled?
if(EXISTS ${CMAKE_SOURCE_DIR}/.git AND EXISTS ${CMAKE_SOURCE_DIR}/.gitmodules)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/.git/modules)
        message(FATAL_ERROR "git submodules not initialized. Did you forget to run 'git submodule update --init'?")
    endif()
endif()

if(UNIX)
    set(CVCUDA_SYSTEM_NAME "x86_64-linux")
else()
    message(FATAL_ERROR "Architecture not supported")
endif()

set(CVCUDA_BUILD_SUFFIX "cuda${CUDAToolkit_VERSION_MAJOR}-${CVCUDA_SYSTEM_NAME}")

function(setup_dso target version)
    string(REGEX MATCHALL "[0-9]+" version_list "${version}")
    list(GET version_list 0 VERSION_MAJOR)
    list(GET version_list 1 VERSION_MINOR)
    list(GET version_list 2 VERSION_PATCH)

    set_target_properties(${target} PROPERTIES
        VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
        SOVERSION "${VERSION_MAJOR}"
    )

    # Reduce executable size ==========================

    # Configure the library linker to remove unused code
    target_link_options(${target} PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,--gc-sections -Wl,--as-needed)
    # Put each function and it's data into separate linker sections
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)

    # Link with static C/C++ libs ==========================
    target_link_libraries(${target} PRIVATE
        -static-libstdc++
        -static-libgcc
    )

    #   Configure symbol visibility ---------------------------------------------
    set_target_properties(${target} PROPERTIES VISIBILITY_INLINES_HIDDEN on
                                               C_VISIBILITY_PRESET hidden
                                               CXX_VISIBILITY_PRESET hidden
                                               CUDA_VISIBILITY_PRESET hidden)
endfunction()
