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

find_program(CCACHE_EXEC ccache)

if(CCACHE_EXEC)
    if(NOT CCACHE_STATSLOG)
        set(CCACHE_STATSLOG ${CMAKE_BINARY_DIR}/ccache_stats.log)
    endif()
    set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES ${CCACHE_STATSLOG})

    # Absolute source paths are part of ccache's hash, so the same file compiled
    # from two checkouts misses. CCACHE_BASEDIR rewrites paths below it before
    # hashing, while CCACHE_NOHASHDIR excludes the working directory from the key.
    # Deriving the base from Git's common directory keeps linked checkouts aligned;
    # deriving it from CMAKE_SOURCE_DIR would give each linked checkout a different
    # base and silently defeat cache sharing.
    #
    # CI exports CCACHE_BASEDIR to its own build root, which is wider still, so
    # the environment wins over both; otherwise the driver would clobber it.
    if(NOT CCACHE_BASEDIR)
        if(DEFINED ENV{CCACHE_BASEDIR})
            set(CCACHE_BASEDIR "$ENV{CCACHE_BASEDIR}")
        else()
            # find_package(Git), not find_program: GetGitRevisionDescription.cmake
            # already populates GIT_EXECUTABLE, and a toolchain that overrides it
            # should win here too.
            find_package(Git QUIET)
            set(_git_common_dir "")
            if(GIT_EXECUTABLE)
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} rev-parse --path-format=absolute --git-common-dir
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                    OUTPUT_VARIABLE _git_common_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
            endif()
            if(_git_common_dir)
                # <root>/<checkout>/.git -> <root>
                get_filename_component(_primary_checkout "${_git_common_dir}" DIRECTORY)
                get_filename_component(CCACHE_BASEDIR "${_primary_checkout}" DIRECTORY)
            else()
                # Not a git checkout (a source tarball); the source dir's parent
                # is the best available guess.
                get_filename_component(CCACHE_BASEDIR "${CMAKE_SOURCE_DIR}" DIRECTORY)
            endif()
        endif()
    endif()

    set(compiler_driver ${CMAKE_BINARY_DIR}/compiler_driver.sh)
    file(WRITE ${compiler_driver}
"#!/bin/bash
exec env CCACHE_STATSLOG=${CCACHE_STATSLOG} \
CCACHE_BASEDIR=${CCACHE_BASEDIR} \
CCACHE_NOHASHDIR=1 \
${CCACHE_EXEC} \"$@\"
")
    execute_process(COMMAND "chmod" "755" ${compiler_driver})

    set(CMAKE_CXX_COMPILER_LAUNCHER ${compiler_driver})
    set(CMAKE_C_COMPILER_LAUNCHER ${compiler_driver})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${compiler_driver})

    message(STATUS "ccache: sharing across checkouts under ${CCACHE_BASEDIR}")
    # This build fans every translation unit out over CMAKE_CUDA_ARCHITECTURES,
    # so objects are large and ccache's 5G default evicts faster than it reuses.
    # Reported rather than set: max_size is global to the user's cache, not this
    # build's to change.
    execute_process(COMMAND ${CCACHE_EXEC} --get-config max_size
                    OUTPUT_VARIABLE ccache_max_size
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(ccache_max_size)
        message(STATUS "ccache: max_size is ${ccache_max_size}"
                       " (raise it with `ccache --max-size=100G` if hit rates are low)")
    endif()
endif()
