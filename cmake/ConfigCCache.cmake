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

find_program(CCACHE_EXEC ccache)

if(CCACHE_EXEC)
    if(NOT CCACHE_STATSLOG)
        set(CCACHE_STATSLOG ${CMAKE_BINARY_DIR}/ccache_stats.log)
    endif()
    set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES ${CCACHE_STATSLOG})

    set(compiler_driver ${CMAKE_BINARY_DIR}/compiler_driver.sh)
    file(WRITE ${compiler_driver}
"#!/bin/bash
CCACHE_STATSLOG=${CCACHE_STATSLOG} ${CCACHE_EXEC} $@
")
    file(CHMOD ${compiler_driver} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)

    set(CMAKE_CXX_COMPILER_LAUNCHER ${compiler_driver})
    set(CMAKE_C_COMPILER_LAUNCHER ${compiler_driver})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${compiler_driver})
endif()
