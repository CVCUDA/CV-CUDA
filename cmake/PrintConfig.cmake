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

message(STATUS "General configuration for CVCUDA-${PROJECT_VERSION}")
message(STATUS "")
message(STATUS "Build options")
message(STATUS "    CMAKE_INSTALL_PREFIX     : ${CMAKE_INSTALL_PREFIX}")

if(WARNINGS_AS_ERRORS)
    message(STATUS "    WARNINGS_AS_ERRORS       : ON")
else()
    message(STATUS "    WARNINGS_AS_ERRORS       : off")
endif()

if(ENABLE_COMPAT_OLD_GLIBC)
    message(STATUS "    ENABLE_COMPAT_OLD_GLIBC  : ON")
else()
    message(STATUS "    ENABLE_COMPAT_OLD_GLIBC  : off")
endif()

if(BUILD_TESTS)
    message(STATUS "    BUILD_TESTS              : ON")
else()
    message(STATUS "    BUILD_TESTS              : off")
endif()

if(BUILD_PYTHON)
    message(STATUS "    BUILD_PYTHON             : ON")
    message(STATUS "        Python versions : ${PYTHON_VERSIONS}")
else()
    message(STATUS "    BUILD_PYTHON             : off")
endif()

if(ENABLE_SANITIZER)
    message(STATUS "    ENABLE_SANITIZER         : ON")
else()
    message(STATUS "    ENABLE_SANITIZER         : off")
endif()

# Compilation

message(STATUS "")
message(STATUS "Platform")
message(STATUS "    Host             : ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "    Target           : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "    CMake            : ${CMAKE_VERSION}")
message(STATUS "    CMake generator  : ${CMAKE_GENERATOR}")
message(STATUS "    CMake build tool : ${CMAKE_BUILD_TOOL}")
message(STATUS "    Configuration    : ${CMAKE_BUILD_TYPE}")
message(STATUS "    ccache           : ${CCACHE_EXEC}")
message(STATUS "    ccache stats log : ${CCACHE_STATSLOG}")

message(STATUS "")

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
get_directory_property(dir_defs_ COMPILE_DEFINITIONS)
set(dir_defs "")
foreach(def ${dir_defs_})
    set(dir_defs "${dir_defs} -D${def}")
endforeach()
get_directory_property(dir_opt COMPILE_OPTIONS)

message(STATUS "Default compiler/linker config")
message(STATUS "    C++ Compiler : ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "    C++ Standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "    C++ Flags    : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE}}")
message(STATUS "")
message(STATUS "    C Compiler   : ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_VERSION})")
message(STATUS "    C Flags      : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE}}")
message(STATUS "")
message(STATUS "    CUDA Compiler : ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
message(STATUS "    CUDA Arch     : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "    CUDA flags    : ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
message(STATUS "    CUDA toolkit target dir : ${CUDAToolkit_TARGET_DIR}")
message(STATUS "")
message(STATUS "    Compiler Options    : ${dir_opt}")
message(STATUS "    Definitions         : ${dir_defs}")
message(STATUS "")
message(STATUS "    Linker flags (exec) : ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${BUILD_TYPE}}")
message(STATUS "    Linker flags (lib)  : ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${BUILD_TYPE}}")
message(STATUS "")
message(STATUS "    Link-time optim.: supported ${LTO_SUPPORTED}, enabled ${LTO_ENABLED}")

message(STATUS "")
