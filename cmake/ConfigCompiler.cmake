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

set(CMAKE_CXX_STANDARD 20)
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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    message(FATAL_ERROR "Must use gcc>=11.1 to compile CV-CUDA, you're using ${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
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
