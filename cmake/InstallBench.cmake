# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CPACK_COMPONENTS_ALL bench)

# Enable bench package when benchmarks are being built
# This file is only included when BUILD_BENCH is ON
set(CPACK_COMPONENT_BENCH_DISABLED false)
set(CPACK_COMPONENT_BENCH_DISPLAY_NAME "Benchmarks")
set(CPACK_COMPONENT_BENCH_DESCRIPTION "NVIDIA CV-CUDA benchmark suite")
set(CPACK_COMPONENT_BENCH_GROUP internal)

# Depend on current or any future ABI with same major version
set(CPACK_DEBIAN_BENCH_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")
set(CPACK_DEBIAN_BENCH_PACKAGE_NAME "cvcuda${PROJECT_VERSION_MAJOR}-bench")
set(CVCUDA_BENCH_FILE_NAME "cvcuda-bench-${CVCUDA_VERSION_BUILD}")
set(CPACK_DEBIAN_BENCH_FILE_NAME "${CVCUDA_BENCH_FILE_NAME}.deb")
set(CPACK_ARCHIVE_BENCH_FILE_NAME "${CVCUDA_BENCH_FILE_NAME}")
