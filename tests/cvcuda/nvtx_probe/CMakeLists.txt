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

# MODULE: the probe is only ever dlopen()ed (by NVTX, via NVTX_INJECTION64_PATH), never linked.
add_library(cvcuda_nvtx_probe MODULE NvtxProbe.cpp)

target_link_libraries(cvcuda_nvtx_probe PRIVATE cvcuda_nvtx_config)

set_target_properties(cvcuda_nvtx_probe PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Install next to the python tests so packaged (DEB/TAR) test runs can inject it too. Must match
# PYTHON_TEST_INSTDIR in ../python/CMakeLists.txt.
install(TARGETS cvcuda_nvtx_probe
        LIBRARY DESTINATION "${CMAKE_INSTALL_DATADIR}/python/cvcuda"
        COMPONENT tests)
