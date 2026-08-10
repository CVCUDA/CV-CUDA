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

# sanitized builds of our python module don't work,
# python interpreter exits with errorcode 1, nothing else
# is output
if(ENABLE_SANITIZER)
    return()
endif()

file(GLOB SOURCES CONFIGURE_DEPENDS "*.py")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MODULE_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# create the test script for our build tree
set(PYTHON_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
string(REPLACE ";" " " PYTHON_TEST_VERSIONS "${PYTHON_VERSIONS}")

set(PYTHON_MODULE_DIR ${CMAKE_BINARY_DIR}/python3)
# Probe path in the build tree (target cvcuda_nvtx_probe in ../nvtx_probe).
set(NVTX_PROBE_PATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_MODULE_PREFIX}cvcuda_nvtx_probe${CMAKE_SHARED_MODULE_SUFFIX}")
configure_file(cvcuda_test_python.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cvcuda_test_python @ONLY)

set(PYTHON_TEST_INSTDIR "${CMAKE_INSTALL_DATADIR}/python/cvcuda")

install(FILES ${SOURCES}
        DESTINATION "${PYTHON_TEST_INSTDIR}"
        COMPONENT tests)

# create the test script that will get installed
set(PYTHON_TEST_DIR ${CMAKE_INSTALL_PREFIX}/${PYTHON_TEST_INSTDIR})
# Probe is installed alongside the python tests (see ../nvtx_probe/CMakeLists.txt).
set(NVTX_PROBE_PATH "${CMAKE_INSTALL_PREFIX}/${PYTHON_TEST_INSTDIR}/${CMAKE_SHARED_MODULE_PREFIX}cvcuda_nvtx_probe${CMAKE_SHARED_MODULE_SUFFIX}")
# CRITICAL: Python modules are installed to ${CMAKE_INSTALL_LIBDIR}/python/, not just ${CMAKE_INSTALL_LIBDIR}/
# The /python suffix is required for the test script to correctly set PYTHONPATH and find cvcuda/nvcv modules.
# Without this suffix, Python will fail with "ModuleNotFoundError: No module named 'cvcuda'"
#
# Why this matters:
# - DEB packages install to system paths (/usr/lib/python3.X/dist-packages/) which are in Python's default search path
# - TAR archives install to custom paths (/opt/nvidia/cvcuda*/lib/python/) which require PYTHONPATH to be set
# - Removing /python will cause TAR tests to fail while DEB tests still pass, making the bug subtle to detect
set(PYTHON_MODULE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/python)
configure_file(cvcuda_test_python.in cvcuda_test_python @ONLY)

nvcv_add_test(${CMAKE_CURRENT_BINARY_DIR}/cvcuda_test_python cvcuda python)
