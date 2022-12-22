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

list(APPEND CPACK_COMPONENTS_ALL tests)

set(CPACK_COMPONENT_TESTS_DISABLED true)
set(CPACK_COMPONENT_TESTS_DISPLAY_NAME "Tests")
set(CPACK_COMPONENT_TESTS_DESCRIPTION "NVIDIA CV-CUDA test suite (internal use only)")
set(CPACK_COMPONENT_TESTS_GROUP internal)

if(UNIX)
    # Depend on current or any future ABI with same major version
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")
    # External dependencies
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS},libssl3")

    set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "cvcuda${PROJECT_VERSION_MAJOR}-tests")

    set(CVCUDA_TESTS_FILE_NAME "cvcuda-tests-${CVCUDA_VERSION_BUILD}")

    set(CPACK_DEBIAN_TESTS_FILE_NAME "${CVCUDA_TESTS_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_TESTS_FILE_NAME "${CVCUDA_TESTS_FILE_NAME}")

else()
    set(CPACK_COMPONENT_TESTS_DEPENDS lib)
endif()
