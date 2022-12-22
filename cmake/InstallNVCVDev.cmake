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

list(APPEND CPACK_COMPONENTS_ALL dev)

set(CPACK_COMPONENT_DEV_DISPLAY_NAME "Development")
set(CPACK_COMPONENT_DEV_DESCRIPTION "NVIDIA CV-CUDA C/C++ development library and headers")

if(UNIX)
    set(NVCV_DEV_FILE_NAME "nvcv-dev-${NVCV_VERSION_BUILD}")

    set(CPACK_DEBIAN_DEV_FILE_NAME "${NVCV_DEV_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_DEV_FILE_NAME "${NVCV_DEV_FILE_NAME}")

    # dev package works with any current and futures ABIs, provided major version
    # is the same
    set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")

    set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${NVCV_PACKAGE_NAME}-dev")

    # We're not adding compiler and cmake as dependencies, users can choose
    # whatever toolchain they want.

    # Set up control files
    set(CVCUDA_USR_LIB_DIR /usr/lib)

    set(args -DCVCUDA_SOURCE_DIR=${PROJECT_SOURCE_DIR}
             -DCVCUDA_BINARY_DIR=${PROJECT_BINARY_DIR}
             -DNVCV_LIB_LINKER_FILE_NAME=$<TARGET_LINKER_FILE_NAME:nvcv_types>)

    foreach(var CMAKE_INSTALL_PREFIX
                CMAKE_INSTALL_INCLUDEDIR
                CMAKE_INSTALL_LIBDIR
                NVCV_TYPES_PACKAGE_NAME
                CVCUDA_PACKAGE_NAME
                CMAKE_LIBRARY_ARCHITECTURE
                NVCV_VERSION_API_CODE
                CVCUDA_USR_LIB_DIR)

        list(APPEND args "-D${var}=${${var}}")
    endforeach()

    add_custom_target(nvcv_dev_control_extra ALL
        COMMAND cmake ${args} -DSOURCE=${PROJECT_SOURCE_DIR}/cpack/debian_dev_prerm.in -DDEST=cpack/dev/prerm -P ${PROJECT_SOURCE_DIR}/cpack/ConfigureFile.cmake
        COMMAND cmake ${args} -DSOURCE=${PROJECT_SOURCE_DIR}/cpack/debian_dev_postinst.in -DDEST=cpack/dev/postinst -P ${PROJECT_SOURCE_DIR}/cpack/ConfigureFile.cmake
        BYPRODUCTS cpack/dev/prerm cpack/dev/postinst
        DEPENDS cpack/debian_dev_prerm.in cpack/debian_dev_postinst.in
        VERBATIM)

    set(CPACK_DEBIAN_DEV_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/dev/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/dev/prerm")
else()
    set(CPACK_COMPONENT_DEV_DEPENDS lib)
endif()
