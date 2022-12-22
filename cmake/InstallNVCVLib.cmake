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

list(APPEND CPACK_COMPONENTS_ALL lib)
set(CPACK_COMPONENT_LIB_DISPLAY_NAME "Runtime libraries")
set(CPACK_COMPONENT_LIB_DESCRIPTION "NVIDIA NVCV library")
set(CPACK_COMPONENT_LIB_REQUIRED true)

set(NVCV_PACKAGE_NAME "nvcv${NVCV_VERSION_MAJOR}")
set(NVCV_TYPES_PACKAGE_NAME "nvcv_types${NVCV_VERSION_MAJOR}")
set(CVCUDA_PACKAGE_NAME "cvcuda${NVCV_VERSION_MAJOR}")

if(UNIX)
    set(NVCV_LIB_FILE_NAME "nvcv-lib-${NVCV_VERSION_BUILD}")

    set(CPACK_DEBIAN_LIB_FILE_NAME "${NVCV_LIB_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_LIB_FILE_NAME "${NVCV_LIB_FILE_NAME}")

    configure_file(cpack/debian_lib_postinst.in cpack/lib/postinst @ONLY)
    configure_file(cpack/debian_lib_prerm.in cpack/lib/prerm @ONLY)

    set(CPACK_DEBIAN_LIB_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/prerm")

    # as per debian convention, use the library file name
    set(CPACK_DEBIAN_LIB_PACKAGE_NAME "lib${NVCV_PACKAGE_NAME}")

    set(CPACK_DEBIAN_LIB_PACKAGE_DEPENDS "libstdc++6, libc6")

    if(ENABLE_SANITIZER)
        set(CPACK_DEBIAN_LIB_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_DEPENDS}, libasan6")
    endif()

    configure_file(cpack/ld.so.conf.in cpack/ld.so.conf @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpack/ld.so.conf
        DESTINATION "etc/ld.so.conf.d"
        RENAME ${CPACK_PACKAGE_NAME}.conf
        COMPONENT lib)
endif()

# Handle licenses, they go together with the library
install(FILES ${CPACK_RESOURCE_FILE_LICENSE}
    DESTINATION doc
    RENAME CVCUDA_EULA.txt
    COMPONENT lib)
