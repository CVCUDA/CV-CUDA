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


# Create the binary packages for all python versions supported
foreach(VER ${PYTHON_VERSIONS})
    string(REPLACE "." "" VERNAME ${VER})

    set(python_module_name python${VER})
    # NSIS doesn't like the dot
    string(REPLACE "." "" python_module_name "${python_module_name}")
    string(TOUPPER ${python_module_name} PYTHON_MODULE_NAME)

    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_DISABLED true)
    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_DISPLAY_NAME "Python ${VER}")
    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_DESCRIPTION "NVIDIA NVCV python ${VER} bindings")
    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_GROUP python)

    if(UNIX)
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_NAME python${VER}-${CPACK_PACKAGE_NAME})

        set(NVCV_${PYTHON_MODULE_NAME}_FILE_NAME "nvcv-python${VER}-${NVCV_VERSION_BUILD}")
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_FILE_NAME "${NVCV_${PYTHON_MODULE_NAME}_FILE_NAME}.deb")
        set(CPACK_ARCHIVE_${PYTHON_MODULE_NAME}_FILE_NAME "${NVCV_${PYTHON_MODULE_NAME}_FILE_NAME}")

        # Depend on current or any future ABI with same major version
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")

        # Depend on python interpreter
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_DEPENDS "${CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_DEPENDS}, python${VER}")

        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_CONTROL_EXTRA
            "${CMAKE_BINARY_DIR}/python${VER}/build/cpack/postinst"
            "${CMAKE_BINARY_DIR}/python${VER}/build/cpack/prerm")
    endif()

    # Execute module's installer script
    install(CODE "include(\"${CMAKE_BINARY_DIR}/python${VER}/build/cmake_install.cmake\")"
            COMPONENT ${python_module_name})

    if(BUILD_TESTS)
        set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS
                "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS},
                ${CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")

        # For some reason these are needed with python-3.7
        if(VER VERSION_EQUAL "3.7")
            set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS
                    "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS}
                     , python3-typing-extensions")
        endif()
    endif()

    list(APPEND CPACK_COMPONENTS_ALL ${python_module_name})
endforeach()

if(BUILD_TESTS)
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS
            "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS}, python3-pytest")
endif()
