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

# Global configuration =========================================

if(UNIX)
    set(CPACK_SYSTEM_NAME "x86_64-linux")
else()
    message(FATAL_ERROR "Architecture not supported")
endif()

set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_CONTACT "CV-CUDA Support <cv-cuda@exchange.nvidia.com>")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://confluence.nvidia.com/display/CVCUDA")

# ARCHIVE installer doesn't work with absolute install destination
# we have to error out in this case
set(CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)

set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_TWEAK "${PROJECT_VERSION_TWEAK}")
set(CPACK_PACKAGE_VERSION_SUFFIX "${PROJECT_VERSION_SUFFIX}")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_MONOLITHIC_INSTALL OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CPACK_STRIP_FILES false)
else()
    set(CPACK_STRIP_FILES true)
endif()

set(CPACK_VERBATIM_VARIABLES true)
set(CPACK_GENERATOR TXZ)
set(CPACK_THREADS 0) # use all cores
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# we split the file name components with '-', so the version component can't
# have this character, let's replace it by '_'
string(REPLACE "-" "_" tmp ${CPACK_PACKAGE_VERSION})
set(CVCUDA_VERSION_BUILD "${tmp}-${CVCUDA_BUILD_SUFFIX}")

set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${CVCUDA_VERSION_BUILD}")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}${PROJECT_VERSION_MAJOR}")

# CI needs this VERSION file to select the correct installer packages
add_custom_target(cvcuda_version_file ALL
    COMMAND ${CMAKE_COMMAND} -E echo ${CVCUDA_VERSION_BUILD} > ${cvcuda_BINARY_DIR}/VERSION)

if(UNIX)
    set(CPACK_GENERATOR ${CPACK_GENERATOR} DEB)

    set(CPACK_COMPONENTS_GROUPING IGNORE)

    # Debian options ----------------------------------------
    set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS true)
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION ON)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS off)
    set(CPACK_DEBIAN_COMPRESSION_TYPE xz)

    # Create several .debs, one for each component
    set(CPACK_DEB_COMPONENT_INSTALL ON)

    # Archive options -----------------------------------
    set(CPACK_ARCHIVE_THREADS 0) # use all cores
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
endif()

# Configure installer components ================================================

set(CPACK_COMPONENTS_ALL "")

include(InstallNVCVLib)
include(InstallNVCVDev)

if(BUILD_TESTS)
    include(InstallTests)
endif()

if(BUILD_PYTHON)
    include(InstallPython)
endif()

if(BUILD_SAMPLES)
    include(InstallSamples)
endif()

# Finish up GPack configuration =================================================

include(CPack)

cpack_add_component_group(internal DISPLAY_NAME Internal DESCRIPTION "Internal packages, do not distribute")
