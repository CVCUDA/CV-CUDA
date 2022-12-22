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

# nvcv_types private implementation
add_subdirectory(priv)

add_library(nvcv_types SHARED
    Config.cpp
    Version.cpp
    Status.cpp
    Allocator.cpp
    Requirements.cpp
    Image.cpp
    ImageBatch.cpp
    Tensor.cpp
    TensorShape.cpp
    TensorLayout.cpp
    ColorSpec.cpp
    DataLayout.cpp
    DataType.cpp
    ImageFormat.cpp
)

target_link_libraries(nvcv_types
    PRIVATE
        nvcv_types_priv
        nvcv_util
)

target_include_directories(nvcv_types
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

configure_version(nvcv_types NVCV nvcv ${PROJECT_VERSION})
configure_symbol_versioning(nvcv_types NVCV nvcv_types)
setup_dso(nvcv_types ${PROJECT_VERSION})

target_compile_definitions(nvcv_types PRIVATE -DNVCV_EXPORTING=1)

# target used when only public headers are needed
add_library(nvcv_types_headers INTERFACE)
target_include_directories(nvcv_types_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)

# Installer

install(TARGETS nvcv_types
        EXPORT nvcv_types
        COMPONENT lib
        LIBRARY NAMELINK_COMPONENT dev)

install(DIRECTORY include/nvcv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inc")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(nvcv_types-config-version.cmake
                                 COMPATIBILITY SameMajorVersion)

if(UNIX)
    install(EXPORT nvcv_types
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nvcv_types"
        FILE nvcv_types-config.cmake
        COMPONENT dev)

    # WAR for https://gitlab.kitware.com/cmake/cmake/-/issues/23563
    install(CODE "set(CVCUDA_CONFIG_PATH \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/nvcv_types/nvcv_types-config.cmake\")"
        COMPONENT dev)

    install(CODE [=[
        file(READ "${CVCUDA_CONFIG_PATH}" contents)
        string(REPLACE "get_filename_component(_IMPORT_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)"
[[
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" REALPATH)
]]
            contents "${contents}")
        file(WRITE "${CVCUDA_CONFIG_PATH}" "${contents}")
        ]=]
        COMPONENT dev)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nvcv_types-config-version.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nvcv_types"
        COMPONENT dev)
endif()
