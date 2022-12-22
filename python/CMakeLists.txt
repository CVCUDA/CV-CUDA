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

cmake_minimum_required(VERSION 3.22)

project(nvcv_python CXX C)

set(CMAKE_CXX_STANDARD 20)

include(../cmake/ConfigCompiler.cmake)

set_directory_properties(PROPERTIES COMPILE_OPTIONS "${COMPILE_WARNINGS}")

set(CMAKE_POSITION_INDEPENDENT_CODE true)

set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
add_subdirectory(${PYBIND11_SOURCE_DIR} pybind11)

find_package(nvcv_types REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/util nvcv_util)

set(PYTHON_MODULE_NAME python${PYTHON_VERSION})
string(REPLACE "." "" PYTHON_MODULE_NAME "${PYTHON_MODULE_NAME}")

include(GNUInstallDirs)
set(PYTHON_MODULE_FILENAME_LIST "" CACHE INTERNAL "")
function(nvcv_python_add_module)
    cmake_parse_arguments(ARG "SHARED;MODULE" "TARGET;OUTPUT_NAME" "SOURCES" ${ARGV})

    if(ARG_SHARED AND ARG_MODULE)
        message(FATAL_ERROR "Must specify either SHARED or MODULE")
    elseif(ARG_SHARED)
        set(type "SHARED")
    else()
        set(type "MODULE")
    endif()

    pybind11_add_module(${ARG_TARGET} ${type} ${ARG_SOURCES})

    set_target_properties(${ARG_TARGET} PROPERTIES
        DEBUG_POSTFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )

    if(ARG_OUTPUT_NAME)
        set_target_properties(${ARG_TARGET} PROPERTIES
            OUTPUT_NAME "${ARG_OUTPUT_NAME}"
        )
    else()
        set(ARG_OUTPUT_NAME ${ARG_TARGET})
    endif()

    target_link_libraries(${ARG_TARGET}
        PRIVATE
            -static-libstdc++
    )

    get_target_property(prefix ${ARG_TARGET} PREFIX)
    get_target_property(suffix ${ARG_TARGET} SUFFIX)

    set(PYTHON_MODULE_FILENAME_LIST
        "${PYTHON_MODULE_FILENAME_LIST};${prefix}${ARG_OUTPUT_NAME}${suffix}" CACHE INTERNAL "")

    install(TARGETS ${ARG_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
        COMPONENT ${PYTHON_MODULE_NAME}
)
endfunction()

add_subdirectory(common)
add_subdirectory(mod_nvcv)
add_subdirectory(mod_cvcuda)

# Set up debian control files
set(SRC_PYTHON_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/python)
set(DST_PYTHON_MODULE_PATH /usr/lib/python3/dist-packages)
string(JOIN " " PYTHON_MODULE_FILENAME_LIST ${PYTHON_MODULE_FILENAME_LIST})

configure_file(cpack/debian_python_postinst.in cpack/postinst @ONLY)
configure_file(cpack/debian_python_prerm.in cpack/prerm @ONLY)
