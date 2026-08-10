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

cmake_minimum_required(VERSION 3.20.1)

project(cvcuda_python CXX C)

set(CMAKE_CXX_STANDARD 20)

include(../cmake/ConfigCompiler.cmake)

set_directory_properties(PROPERTIES COMPILE_OPTIONS "${COMPILE_WARNINGS}")

set(CMAKE_POSITION_INDEPENDENT_CODE true)

string(REGEX MATCH "^[0-9]+\\.[0-9]+" CVCUDA_PYTHON_VERSION "${PYTHON_VERSION}")
set(PYBIND11_PYTHON_VERSION ${CVCUDA_PYTHON_VERSION})
execute_process(
    COMMAND python${CVCUDA_PYTHON_VERSION} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_cmakedir_result
)
if(NOT pybind11_cmakedir_result EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11 cmake dir for python${CVCUDA_PYTHON_VERSION}. "
                        "Is pybind11 installed? Try: python${CVCUDA_PYTHON_VERSION} -m pip install pybind11")
endif()
find_package(pybind11 REQUIRED CONFIG)

find_package(nvcv_types REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/nvcv/util nvcv_util)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/cvcuda/util cvcuda_util)

set(PYTHON_MODULE_NAME python${CVCUDA_PYTHON_VERSION})
string(REPLACE "." "" PYTHON_MODULE_NAME "${PYTHON_MODULE_NAME}")

include(GNUInstallDirs)
set(PYTHON_MODULE_FILENAME_LIST "" CACHE INTERNAL "")

function(cvcuda_python_add_module)
    cmake_parse_arguments(ARG "SHARED;MODULE" "TARGET;OUTPUT_NAME;PACKAGE_DIR" "SOURCES" ${ARGV})

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
        # Set RPATH so Python extensions can find C++ libraries.
        # Install layout: _cvcuda*.so → /opt/nvidia/cvcuda<N>/lib/python/cvcuda/_cvcuda*.so
        #                 libcvcuda.so.<N> → /opt/nvidia/cvcuda<N>/lib/libcvcuda.so.<N>
        # From the .so's perspective $ORIGIN is .../lib/python/cvcuda/, so libcvcuda
        # lives at $ORIGIN/../.. (two dirs up). Without this, tar-installed Python
        # modules fail to load libcvcuda.so on systems where the build tree no
        # longer exists (anything other than the build pod itself).
        BUILD_RPATH "$ORIGIN/../.."
        INSTALL_RPATH "$ORIGIN/../.."
        BUILD_RPATH_USE_ORIGIN ON
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

    # Suppress false positive GCC warnings in pybind11 internals. These appear
    # during LTO, so they must be passed to both compiler and linker:
    #  - stringop-overread: GCC-12 misanalysis of pybind11 string handling.
    #  - odr: pybind11 value-type type_caster specializations (e.g. for
    #    nvcv::ImageFormat, nvcv::DataType, nvcv::Size2D) use PYBIND11_TYPE_CASTER
    #    and so do not derive from type_caster_base like the primary template
    #    does. GCC's LTO ODR check cannot model an explicit specialization as a
    #    distinct type and reports the layout difference as a violation; each
    #    specialization is in fact defined once and identically across all TUs.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${ARG_TARGET}
            PRIVATE
                -Wno-stringop-overread
                -Wno-odr
        )
        target_link_options(${ARG_TARGET}
            PRIVATE
                -Wno-stringop-overread
                -Wno-odr
        )
    endif()

    get_target_property(prefix ${ARG_TARGET} PREFIX)
    get_target_property(suffix ${ARG_TARGET} SUFFIX)

    # Determine install destination - use package subdirectory if specified
    if(ARG_PACKAGE_DIR)
        set(install_dest "${CMAKE_INSTALL_LIBDIR}/python/${ARG_PACKAGE_DIR}")
        set(PYTHON_MODULE_FILENAME_LIST
            "${PYTHON_MODULE_FILENAME_LIST};${ARG_PACKAGE_DIR}/${prefix}${ARG_OUTPUT_NAME}${suffix}" CACHE INTERNAL "")
    else()
        set(install_dest "${CMAKE_INSTALL_LIBDIR}/python")
        set(PYTHON_MODULE_FILENAME_LIST
            "${PYTHON_MODULE_FILENAME_LIST};${prefix}${ARG_OUTPUT_NAME}${suffix}" CACHE INTERNAL "")
    endif()

    install(TARGETS ${ARG_TARGET}
        LIBRARY DESTINATION ${install_dest}
        COMPONENT ${PYTHON_MODULE_NAME}
)
endfunction()

add_subdirectory(common)
add_subdirectory(mod_cvcuda)

# Set up debian control files
set(SRC_PYTHON_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/python)
set(DST_PYTHON_MODULE_PATH /usr/lib/python3/dist-packages)

# Add package files to the list before converting to string.
set(PYTHON_MODULE_FILENAME_LIST "${PYTHON_MODULE_FILENAME_LIST};cvcuda/__init__.py" CACHE INTERNAL "")
# Now convert to space-separated string for the postinst script
string(JOIN " " PYTHON_MODULE_FILENAME_LIST ${PYTHON_MODULE_FILENAME_LIST})

configure_file(cpack/debian_python_postinst.in cpack/postinst @ONLY)
configure_file(cpack/debian_python_prerm.in cpack/prerm @ONLY)
