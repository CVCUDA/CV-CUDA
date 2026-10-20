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
find_program(CVCUDA_PYTHON_EXECUTABLE NAMES python${PYTHON_VERSION} REQUIRED)
set(Python_EXECUTABLE "${CVCUDA_PYTHON_EXECUTABLE}" CACHE FILEPATH
    "Exact CV-CUDA Python interpreter" FORCE)
find_package(Python ${CVCUDA_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import sys, sysconfig; print(f'{sys.version_info.major}.{sys.version_info.minor}' + ('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))"
                OUTPUT_VARIABLE CVCUDA_DETECTED_PYTHON OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
if(NOT CVCUDA_DETECTED_PYTHON STREQUAL PYTHON_VERSION)
    message(FATAL_ERROR "${Python_EXECUTABLE} is Python ${CVCUDA_DETECTED_PYTHON}, requested ${PYTHON_VERSION}")
endif()
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE nanobind_cmakedir_result
)
if(NOT nanobind_cmakedir_result EQUAL 0)
    message(FATAL_ERROR "Failed to find nanobind cmake dir for ${Python_EXECUTABLE}. "
                        "Is nanobind installed? Try: ${Python_EXECUTABLE} -m pip install nanobind")
endif()
find_package(nanobind 3.0.1 REQUIRED CONFIG)

find_package(nvcv_types REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/nvcv/util nvcv_util)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/cvcuda/util cvcuda_util)

set(PYTHON_MODULE_NAME python${PYTHON_VERSION})
string(REPLACE "." "" PYTHON_MODULE_NAME "${PYTHON_MODULE_NAME}")

include(GNUInstallDirs)
set(PYTHON_MODULE_FILENAME_LIST "" CACHE INTERNAL "")

function(cvcuda_python_add_module)
    cmake_parse_arguments(ARG "" "TARGET;OUTPUT_NAME;PACKAGE_DIR" "SOURCES" ${ARGV})

    # Operator submission is a host-side hot path. Keep the project's Release
    # optimization level instead of nanobind's default extension-wide -Os;
    # the latter saves some binary size but measurably slows every dispatch.
    nanobind_add_module(${ARG_TARGET} FREE_THREADED NOMINSIZE LTO ${ARG_SOURCES})

    foreach(nanobind_target nanobind-static nanobind-static-ft)
    if(TARGET ${nanobind_target} AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Third-party nanobind sources are built inside this warning-strict
        # project but do not follow CV-CUDA's declaration warning policy.
        target_compile_options(${nanobind_target} PRIVATE
            -Wno-missing-declarations
            -Wno-redundant-decls
        )
    endif()
    endforeach()

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

# Add package files/directories to the list before converting to string.
# The pure-Python cvcuda.tv sources are installed by the lib component; expose
# them in dist-packages alongside __init__.py and the extension module symlinks.
set(PYTHON_MODULE_FILENAME_LIST "${PYTHON_MODULE_FILENAME_LIST};cvcuda/__init__.py" CACHE INTERNAL "")
set(PYTHON_MODULE_FILENAME_LIST "${PYTHON_MODULE_FILENAME_LIST};cvcuda/py.typed" CACHE INTERNAL "")
set(PYTHON_MODULE_FILENAME_LIST "${PYTHON_MODULE_FILENAME_LIST};cvcuda/tv" CACHE INTERNAL "")
# Now convert to space-separated string for the postinst script
string(JOIN " " PYTHON_MODULE_FILENAME_LIST ${PYTHON_MODULE_FILENAME_LIST})

configure_file(cpack/debian_python_postinst.in cpack/postinst @ONLY)
configure_file(cpack/debian_python_prerm.in cpack/prerm @ONLY)
