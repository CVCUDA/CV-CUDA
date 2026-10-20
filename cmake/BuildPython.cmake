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

include(ExternalProject)

# Where our python module installed, it'll end up being in the same
# directory nvcv shared library resides
set(PYPROJ_COMMON_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
                       -Dnvcv_types_ROOT=${CMAKE_CURRENT_BINARY_DIR}/cmake)

if(CMAKE_BUILD_TYPE)
    list(APPEND PYPROJ_COMMON_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

get_target_property(NVCV_TYPES_SOURCE_DIR nvcv_types SOURCE_DIR)

# Needed so that nvcv_types library's build path gets added
# as RPATH to the plugin module. When outer project gets installed,
# it shall overwrite the RPATH with the final installation path.
list(APPEND PYPROJ_COMMON_ARGS
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=true
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=true
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_MODULE_PATH=${CMAKE_CURRENT_BINARY_DIR}/cmake
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    -DNVCV_TYPES_SOURCE_DIR=${NVCV_TYPES_SOURCE_DIR}
    -DWARNINGS_AS_ERRORS=${WARNINGS_AS_ERRORS}
    -DENABLE_COMPAT_OLD_GLIBC=${ENABLE_COMPAT_OLD_GLIBC}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

# The sub-builds are separate CMake projects, so they do not inherit the ccache launcher from
# cmake/ConfigCCache.cmake; without it every Python extension source recompiled on every build.
foreach(lang C CXX)
    if(CMAKE_${lang}_COMPILER_LAUNCHER)
        list(APPEND PYPROJ_COMMON_ARGS -DCMAKE_${lang}_COMPILER_LAUNCHER=${CMAKE_${lang}_COMPILER_LAUNCHER})
    endif()
endforeach()

# It needs to overwrite the PYTHON_MODULE_EXTENSION to generate
# python module name with correct name when cross compiling
# example: set(PYTHON_MODULE_EXTENSION .cpython-py38-aarch64-linux-gnu.so)
if (CMAKE_CROSSCOMPILING)
    list(APPEND PYPROJ_COMMON_ARGS
        -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}
        -DPYTHON_MODULE_EXTENSION=${PYTHON_MODULE_EXTENSION}
    )
endif()

# The Python wheels build as concurrent ExternalProjects that each default to a
# full host-core Ninja, so their combined fan-out (versions x cores) can
# oversubscribe memory and stall or OOM large builds. Split the outer job budget
# (CVCUDA_BUILD_JOBS from build.sh; host cores otherwise) across them.
list(LENGTH PYTHON_VERSIONS _NUM_PY_VERSIONS)
if(DEFINED CVCUDA_BUILD_JOBS AND CVCUDA_BUILD_JOBS GREATER 0)
    set(_PY_TOTAL_JOBS ${CVCUDA_BUILD_JOBS})
else()
    include(ProcessorCount)
    ProcessorCount(_PY_TOTAL_JOBS)
    if(_PY_TOTAL_JOBS EQUAL 0)
        set(_PY_TOTAL_JOBS 1)
    endif()
endif()
if(_NUM_PY_VERSIONS GREATER 0)
    math(EXPR _PY_BUILD_JOBS "${_PY_TOTAL_JOBS} / ${_NUM_PY_VERSIONS}")
else()
    set(_PY_BUILD_JOBS ${_PY_TOTAL_JOBS})
endif()
if(_PY_BUILD_JOBS LESS 1)
    set(_PY_BUILD_JOBS 1)
endif()
message(STATUS "Python wheel sub-builds: ${_NUM_PY_VERSIONS} x -j${_PY_BUILD_JOBS} (budget ${_PY_TOTAL_JOBS})")

foreach(VER ${PYTHON_VERSIONS})
    set(BASEDIR ${CMAKE_CURRENT_BINARY_DIR}/python${VER})

    ExternalProject_Add(cvcuda_python${VER}
        PREFIX ${BASEDIR}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python
        CMAKE_ARGS ${PYPROJ_COMMON_ARGS} -DPYTHON_VERSION=${VER}
        BINARY_DIR ${BASEDIR}/build
        TMP_DIR ${BASEDIR}/tmp
        STAMP_DIR ${BASEDIR}/stamp
        BUILD_ALWAYS true
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${_PY_BUILD_JOBS}
        DEPENDS nvcv_types cvcuda
        INSTALL_COMMAND ""
    )
endforeach()

# nanobind.stubgen must run against a Python that can import the just-built
# cvcuda extension and has the pinned nanobind package installed. Prefer the
# first version in PYTHON_VERSIONS, and fall back to the generic Python3
# interpreter CMake discovered. Without this, CMake's `find_package(Python3)`
# may pick a newer system Python that has no nanobind package and
# breaks the `wheel` target.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(PYTHON_VERSIONS)
    list(GET PYTHON_VERSIONS 0 _STUBGEN_PY_VER)
    set(STUBGEN_PYTHON "python${_STUBGEN_PY_VER}")
    unset(_STUBGEN_PY_VER)
elseif(Python3_FOUND)
    set(STUBGEN_PYTHON ${Python3_EXECUTABLE})
endif()

if(STUBGEN_PYTHON)
    add_custom_target(generate_stubs
        COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_BINARY_DIR}/python3
                ${STUBGEN_PYTHON} -m nanobind.stubgen -m cvcuda --recursive
                --pattern-file ${CMAKE_SOURCE_DIR}/python/stubgen.patterns
                --output-dir ${CMAKE_BINARY_DIR}/python3/cvcuda
                --quiet
        COMMAND ${STUBGEN_PYTHON} ${CMAKE_SOURCE_DIR}/python/validate_stubs.py
                ${CMAKE_BINARY_DIR}/python3/cvcuda
        COMMENT "Generating Python type stubs (dev target)"
        VERBATIM
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python3/cvcuda)

    # Configure __init__.py for each package with the appropriate module name

    # cvcuda: all types and operators in single module
    set(PACKAGE_NAME "cvcuda")
    set(EXTRA_IMPORTS "
# Explicitly export private attributes not included in 'import *'.
from ._cvcuda import _C_API, _test  # noqa: F401")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py.in" "${CMAKE_BINARY_DIR}/python3/cvcuda/__init__.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/py.typed" "${CMAKE_BINARY_DIR}/python3/cvcuda/py.typed" COPYONLY)

    # Install __init__.py files for package structure in Debian packages
    # Install in lib component since they're shared across all Python versions
    install(FILES
            "${CMAKE_BINARY_DIR}/python3/cvcuda/__init__.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/py.typed"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/cvcuda
            COMPONENT lib)

    # cvcuda.tv: pure-Python torchvision-compatible subpackage. Its sources live
    # under python/cvcuda_tv/ (kept separate from the generated python3/cvcuda/
    # staging tree) and are shared across all Python versions, so install them in
    # the lib component mirroring __init__.py above. The trailing slash installs
    # the directory contents into .../cvcuda/tv (not .../cvcuda/tv/cvcuda_tv).
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/cvcuda_tv/"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/cvcuda/tv
            COMPONENT lib
            FILES_MATCHING PATTERN "*.py")

    # BUILD_PYTHON must leave an importable build-tree package even when wheel
    # packaging is disabled. Benchmarks deliberately prefer this package over
    # ambient installations so they exercise the bindings from the current build.
    add_custom_target(stage_python_build_tree ALL
        COMMAND sh -c "cp \"${CMAKE_BINARY_DIR}\"/lib/python/_cvcuda*.so \"${CMAKE_BINARY_DIR}\"/python3/cvcuda/"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/python/cvcuda_tv
                ${CMAKE_BINARY_DIR}/python3/cvcuda/tv
        COMMENT "Staging Python package for build-tree use"
        VERBATIM
    )

    foreach(VER ${PYTHON_VERSIONS})
        add_dependencies(stage_python_build_tree cvcuda_python${VER})
    endforeach()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND BUILD_PYTHON_WHEEL)
    if(NOT DEFINED PYTHON_WHEEL_VERSION_SUFFIX)
        set(PYTHON_WHEEL_VERSION_SUFFIX "${PROJECT_VERSION_SUFFIX}")
    endif()

    set(PACKAGE_LIB_DIR ${CMAKE_BINARY_DIR}/python3/lib)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python3/lib)

    # Configure Python packaging files
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py.in" "${CMAKE_BINARY_DIR}/python3/setup.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml.in" "${CMAKE_BINARY_DIR}/python3/pyproject.toml")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/README.md.in" "${CMAKE_BINARY_DIR}/python3/README.md")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/MANIFEST.in" "${CMAKE_BINARY_DIR}/python3/MANIFEST.in")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" "${CMAKE_BINARY_DIR}/python3/LICENSE.md" COPYONLY)

    add_custom_target(wheel ALL)
    add_dependencies(wheel stage_python_build_tree)

    add_custom_command(
        TARGET wheel
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cvcuda> ${CMAKE_BINARY_DIR}/python3/lib
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nvcv_types> ${CMAKE_BINARY_DIR}/python3/lib
    )

    # Ensure numpy is importable by the stubgen Python before nanobind.stubgen
    # runs: stubgen imports cvcuda, and the extension imports NumPy while
    # registering its dtype compatibility layer. The manylinux builder ships without
    # numpy for cp3{10..14}, so without this the wheel target fails with
    # "ModuleNotFoundError: No module named 'numpy'".  On images that already
    # have numpy (Ubuntu devel, Jetson edge venv) this is a fast no-op.
    # TODO: drop this COMMAND once the manylinux builder image is rebuilt with
    # numpy preinstalled (see docker/Dockerfile.builder.deps).
    add_custom_command(
        TARGET wheel
        COMMAND sh -c "${STUBGEN_PYTHON} -c 'import numpy' >/dev/null 2>&1 || ${STUBGEN_PYTHON} -m pip install --quiet --disable-pip-version-check -r ${CMAKE_SOURCE_DIR}/tests/requirements.tests.numpy2.txt"
        COMMENT "Ensuring numpy is available for nanobind.stubgen"
        VERBATIM
    )

    add_custom_command(
        TARGET wheel
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python3
                ${STUBGEN_PYTHON} -m nanobind.stubgen -m cvcuda --recursive
                --pattern-file ${CMAKE_SOURCE_DIR}/python/stubgen.patterns
                --output-dir ${CMAKE_BINARY_DIR}/python3/cvcuda
                --quiet
        COMMAND ${STUBGEN_PYTHON} ${CMAKE_SOURCE_DIR}/python/validate_stubs.py
                ${CMAKE_BINARY_DIR}/python3/cvcuda
        COMMENT "Generating Python type stubs for cvcuda"
        VERBATIM
    )

    add_custom_command(
        TARGET wheel
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/python/build_wheels.sh" "${CMAKE_BINARY_DIR}/python3"
    )
endif()
