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

# Enum classes exposed under `cvcuda.`.  pybind11_stubgen emits warnings and
# falls back to ugly default-value reprs like `<Border.CONSTANT: 0>` unless
# it knows where each enum class lives.  Keep this list in sync with enums
# registered via py::enum_ under the cvcuda module.
set(CVCUDA_STUBGEN_ENUM_LOCATIONS
    --enum-class-locations Border:cvcuda.Border
    --enum-class-locations Interp:cvcuda.Interp
    --enum-class-locations ThresholdType:cvcuda.ThresholdType
    --enum-class-locations AdaptiveThresholdType:cvcuda.AdaptiveThresholdType
    --enum-class-locations Remap:cvcuda.Remap
    --enum-class-locations ChannelManip:cvcuda.ChannelManip
    --enum-class-locations LabelMaskType:cvcuda.LabelMaskType
    --enum-class-locations ThreadScope:cvcuda.ThreadScope
    --enum-class-locations LABEL:cvcuda.LABEL
    --enum-class-locations SIFT:cvcuda.SIFT
    --enum-class-locations Matcher:cvcuda.Matcher
    --enum-class-locations ConnectivityType:cvcuda.ConnectivityType
)

# pybind11_stubgen must run against a Python that (a) can import the just-built
# cvcuda extension and (b) has pybind11-stubgen installed.  Prefer the first
# version in PYTHON_VERSIONS (always paired with a pybind11-stubgen install in
# our Docker images), and fall back to the generic Python3 interpreter CMake
# discovered.  Without this, CMake's `find_package(Python3)` may pick a newer
# system Python (e.g. /usr/bin/python3.12) that has no pybind11-stubgen and
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
                PYTHONPATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python
                ${STUBGEN_PYTHON} -m pybind11_stubgen cvcuda
                ${CVCUDA_STUBGEN_ENUM_LOCATIONS}
                --output-dir ${CMAKE_BINARY_DIR}/python3
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

    # Install __init__.py files for package structure in Debian packages
    # Install in lib component since they're shared across all Python versions
    install(FILES "${CMAKE_BINARY_DIR}/python3/cvcuda/__init__.py"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/cvcuda
            COMPONENT lib)

endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND BUILD_PYTHON_WHEEL)
    set(PACKAGE_LIB_DIR ${CMAKE_BINARY_DIR}/python3/lib)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python3/lib)

    # Configure Python packaging files
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py.in" "${CMAKE_BINARY_DIR}/python3/setup.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml.in" "${CMAKE_BINARY_DIR}/python3/pyproject.toml")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/README.md.in" "${CMAKE_BINARY_DIR}/python3/README.md")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/MANIFEST.in" "${CMAKE_BINARY_DIR}/python3/MANIFEST.in")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/py.typed" "${CMAKE_BINARY_DIR}/python3/cvcuda/py.typed" COPYONLY)

    add_custom_target(wheel ALL)

    foreach(VER ${PYTHON_VERSIONS})
        add_dependencies(wheel cvcuda_python${VER})
    endforeach()

    add_custom_command(
        TARGET wheel
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cvcuda> ${CMAKE_BINARY_DIR}/python3/lib
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nvcv_types> ${CMAKE_BINARY_DIR}/python3/lib
        COMMAND sh -c "cp ${CMAKE_BINARY_DIR}/lib/python/_cvcuda*.so ${CMAKE_BINARY_DIR}/python3/cvcuda/"
    )

    # Ensure numpy is importable by the stubgen Python before pybind11_stubgen
    # runs: stubgen imports cvcuda, and _cvcuda.so pulls numpy via pybind11's
    # npy_api at module init.  The manylinux :v11 builder image ships without
    # numpy for cp3{10..14}, so without this the wheel target fails with
    # "ModuleNotFoundError: No module named 'numpy'".  On images that already
    # have numpy (Ubuntu devel, Jetson edge venv) this is a fast no-op.
    # TODO: drop this COMMAND once the manylinux builder image is rebuilt with
    # numpy preinstalled (see docker/Dockerfile.builder.deps).
    add_custom_command(
        TARGET wheel
        COMMAND sh -c "${STUBGEN_PYTHON} -c 'import numpy' >/dev/null 2>&1 || ${STUBGEN_PYTHON} -m pip install --quiet --disable-pip-version-check -r ${CMAKE_SOURCE_DIR}/tests/requirements.tests.numpy2.txt"
        COMMENT "Ensuring numpy is available for pybind11_stubgen"
        VERBATIM
    )

    add_custom_command(
        TARGET wheel
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python3
                ${STUBGEN_PYTHON} -m pybind11_stubgen cvcuda
                ${CVCUDA_STUBGEN_ENUM_LOCATIONS}
                --output-dir ${CMAKE_BINARY_DIR}/python3
        COMMENT "Generating Python type stubs for cvcuda"
        VERBATIM
    )

    add_custom_command(
        TARGET wheel
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/python/build_wheels.sh" "${CMAKE_BINARY_DIR}/python3"
    )
endif()
