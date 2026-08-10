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

# Setup Doxygen input/output
find_package(Doxygen)

if(DOXYGEN_DOT_FOUND)
    set(DOXYGEN_HAVE_DOT YES)
else()
    message(STATUS "Graphviz dot not found — disabling Doxygen graph generation")
    set(DOXYGEN_HAVE_DOT NO)
endif()

get_target_property(CORE_SRCPATH nvcv_types SOURCE_DIR)
get_target_property(OPS_SRCPATH cvcuda SOURCE_DIR)

set(DOXYGEN_INPUT_DIR "${CORE_SRCPATH}/include ${OPS_SRCPATH}/include")
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(DOXYGEN_IMAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/assets)

# Replace variables inside @@ with the current value
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Only regenerate Doxygen when the Doxyfile changes

# Build docs once python libraries are generated
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# Use explicitly provided DOC_PYTHON_VERSION if set, otherwise auto-detect system Python
if(DOC_PYTHON_VERSION)
    set(PYTHON_VER ${DOC_PYTHON_VERSION})
else()
    # Auto-detect system's default Python version
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(PYTHON_VER "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    message(STATUS "DOC_PYTHON_VERSION not set, auto-detected system Python: ${PYTHON_VER}")
endif()

# Convert version to cpXXX format for manylinux search (e.g., 3.10 -> cp310-cp310)
string(REPLACE "." "" PYTHON_VER_NODOT ${PYTHON_VER})
set(MANYLINUX_PYTHON_PATH "/opt/python/cp${PYTHON_VER_NODOT}-cp${PYTHON_VER_NODOT}")

# If we're in a manylinux environment, prioritize the manylinux Python installation
if(EXISTS ${MANYLINUX_PYTHON_PATH})
    message(STATUS "Found manylinux Python path: ${MANYLINUX_PYTHON_PATH}")
    set(Python_ROOT_DIR ${MANYLINUX_PYTHON_PATH})
endif()

# Find Python using the specified version
find_package(Python ${PYTHON_VER} EXACT COMPONENTS Interpreter REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                    COMMENT "Generating doxygen xml"
                    DEPENDS cvcuda_python${PYTHON_VER})

add_custom_target(cvcuda_doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
set(C_CPP_API_RST ${SPHINX_SOURCE}/_c_cpp_api)

# Start from clean directory for rst files, otherwise build could be affected due to old files
if(EXISTS ${C_CPP_API_RST})
    file(REMOVE_RECURSE ${C_CPP_API_RST})
endif()

# Generate rst files for groups from doxygen index.xml
add_custom_target(cvcuda_groups ALL ${PYTHON_EXECUTABLE} ${SPHINX_SOURCE}/generate_groups.py ${C_CPP_API_RST} ${DOXYGEN_OUTPUT_DIR}/xml
                    DEPENDS cvcuda_doxygen)

# Create virtual environment and install wheel for documentation
# This isolates the installation and leaves system Python untouched
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND BUILD_PYTHON)
    set(DOC_VENV_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc_venv)
    set(DOC_VENV_PYTHON ${DOC_VENV_DIR}/bin/python)
    set(WHEEL_INSTALL_MARKER ${CMAKE_CURRENT_BINARY_DIR}/wheel_installed.stamp)
    set(INSTALL_WHEEL_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/install_wheel.cmake)

    # Generate CMake script to find and install wheel at build time
    file(WRITE ${INSTALL_WHEEL_SCRIPT} "
# Script to find and install the cvcuda wheel at build time
file(GLOB WHEEL_FILES \"\${WHEEL_DIR}/*.whl\")

if(NOT WHEEL_FILES)
    message(FATAL_ERROR \"No wheel files found in \${WHEEL_DIR}\")
endif()

list(GET WHEEL_FILES 0 WHEEL_FILE)
message(STATUS \"Installing wheel: \${WHEEL_FILE}\")

execute_process(
    COMMAND \"\${VENV_PYTHON}\" -m pip install --force-reinstall --no-deps \"\${WHEEL_FILE}\"
    RESULT_VARIABLE INSTALL_RESULT
)

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR \"Failed to install wheel: \${WHEEL_FILE}\")
endif()

# Create marker file to indicate successful installation
file(TOUCH \"\${MARKER_FILE}\")
")

    # Create venv and install wheel
    add_custom_command(
        OUTPUT ${WHEEL_INSTALL_MARKER}
        # Create virtual environment with system site packages
        COMMAND ${PYTHON_EXECUTABLE} -m venv --system-site-packages ${DOC_VENV_DIR}
        # Install documentation dependencies (Sphinx, breathe, etc.)
        # this will skip packages installed via the system-site-packages flag
        # in existing CI setup this will do nothing, on local it may install packages if venv creator
        # does not properly support system-site-packages flag
        COMMAND ${DOC_VENV_PYTHON} -m pip install -r ${CMAKE_SOURCE_DIR}/docs/requirements.docs.txt
        # Find the wheel file dynamically and install it using CMake script
        COMMAND ${CMAKE_COMMAND}
                -DWHEEL_DIR=${CMAKE_BINARY_DIR}/python3/repaired_wheels
                -DVENV_PYTHON=${DOC_VENV_PYTHON}
                -DMARKER_FILE=${WHEEL_INSTALL_MARKER}
                -P ${INSTALL_WHEEL_SCRIPT}
        DEPENDS wheel
        COMMENT "Creating venv and installing cvcuda wheel for documentation (requires: -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=1)"
    )

    add_custom_target(install_wheel_for_docs DEPENDS ${WHEEL_INSTALL_MARKER})

    # Use venv Python for Sphinx
    set(SPHINX_PYTHON ${DOC_VENV_PYTHON})
else()
    message(FATAL_ERROR "Documentation build requires Release build type and Python bindings to generate wheels. Use: cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=1")
endif()

# Generate Sphinx documentation using venv Python with installed wheel
# Ensure CUDA libraries are in LD_LIBRARY_PATH so the Python bindings can load
# The module needs to initialize CUDA to make docstrings available, but Sphinx
# only imports the module and doesn't execute GPU operations

# Find CUDA library path for documentation build
find_package(CUDAToolkit REQUIRED)
get_filename_component(CUDA_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}" ABSOLUTE)

# NOTE: use `-j 1` (serial) rather than `-j auto`.  With `-j auto`, each
# worker builds its own C++ domain symbol tree and the main process then
# merges them back, which is O(n^2) in the symbol count.  CV-CUDA exposes a
# large C++ API (nvcv::, cvcuda::, enums, macros), and this merge step takes
# longer than the whole serial build would.  Single-process is faster here
# and avoids that wart.
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                    COMMAND ${CMAKE_COMMAND} -E env
                            "LD_LIBRARY_PATH=${CUDA_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
                            ${SPHINX_PYTHON} -m sphinx -j 1 -b html
                            # Tell Breathe where to find the Doxygen's xml output. Needed to have c/cpp documentation.
                            -Dbreathe_projects.cvcuda=${DOXYGEN_OUTPUT_DIR}/xml
                            ${SPHINX_SOURCE} ${SPHINX_BUILD}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    DEPENDS ${SPHINX_SOURCE}/index.rst
                            ${DOXYGEN_INDEX_FILE}
                            cvcuda_doxygen
                            cvcuda_groups
                            install_wheel_for_docs
                    MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                    COMMENT "Building Sphinx HTML documentation")

add_custom_target(cvcuda_sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

# Automatically clean up venv after documentation is built
add_custom_command(
    TARGET cvcuda_sphinx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${DOC_VENV_DIR}
    COMMENT "Cleaning up documentation virtual environment"
)

# Also clean venv with standard clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${DOC_VENV_DIR}
    ${WHEEL_INSTALL_MARKER}
)

# Install only the Sphinx HTML documentation
# Note: Doxygen output is not installed separately because:
#   - Sphinx's Breathe extension consumes Doxygen XML during build
#   - The C/C++ API documentation is already integrated into Sphinx HTML output
#   - Doxygen HTML is redundant and not needed in the distribution
# Exclude CMake build artifacts, temporary files, and build configuration
install(DIRECTORY ${SPHINX_BUILD}/
    DESTINATION docs
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "*.cmake" EXCLUDE
    PATTERN "*.stamp" EXCLUDE
    PATTERN ".doctrees" EXCLUDE
    PATTERN ".buildinfo" EXCLUDE
)
