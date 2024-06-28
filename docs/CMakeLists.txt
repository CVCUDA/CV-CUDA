# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_program(SPHINX_EXECUTABLE NAMES sphinx-build DOC "Path to sphinx-build executable")

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
list(GET PYTHON_VERSIONS -1 VER)

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                    COMMENT "Generating doxygen xml"
	   	    DEPENDS cvcuda_python${VER})

add_custom_target(cvcuda_doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
set(C_CPP_API_RST ${SPHINX_SOURCE}/_c_cpp_api)
set(PY_CVCUDA_API_RST ${SPHINX_SOURCE}/_python_api/_cvcuda_api)

# Start from clean directory for rst files, otherwise build could be affected due to old files
file(REMOVE_RECURSE ${C_CPP_API_RST}/*)
file(REMOVE_RECURSE ${PY_CVCUDA_API_RST}/*)

# Generate rst files for groups from doxygen index.xml
add_custom_target(cvcuda_groups ALL python3 ${SPHINX_SOURCE}/generate_groups.py ${C_CPP_API_RST} ${DOXYGEN_OUTPUT_DIR}/xml
		            DEPENDS cvcuda_doxygen)

# Generate rst files for python documentation
add_custom_target(cvcuda_python_docs ALL python3 ${SPHINX_SOURCE}/gen_py_doc_rsts.py ${PY_CVCUDA_API_RST} ${CMAKE_SOURCE_DIR}
                        DEPENDS cvcuda_python${VER})

# Generate Sphinx
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                    COMMAND ${CMAKE_COMMAND} -E env "SPHINX_PYTHON_SRC=${CMAKE_BINARY_DIR}/lib/python"
                            ${SPHINX_EXECUTABLE} -j auto -b html
                            # Tell Breathe where to find the Doxygen's xml output. Needed to have c/cpp documentation.
                            -Dbreathe_projects.cvcuda=${DOXYGEN_OUTPUT_DIR}/xml
                            ${SPHINX_SOURCE} ${SPHINX_BUILD}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    DEPENDS ${SPHINX_SOURCE}/index.rst
                            cvcuda_doxygen
		            cvcuda_groups
                    MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                    COMMENT "Generating documentation with Sphinx")

add_custom_target(cvcuda_sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

install(DIRECTORY ${SPHINX_BUILD} DESTINATION docs)
