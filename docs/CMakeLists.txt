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

# Replace variables inside @@ with the current value
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Only regenerate Doxygen when the Doxyfile change
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating doxygen xml")

add_custom_target(cvcuda_doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(EXHALE_SOURCE ${SPHINX_SOURCE}/_exhale_api)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
set(SPHINX_GROUP_INDEX_FILE ${SPHINX_BUILD}/groupindex.html)

# Generate rst files for groups from doxygen index.xml
add_custom_target(cvcuda_groups ALL
	COMMAND python3 ${SPHINX_SOURCE}/generate_groups.py ${EXHALE_SOURCE} ${DOXYGEN_OUTPUT_DIR}/xml
		  DEPENDS ${DOXYGEN_INDEX_FILE})

# Generate Sphinx
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND
                   ${SPHINX_EXECUTABLE} -j auto -b html
                   # Tell Breathe where to find the Doxygen output
                   -Dbreathe_projects.cvcuda=${DOXYGEN_OUTPUT_DIR}/xml
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
		   ${SPHINX_SOURCE}/index.rst
                   ${DOXYGEN_INDEX_FILE}
                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                   COMMENT "Generating documentation with Sphinx")


add_custom_target(cvcuda_sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

install(DIRECTORY ${SPHINX_BUILD} DESTINATION docs)
