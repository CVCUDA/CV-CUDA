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

project(cvcuda_samples)

# Install main sample script
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/common.py
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples)

# Install runner scripts
install(PROGRAMS
        ${CMAKE_CURRENT_SOURCE_DIR}/run_samples.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/install_samples_dependencies.sh
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples)

# Install sample applications
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/applications
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples
        FILES_MATCHING PATTERN "*.py")

# Install operator examples
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/operators
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples
        FILES_MATCHING PATTERN "*.py")

# Install datatype examples
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/datatypes
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples
        FILES_MATCHING PATTERN "*.py")

# Install interoperability examples
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/interoperability
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples
        FILES_MATCHING PATTERN "*.py")

# Install sample assets
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples)

# Install object cache support
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/object_cache
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples
        FILES_MATCHING PATTERN "*.py")

# Install requirements files
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.samples.hello_world_cu12.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.samples.hello_world_cu13.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.samples.common.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.samples.cu12.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.samples.cu13.txt
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples)

# Install documentation
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.md
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT samples)
