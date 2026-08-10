# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(cvcuda_benchmarks)

# Build C++ benchmarks
add_subdirectory(cpp)

# Install Python benchmarks
add_subdirectory(python)

# Copy benchmark config files to build directory
# Use custom command to ensure config files are updated when source changes
# Note: load_config.py is imported by python_bench_utils.py and run_bench.py
set(CONFIG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/config")
set(CONFIG_DEST_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config")
set(CONFIG_JSON_SOURCE "${CONFIG_SOURCE_DIR}/bench_params.json")
set(CONFIG_JSON_DEST "${CONFIG_DEST_DIR}/bench_params.json")
set(CONFIG_SKU_MAP_SOURCE "${CONFIG_SOURCE_DIR}/sku_map.json")
set(CONFIG_SKU_MAP_DEST "${CONFIG_DEST_DIR}/sku_map.json")
set(CONFIG_LOADER_SOURCE "${CONFIG_SOURCE_DIR}/load_config.py")
set(CONFIG_LOADER_DEST "${CONFIG_DEST_DIR}/load_config.py")
set(CONFIG_AXIS_ORDER_SOURCE "${CONFIG_SOURCE_DIR}/axis_order.py")
set(CONFIG_AXIS_ORDER_DEST "${CONFIG_DEST_DIR}/axis_order.py")
file(GLOB CONFIG_OPERATOR_SOURCES CONFIGURE_DEPENDS "${CONFIG_SOURCE_DIR}/operators/*.json")

file(MAKE_DIRECTORY "${CONFIG_DEST_DIR}")
file(MAKE_DIRECTORY "${CONFIG_DEST_DIR}/operators")

# Add custom commands to copy config files when they change
add_custom_command(
    OUTPUT "${CONFIG_JSON_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CONFIG_JSON_SOURCE}" "${CONFIG_JSON_DEST}"
    DEPENDS "${CONFIG_JSON_SOURCE}"
    COMMENT "Copying bench_params.json to build directory"
    VERBATIM
)

add_custom_command(
    OUTPUT "${CONFIG_LOADER_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CONFIG_LOADER_SOURCE}" "${CONFIG_LOADER_DEST}"
    DEPENDS "${CONFIG_LOADER_SOURCE}"
    COMMENT "Copying load_config.py to build directory"
    VERBATIM
)

add_custom_command(
    OUTPUT "${CONFIG_AXIS_ORDER_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CONFIG_AXIS_ORDER_SOURCE}" "${CONFIG_AXIS_ORDER_DEST}"
    DEPENDS "${CONFIG_AXIS_ORDER_SOURCE}"
    COMMENT "Copying canonical benchmark axis order to build directory"
    VERBATIM
)

add_custom_command(
    OUTPUT "${CONFIG_SKU_MAP_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CONFIG_SKU_MAP_SOURCE}" "${CONFIG_SKU_MAP_DEST}"
    DEPENDS "${CONFIG_SKU_MAP_SOURCE}"
    COMMENT "Copying bench sku_map.json to build directory"
    VERBATIM
)

set(CONFIG_OPERATOR_DESTS "")
foreach(CONFIG_OPERATOR_SOURCE IN LISTS CONFIG_OPERATOR_SOURCES)
    get_filename_component(CONFIG_OPERATOR_FILE "${CONFIG_OPERATOR_SOURCE}" NAME)
    set(CONFIG_OPERATOR_DEST "${CONFIG_DEST_DIR}/operators/${CONFIG_OPERATOR_FILE}")
    add_custom_command(
        OUTPUT "${CONFIG_OPERATOR_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CONFIG_OPERATOR_SOURCE}" "${CONFIG_OPERATOR_DEST}"
        DEPENDS "${CONFIG_OPERATOR_SOURCE}"
        COMMENT "Copying ${CONFIG_OPERATOR_FILE} benchmark config to build directory"
        VERBATIM
    )
    list(APPEND CONFIG_OPERATOR_DESTS "${CONFIG_OPERATOR_DEST}")
endforeach()

# Create a custom target that depends on the copied files
add_custom_target(copy_bench_config ALL
    DEPENDS "${CONFIG_JSON_DEST}" "${CONFIG_LOADER_DEST}" "${CONFIG_AXIS_ORDER_DEST}"
            "${CONFIG_SKU_MAP_DEST}" ${CONFIG_OPERATOR_DESTS}
)

# Install benchmark config files
install(FILES
        config/bench_params.json
        config/sku_map.json
        config/load_config.py
        config/axis_order.py
        DESTINATION ${CMAKE_INSTALL_BINDIR}/config
        COMPONENT bench)

install(DIRECTORY config/operators
        DESTINATION ${CMAKE_INSTALL_BINDIR}/config
        COMPONENT bench
        FILES_MATCHING PATTERN "*.json")
