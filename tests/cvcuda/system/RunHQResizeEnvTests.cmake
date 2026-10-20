# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REFERENCE_FILE)
    message(FATAL_ERROR "TEST_EXECUTABLE and REFERENCE_FILE are required")
endif()

file(REMOVE "${REFERENCE_FILE}")
foreach(LANES 8 16 32)
    set(LANE_MARKER_FILE "${REFERENCE_FILE}.${LANES}.ran")
    file(REMOVE "${LANE_MARKER_FILE}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            NVCV_LEAK_DETECTION=abort
            CVCUDA_HQ_RESIZE_BLOCK_LANES=${LANES}
            CVCUDA_TEST_EXPECTED_BLOCK_LANES=${LANES}
            CVCUDA_TEST_BLOCK_LANES_REFERENCE_FILE=${REFERENCE_FILE}
            CVCUDA_TEST_BLOCK_LANES_MARKER_FILE=${LANE_MARKER_FILE}
            "${TEST_EXECUTABLE}"
            --gtest_filter=OpHQResize*
        RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        file(REMOVE "${REFERENCE_FILE}")
        message(FATAL_ERROR "HQResize ${LANES}-lane parity run failed with status ${RESULT}")
    endif()
    if(LANES EQUAL 8 AND NOT EXISTS "${REFERENCE_FILE}")
        message(FATAL_ERROR "HQResize 8-lane parity run did not produce its reference")
    endif()
    if(NOT EXISTS "${LANE_MARKER_FILE}")
        message(FATAL_ERROR "HQResize ${LANES}-lane parity test did not produce its liveness marker")
    endif()
    file(REMOVE "${LANE_MARKER_FILE}")
endforeach()

set(INVALID_LANE_MARKER_FILE "${REFERENCE_FILE}.invalid")
file(REMOVE "${INVALID_LANE_MARKER_FILE}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        NVCV_LEAK_DETECTION=abort
        CVCUDA_HQ_RESIZE_BLOCK_LANES=0
        CVCUDA_TEST_INVALID_LANE_MARKER_FILE=${INVALID_LANE_MARKER_FILE}
        "${TEST_EXECUTABLE}"
        --gtest_filter=OpHQResizeNegative.invalid_block_lane_override_returns_invalid_argument
    RESULT_VARIABLE RESULT)
file(REMOVE "${REFERENCE_FILE}")
if(NOT RESULT EQUAL 0)
    file(REMOVE "${INVALID_LANE_MARKER_FILE}")
    message(FATAL_ERROR "HQResize invalid-lane run failed with status ${RESULT}")
endif()
if(NOT EXISTS "${INVALID_LANE_MARKER_FILE}")
    message(FATAL_ERROR "HQResize invalid-lane test did not produce its liveness marker")
endif()
file(REMOVE "${INVALID_LANE_MARKER_FILE}")
