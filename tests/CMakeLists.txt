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

project(cvcuda_tests)
set(CMAKE_FOLDER tests)

enable_testing()

add_library(nvcv_test_main Main.cpp)
target_link_libraries(nvcv_test_main
    PUBLIC
        GTest::gtest
        GTest::gmock
        nvcv_util_compat
    PRIVATE
        nvcv_types
        CUDA::cudart_static
)

# Now we want to create a script in the output directory that will run
# all tests we define.
if(UNIX)
    file(TO_NATIVE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/" TESTS_DRIVER_DIR)
    set(TESTS_DRIVER "${TESTS_DRIVER_DIR}/run_tests.sh")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh.in ${TESTS_DRIVER}
        @ONLY)

    macro(nvcv_add_test TESTCMD)
        get_filename_component(TESTNAME "${TESTCMD}" NAME)

        add_test(NAME "${TESTNAME}" COMMAND "${TESTCMD}")

        if(TARGET ${TESTNAME})
            target_link_libraries(${TESTNAME} PRIVATE nvcv_test_main)
        endif()

        # Timeouts are in seconds
        set(TIMEOUT_KILL 800)
        set(TIMEOUT_TERM 620)

        set_tests_properties(${TESTNAME} PROPERTIES TIMEOUT ${TIMEOUT_TERM})

        file(APPEND "${TESTS_DRIVER}" "run ${TESTNAME}\n")

        if(TARGET ${TESTNAME})
            install(TARGETS ${TESTNAME}
                    DESTINATION ${CMAKE_INSTALL_BINDIR}
                    COMPONENT tests)
        else()
            install(PROGRAMS "${TESTCMD}"
                    TYPE BIN
                    COMPONENT tests)
        endif()
    endmacro()

    install(PROGRAMS ${TESTS_DRIVER}
        TYPE BIN
        COMPONENT tests)
else()
    macro(nvcv_add_test)
        add_test(${ARGV})
    endmacro()
endif()

# Create build tree

add_subdirectory(common)
add_subdirectory(nvcv_types)
add_subdirectory(cvcuda)
