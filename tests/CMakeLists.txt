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

project(cvcuda_tests)
set(CMAKE_FOLDER tests)

# Tests require C++20:
# The goal for the infrastructure written for tests is to make it easy to
# - add test cases for new parameter ranges,
# - make the tested parameter set visually match what's defined in the reference doc
# - ...
# so that we can quickly check if everything we claim is being tested.
# In order to achieve this, we created "tests/common/ValueList.hpp" that implements a domain-specific embedded language making it easier to define the above.
# To make usage easier, we had to use C++20 language features.
set(CMAKE_CXX_STANDARD 20)

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
        -lrt
)

# Now we want to create a script in the output directory that will run
# all tests we define.
if(UNIX)
    file(TO_NATIVE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/" TESTS_DRIVER_DIR)
    set(TESTS_DRIVER "${TESTS_DRIVER_DIR}/run_tests.sh")
    set(WHEEL_TESTER "${TESTS_DRIVER_DIR}/test_wheels.sh")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh.in ${TESTS_DRIVER}
        @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_wheels.sh.in ${WHEEL_TESTER}
        @ONLY)

    macro(nvcv_add_test TESTCMD TESTGROUP)
        get_filename_component(TESTNAME "${TESTCMD}" NAME)

        add_test(NAME "${TESTNAME}" COMMAND "${TESTCMD}")

        # If an external target is being added to the tests, it should not be considered as a regular target,
        # avoiding linking it against libraries and using install targets with it
        if(TARGET ${TESTNAME})
            get_target_property(EXTERNAL_TARGET ${TESTNAME} IMPORTED)
        endif()

        if(TARGET ${TESTNAME} AND NOT EXTERNAL_TARGET)
            target_link_libraries(${TESTNAME} PRIVATE nvcv_test_main)
        endif()

        # Timeouts are in seconds
        set(TIMEOUT_KILL 800)
        set(TIMEOUT_TERM 620)

        set_tests_properties(${TESTNAME} PROPERTIES TIMEOUT ${TIMEOUT_TERM})

        file(APPEND "${TESTS_DRIVER}" "run ${TESTNAME} ${TESTGROUP}\n")

        if(TARGET ${TESTNAME} AND NOT EXTERNAL_TARGET)
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
    install(PROGRAMS ${WHEEL_TESTER}
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
