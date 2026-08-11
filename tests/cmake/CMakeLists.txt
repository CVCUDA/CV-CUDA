# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(CVCUDA_CUDA_ARCHITECTURE_POLICY_TEST cvcuda_test_cuda_architecture_policy)
configure_file(
    test_cuda_architecture_policy.py
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CVCUDA_CUDA_ARCHITECTURE_POLICY_TEST}"
    COPYONLY
)

set(CVCUDA_CUDA_ARCHITECTURE_POLICY_TEST_DATA_DIR
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_DATADIR}/cvcuda/tests/cmake"
)
file(MAKE_DIRECTORY "${CVCUDA_CUDA_ARCHITECTURE_POLICY_TEST_DATA_DIR}")
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/CUDAArchitecturePolicy.cmake"
    "${CVCUDA_CUDA_ARCHITECTURE_POLICY_TEST_DATA_DIR}/CUDAArchitecturePolicy.cmake"
    COPYONLY
)

install(
    FILES "${CMAKE_SOURCE_DIR}/cmake/CUDAArchitecturePolicy.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cvcuda/tests/cmake"
    COMPONENT tests
)

nvcv_add_test(
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CVCUDA_CUDA_ARCHITECTURE_POLICY_TEST}"
    cmake
)
