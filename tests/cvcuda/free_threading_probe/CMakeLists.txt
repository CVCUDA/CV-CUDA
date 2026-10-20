# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET CUDA::cupti)
    message(FATAL_ERROR
            "Python tests require CUPTI development headers and libraries from the CUDA Toolkit")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/ConfigDLPack.cmake)

add_library(cvcuda_free_threading_probe MODULE FreeThreadingProbe.cpp)

target_link_libraries(cvcuda_free_threading_probe PRIVATE dlpack::dlpack nvcv_types CUDA::cudart CUDA::cupti ${CMAKE_DL_LIBS}
                                                            Threads::Threads)

set_target_properties(cvcuda_free_threading_probe PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Install next to the Python tests so packaged test runs use the probe from the
# installed test component rather than retaining a build-tree path.
install(TARGETS cvcuda_free_threading_probe
        LIBRARY DESTINATION "${CMAKE_INSTALL_DATADIR}/python/cvcuda"
        COMPONENT tests)
