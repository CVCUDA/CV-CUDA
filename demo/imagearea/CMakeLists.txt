# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(imagearea_demo main.cpp)

target_link_libraries(imagearea_demo
    PRIVATE
        cvcuda
)
