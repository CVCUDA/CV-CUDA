/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_TESTS_DEVICE_ATOMICS_HPP
#define NVCV_TESTS_DEVICE_ATOMICS_HPP

#include <vector>

template<typename DataType>
DataType DeviceRunAtomicMin(const std::vector<DataType> &);

template<typename DataType>
DataType DeviceRunAtomicMax(const std::vector<DataType> &);

#endif // NVCV_TESTS_DEVICE_ATOMICS_HPP
