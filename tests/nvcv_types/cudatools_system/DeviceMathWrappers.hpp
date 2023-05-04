/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TESTS_DEVICE_MATH_WRAPPERS_HPP
#define NVCV_TESTS_DEVICE_MATH_WRAPPERS_HPP

#include <nvcv/cuda/MathWrappers.hpp> // the object of this test

namespace cuda = nvcv::cuda;

template<cuda::RoundMode RM, typename Type>
Type DeviceRunRoundSameType(Type);

template<cuda::RoundMode RM, typename TargetType, typename SourceType>
TargetType DeviceRunRoundDiffType(SourceType);

template<typename Type>
Type DeviceRunMin(Type, Type);

template<typename Type>
Type DeviceRunMax(Type, Type);

template<typename Type1, typename Type2>
Type1 DeviceRunPow(Type1, Type2);

template<typename Type>
Type DeviceRunExp(Type);

template<typename Type>
Type DeviceRunSqrt(Type);

template<typename Type>
Type DeviceRunAbs(Type);

template<typename Type1, typename Type2>
Type1 DeviceRunClamp(Type1, Type2, Type2);

#endif // NVCV_TESTS_DEVICE_MATH_WRAPPERS_HPP
