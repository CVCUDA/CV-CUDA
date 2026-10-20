/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TESTS_DEVICE_BORDER_WRAP_HPP
#define NVCV_TESTS_DEVICE_BORDER_WRAP_HPP

#include <cuda_runtime.h> // for cudaStream_t, etc.

template<class DstWrapper, class SrcWrapper, typename DimType>
void DeviceRunFillBorder(DstWrapper &dstWrap, SrcWrapper &srcWrap, DimType dstSize, int2 borderSize,
                         cudaStream_t &stream);

// Compare value() against operator[] on the device. The host tests cannot reach this: the two
// forms are selected by __CUDA_ARCH__, so on the host only one of them is ever compiled and the
// arch-specific branch that ships is never executed. Writes 0 to \p mismatches for agreement at
// every probed coordinate, or the 1-based index of the first disagreement.
template<class BorderWrapper>
void DeviceRunValueMatchesSubscript(BorderWrapper &wrap, const int *coords, int numCoords, int *mismatches,
                                    cudaStream_t &stream);

// Run both constant-border read forms on the device and compare them, for an in-bounds pointer and
// for a null one. DeviceRunValueMatchesSubscript can only ever exercise the arm the running GPU's
// __CUDA_ARCH__ selects; this runs both on every GPU. Writes 0 to \p mismatches for agreement, 1
// if the in-bounds probes disagree, 2 if the out-of-bounds ones do.
template<typename T>
void DeviceRunConstantBorderFormsAgree(const T *inside, const T *outside, T border, int *mismatches,
                                       cudaStream_t &stream);

// value()/operator[] parity driven with a 4-D coordinate. Operators read border wraps with int3
// and int4, never int1, and doGetPtrForCoord resolves N == 4 through its own arm -- the arm
// InterpolationWrap::doGetValue reaches on a 4-D wrap.
template<class BorderWrapper>
void DeviceRunValueMatchesSubscript4D(BorderWrapper &wrap, const int4 *coords, int numCoords, int *mismatches,
                                      cudaStream_t &stream);

#endif // NVCV_TESTS_DEVICE_BORDER_WRAP_HPP
