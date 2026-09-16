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

/**
 * @file OpThreshold.hpp
 *
 * @brief Defines the private C++ Class for the threshold operation.
 */

#ifndef CVCUDA_PRIV_THRESHOLD_HPP
#define CVCUDA_PRIV_THRESHOLD_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace cvcuda::priv {

class Threshold final : public IOperator
{
public:
    explicit Threshold(uint32_t type, int maxBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &thresh,
                    const nvcv::Tensor &maxval) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &thresh, const nvcv::Tensor &maxval) const;

private:
    // The automatic thresholding modes (OTSU, TRIANGLE) accumulate one 256-bin histogram per image
    // into persistent device scratch sized from maxBatchSize at construction, never per submit.
    // Nothing is allocated for the fixed-threshold modes.
    struct Histogram
    {
        Histogram(uint32_t automaticThresh, int maxBatchSize);
        ~Histogram();

        Histogram(const Histogram &)            = delete;
        Histogram &operator=(const Histogram &) = delete;

        int *data      = nullptr;
        bool allocated = false;
    };

    uint32_t m_type;
    uint32_t m_automaticThresh;
    uint32_t m_maskedType;
    int      m_maxBatchSize;

    // The tensor and var-shape paths keep separate scratch, matching the two legacy operator objects
    // they replace; PerDeviceResource gives each CUDA device its own allocation.
    mutable PerDeviceResource<Histogram> m_histogram;
    mutable PerDeviceResource<Histogram> m_histogramVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_THRESHOLD_HPP
