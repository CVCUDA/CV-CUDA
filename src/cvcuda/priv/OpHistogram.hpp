/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpHistogram.hpp
 *
 * @brief Defines the private C++ Class for the Histogram operation.
 */

#ifndef CVCUDA_PRIV__HISTOGRAM_HPP
#define CVCUDA_PRIV__HISTOGRAM_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"
#include "legacy/CvCudaLegacy.h"

#include <cuda_runtime.h>
#include <nvcv/Tensor.hpp>

#include <memory>
#include <mutex>

namespace cvcuda::priv {

class Reformat;

struct HistogramPlanarBridgeWorkspace
{
    std::mutex   mutex;
    nvcv::Tensor interleavedIn;
    nvcv::Tensor interleavedMask;
    cudaEvent_t  ready = nullptr;
    bool         busy  = false;

    HistogramPlanarBridgeWorkspace();
    ~HistogramPlanarBridgeWorkspace();

    void ensure(const nvcv::TensorShape &inShape, nvcv::DataType inDtype, const nvcv::TensorShape *maskShape,
                nvcv::DataType maskDtype, cudaStream_t stream);
    void record(cudaStream_t stream);
};

class Histogram final : public IOperator
{
public:
    explicit Histogram();
    ~Histogram() override;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, nvcv::OptionalTensorConstRef mask,
                    const nvcv::Tensor &histogram) const;

private:
    static std::unique_ptr<HistogramPlanarBridgeWorkspace> CreatePlanarBridgeWorkspace(int);

    std::unique_ptr<nvcv::legacy::cuda_op::Histogram>         m_legacyOp;
    std::unique_ptr<Reformat>                                 m_reformatOp;
    mutable PerDeviceResource<HistogramPlanarBridgeWorkspace> m_planarWorkspace{CreatePlanarBridgeWorkspace};
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__HISTOGRAM_HPP
