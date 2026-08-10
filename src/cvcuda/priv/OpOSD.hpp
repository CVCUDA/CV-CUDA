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
 * @file OpOSD.hpp
 *
 * @brief Defines the private C++ Class for the OSD operation.
 */

#ifndef CVCUDA_PRIV__O_S_D_HPP
#define CVCUDA_PRIV__O_S_D_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"
#include "legacy/CvCudaLegacy.h"

#include <cuda_runtime.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <array>
#include <memory>
#include <mutex>

namespace cvcuda::priv {

class Reformat;

struct OSDPlanarBridgeTensors
{
    nvcv::TensorDataStridedCuda input;
    nvcv::TensorDataStridedCuda output;
};

class OSDPlanarBridgeWorkspace
{
public:
    OSDPlanarBridgeWorkspace();
    ~OSDPlanarBridgeWorkspace();

    std::unique_lock<std::mutex> lock();

    OSDPlanarBridgeTensors prepare(cudaStream_t stream, const nvcv::Tensor &input,
                                   const nvcv::TensorDataStridedCuda &inputData,
                                   const nvcv::TensorDataStridedCuda &outputData, Reformat &reformat);

    void finish(cudaStream_t stream, const nvcv::Tensor &output, Reformat &reformat);
    void markPending(cudaStream_t stream);

private:
    static constexpr int kInputIndex        = 0;
    static constexpr int kOutputIndex       = 1;
    static constexpr int kBridgeTensorCount = 2;

    void synchronizeIfNeeded(cudaStream_t stream, const std::array<nvcv::TensorShape, kBridgeTensorCount> &targetShapes,
                             nvcv::DataType dtype);

    void resizeBuffer(int which, const nvcv::TensorShape &shape, nvcv::DataType dtype);

    std::mutex                                   m_mutex;
    std::array<nvcv::Tensor, kBridgeTensorCount> m_interleaved;
    cudaEvent_t                                  m_ready   = nullptr;
    bool                                         m_pending = false;
};

class OSD final : public IOperator
{
public:
    explicit OSD();
    ~OSD() override;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVElements &elements) const;

private:
    static std::unique_ptr<OSDPlanarBridgeWorkspace> CreatePlanarBridgeWorkspace(int deviceId);

    std::unique_ptr<nvcv::legacy::cuda_op::OSD>         m_legacyOp;
    std::unique_ptr<Reformat>                           m_reformatOp;
    mutable PerDeviceResource<OSDPlanarBridgeWorkspace> m_planarWorkspace{CreatePlanarBridgeWorkspace};
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__O_S_D_HPP
