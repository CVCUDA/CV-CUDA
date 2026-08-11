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
 * @file OpSIFT.hpp
 *
 * @brief Defines the public C++ Class for the SIFT operation.
 * @defgroup NVCV_CPP_ALGORITHM_SIFT SIFT
 * @{
 */

#ifndef CVCUDA_SIFT_HPP
#define CVCUDA_SIFT_HPP

#include "IOperator.hpp"
#include "OpSIFT.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class SIFT final : public IOperator
{
public:
    explicit SIFT(int3 maxShape, int maxOctaveLayers);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &featCoords,
                    const nvcv::Tensor &featMetadata, const nvcv::Tensor &featDescriptors,
                    const nvcv::Tensor &numFeatures, int numOctaveLayers, float contrastThreshold, float edgeThreshold,
                    float initSigma, NVCVSIFTFlagType flags) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline SIFT::SIFT(int3 maxShape, int maxOctaveLayers)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaSIFTCreate(&h, maxShape, maxOctaveLayers));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void SIFT::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &featCoords,
                             const nvcv::Tensor &featMetadata, const nvcv::Tensor &featDescriptors,
                             const nvcv::Tensor &numFeatures, int numOctaveLayers, float contrastThreshold,
                             float edgeThreshold, float initSigma, NVCVSIFTFlagType flags) const
{
    nvcv::detail::CheckThrow(cvcudaSIFTSubmit(m_handle.get(), stream, in.handle(), featCoords.handle(),
                                              featMetadata.handle(), featDescriptors.handle(), numFeatures.handle(),
                                              numOctaveLayers, contrastThreshold, edgeThreshold, initSigma, flags));
}

inline NVCVOperatorHandle SIFT::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_SIFT_HPP
