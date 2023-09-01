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

/**
 * @file OpSIFT.hpp
 *
 * @brief Defines the private C++ Class for the SIFT operation.
 */

#ifndef CVCUDA_PRIV_SIFT_HPP
#define CVCUDA_PRIV_SIFT_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <cvcuda/OpSIFT.h>
#include <nvcv/Tensor.hpp>

#include <memory>
#include <vector>

namespace cvcuda::priv {

class SIFT final : public IOperator
{
public:
    explicit SIFT(int3 maxShape, int maxOctaveLayers);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &featCoords,
                    const nvcv::Tensor &featMetadata, const nvcv::Tensor &featDescriptors,
                    const nvcv::Tensor &numFeatures, int numOctaveLayers, float contrastThreshold, float edgeThreshold,
                    float initSigma, NVCVSIFTFlagType flags) const;

private:
    // The pyramid type stores one tensor per octave (SIFT pyramid level) where a single tensor contains all layers
    // within an octave (Gaussian filtering done) stacked, see OpSIFT.cu comments for more
    using PyramidType = std::vector<nvcv::Tensor>;

    void CreatePyramids();
    void ReshapePyramids(const int3 &shape, int numOctaves, int numOctaveLayers) const;

    template<typename DT>
    void ComputePyramids(const nvcv::TensorDataStridedCuda &inData, int3 currShape, bool expandInput, int numOctaves,
                         int numOctaveLayers, float initSigma, cudaStream_t stream) const;

    template<typename DT>
    void FindExtrema(const nvcv::TensorDataStridedCuda &featCoordsData,
                     const nvcv::TensorDataStridedCuda &featMetadaData,
                     const nvcv::TensorDataStridedCuda &featDescriptorsData, int maxCapacity,
                     const nvcv::TensorDataStridedCuda &numFeaturesData, int3 currShape, int firstOctave,
                     int numOctaves, int numOctaveLayers, float contrastThreshold, float edgeThreshold, float initSigma,
                     cudaStream_t stream) const;

    int3 m_maxShape;
    int  m_maxOctaves, m_maxOctaveLayers;

    // Maximum allowed pyramids and run (submit) time pyramids
    PyramidType         m_maxPyramidGaussian, m_maxPyramidDoG;
    mutable PyramidType m_runPyramidGaussian, m_runPyramidDoG; // mutable as it changes during run-time
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_SIFT_HPP
