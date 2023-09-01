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

#include "priv/OpSIFT.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaSIFTCreate, (NVCVOperatorHandle * handle, int3 maxShape, int maxOctaveLayers))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::SIFT(maxShape, maxOctaveLayers));
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaSIFTSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle featCoords,
                   NVCVTensorHandle featMetadata, NVCVTensorHandle featDescriptors, NVCVTensorHandle numFeatures,
                   int numOctaveLayers, float contrastThreshold, float edgeThreshold, float initSigma,
                   NVCVSIFTFlagType flags))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle _in(in), _featCoords(featCoords), _featMetadata(featMetadata),
                _featDescriptors(featDescriptors), _numFeatures(numFeatures);
            priv::ToDynamicRef<priv::SIFT>(handle)(stream, _in, _featCoords, _featMetadata, _featDescriptors,
                                                   _numFeatures, numOctaveLayers, contrastThreshold, edgeThreshold,
                                                   initSigma, flags);
        });
}
