/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/OpPairwiseMatcher.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaPairwiseMatcherCreate,
                  (NVCVOperatorHandle * handle, NVCVPairwiseMatcherType algoChoice))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new cvcuda::priv::PairwiseMatcher(algoChoice));
        });
}

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaPairwiseMatcherSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle set1, NVCVTensorHandle set2,
                   NVCVTensorHandle numSet1, NVCVTensorHandle numSet2, NVCVTensorHandle matches,
                   NVCVTensorHandle numMatches, NVCVTensorHandle distances, bool crossCheck, int matchesPerPoint,
                   NVCVNormType normType))
{
    return nvcv::ProtectCall(
        [&]
        {
            cvcuda::priv::ToDynamicRef<cvcuda::priv::PairwiseMatcher>(handle)(
                stream, nvcv::TensorWrapHandle{set1}, nvcv::TensorWrapHandle{set2}, nvcv::TensorWrapHandle{numSet1},
                nvcv::TensorWrapHandle{numSet2}, nvcv::TensorWrapHandle{matches}, nvcv::TensorWrapHandle{numMatches},
                nvcv::TensorWrapHandle{distances}, crossCheck, matchesPerPoint, normType);
        });
}
