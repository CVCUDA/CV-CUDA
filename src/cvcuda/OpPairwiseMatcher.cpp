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

#include "priv/OpPairwiseMatcher.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaPairwiseMatcherCreate,
                  (NVCVOperatorHandle * handle, NVCVPairwiseMatcherType algoChoice))
{
    return nvcv::ProtectCall(
        [&handle, &algoChoice]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<cvcuda::priv::PairwiseMatcher>(algoChoice);
        });
}

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaPairwiseMatcherSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle set1, NVCVTensorHandle set2,
                   NVCVTensorHandle numSet1, NVCVTensorHandle numSet2, NVCVTensorHandle matches,
                   NVCVTensorHandle numMatches, NVCVTensorHandle distances, bool crossCheck, int matchesPerPoint,
                   NVCVNormType normType))
{
    CVCUDA_NVTX_RANGE("cvcudaPairwiseMatcherSubmit");
    return nvcv::ProtectCall(
        [&handle, &stream, &set1, &set2, &numSet1, &numSet2, &matches, &numMatches, &distances, &crossCheck,
         &matchesPerPoint, &normType]
        {
            cvcuda::priv::ToDynamicRef<cvcuda::priv::PairwiseMatcher>(handle)(
                stream, nvcv::TensorWrapHandle{set1}.resource(), nvcv::TensorWrapHandle{set2}.resource(),
                nvcv::TensorWrapHandle{numSet1}.resource(), nvcv::TensorWrapHandle{numSet2}.resource(),
                nvcv::TensorWrapHandle{matches}.resource(), nvcv::TensorWrapHandle{numMatches}.resource(),
                nvcv::TensorWrapHandle{distances}.resource(), crossCheck, matchesPerPoint, normType);
        });
}
