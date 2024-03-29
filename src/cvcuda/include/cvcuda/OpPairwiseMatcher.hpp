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
 * @file OpPairwiseMatcher.hpp
 *
 * @brief Defines the public C++ Class for the PairwiseMatcher operation.
 * @defgroup NVCV_CPP_ALGORITHM_PAIRWISE_MATCHER PairwiseMatcher
 * @{
 */

#ifndef CVCUDA_PAIRWISE_MATCHER_HPP
#define CVCUDA_PAIRWISE_MATCHER_HPP

#include "IOperator.hpp"
#include "OpPairwiseMatcher.h"

#include <cuda_runtime.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class PairwiseMatcher final : public IOperator
{
public:
    explicit PairwiseMatcher(NVCVPairwiseMatcherType algoChoice);

    ~PairwiseMatcher();

    void operator()(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                    const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2, const nvcv::Tensor &matches,
                    const nvcv::Tensor &numMatches, const nvcv::Tensor &distances, bool crossCheck, int matchesPerPoint,
                    NVCVNormType normType);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline PairwiseMatcher::PairwiseMatcher(NVCVPairwiseMatcherType algoChoice)
{
    nvcv::detail::CheckThrow(cvcudaPairwiseMatcherCreate(&m_handle, algoChoice));
    assert(m_handle);
}

inline PairwiseMatcher::~PairwiseMatcher()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void PairwiseMatcher::operator()(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                                        const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2,
                                        const nvcv::Tensor &matches, const nvcv::Tensor &numMatches,
                                        const nvcv::Tensor &distances, bool crossCheck, int matchesPerPoint,
                                        NVCVNormType normType)
{
    nvcv::detail::CheckThrow(cvcudaPairwiseMatcherSubmit(
        m_handle, stream, set1.handle(), set2.handle(), numSet1.handle(), numSet2.handle(), matches.handle(),
        numMatches.handle(), distances.handle(), crossCheck, matchesPerPoint, normType));
}

inline NVCVOperatorHandle PairwiseMatcher::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_PAIRWISE_MATCHER_HPP
