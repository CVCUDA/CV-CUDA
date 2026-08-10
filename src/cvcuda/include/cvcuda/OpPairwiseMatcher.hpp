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

#include <cassert>

namespace cvcuda {

class PairwiseMatcher final : public IOperator
{
public:
    explicit PairwiseMatcher(NVCVPairwiseMatcherType algoChoice);

    void operator()(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                    const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2, const nvcv::Tensor &matches,
                    const nvcv::Tensor &numMatches, const nvcv::Tensor &distances, bool crossCheck, int matchesPerPoint,
                    NVCVNormType normType) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline PairwiseMatcher::PairwiseMatcher(NVCVPairwiseMatcherType algoChoice)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaPairwiseMatcherCreate(&h, algoChoice));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void PairwiseMatcher::operator()(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                                        const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2,
                                        const nvcv::Tensor &matches, const nvcv::Tensor &numMatches,
                                        const nvcv::Tensor &distances, bool crossCheck, int matchesPerPoint,
                                        NVCVNormType normType) const
{
    nvcv::detail::CheckThrow(cvcudaPairwiseMatcherSubmit(
        m_handle.get(), stream, set1.handle(), set2.handle(), numSet1.handle(), numSet2.handle(), matches.handle(),
        numMatches.handle(), distances.handle(), crossCheck, matchesPerPoint, normType));
}

inline NVCVOperatorHandle PairwiseMatcher::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_PAIRWISE_MATCHER_HPP
