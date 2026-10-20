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

#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpPairwiseMatcher.hpp>

#include <optional>

namespace cvcudapy {

namespace {
using native::Stream;
using native::Tensor;
using TupleTensor3    = std::tuple<std::shared_ptr<Tensor>, std::optional<std::shared_ptr<Tensor>>,
                                std::optional<std::shared_ptr<Tensor>>>;
using TupleTensor3Ref = std::tuple<Tensor *, std::optional<Tensor *>, std::optional<Tensor *>>;

const nvcv::Tensor &AsNativeTensor(const std::optional<Tensor *> &tensor, const nvcv::Tensor &nullTensor)
{
    return tensor && *tensor ? (*tensor)->impl() : nullTensor;
}

TupleTensor3Ref PairwiseMatcherInto(Tensor &matches, std::optional<Tensor *> numMatches,
                                    std::optional<Tensor *> distances, Tensor &set1, Tensor &set2,
                                    const std::optional<Tensor *> &numSet1, const std::optional<Tensor *> &numSet2,
                                    bool crossCheck, int matchesPerPoint, std::optional<NVCVNormType> normType,
                                    NVCVPairwiseMatcherType algoChoice, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    if (!normType)
    {
        normType = NVCV_NORM_L2;
    }

    auto op = CreateOperator<cvcuda::PairwiseMatcher>(algoChoice);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {set1, set2});
    guard.add(LockMode::LOCK_MODE_WRITE, {matches});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    if (numSet1 && *numSet1)
    {
        guard.add(LockMode::LOCK_MODE_READ, {**numSet1});
    }
    if (numSet2 && *numSet2)
    {
        guard.add(LockMode::LOCK_MODE_READ, {**numSet2});
    }
    if (numMatches && *numMatches)
    {
        guard.add(LockMode::LOCK_MODE_WRITE, {**numMatches});
    }
    if (distances && *distances)
    {
        guard.add(LockMode::LOCK_MODE_WRITE, {**distances});
    }

    guard.run(
        [&numSet1, &numSet2, &numMatches, &distances, &op, &stream, &set1, &set2, &matches, &crossCheck,
         &matchesPerPoint, &normType]()
        {
            const nvcv::Tensor nullTensor{nullptr};
            op->submit(stream.handle(), set1.impl(), set2.impl(), AsNativeTensor(numSet1, nullTensor),
                       AsNativeTensor(numSet2, nullTensor), matches.impl(), AsNativeTensor(numMatches, nullTensor),
                       AsNativeTensor(distances, nullTensor), crossCheck, matchesPerPoint, *normType);
        });

    return {&matches, numMatches, distances};
}

TupleTensor3 PairwiseMatcher(Tensor &set1, Tensor &set2, const std::optional<Tensor *> &numSet1,
                             const std::optional<Tensor *> &numSet2, std::optional<bool> numMatches, bool distances,
                             bool crossCheck, int matchesPerPoint, std::optional<NVCVNormType> normType,
                             NVCVPairwiseMatcherType algoChoice, std::optional<Stream *> pstream)
{
    const nvcv::TensorShape &set1Shape = set1.impl().shape();
    const nvcv::TensorShape &set2Shape = set2.impl().shape();

    if (set1Shape.rank() != 3 || set2Shape.rank() != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input sets must be rank-3 tensors");
    }

    int64_t numSamples = set1Shape[0];
    int64_t maxMatches = std::max(set1Shape[1], set2Shape[1]) * matchesPerPoint;

    if (!numMatches.has_value())
    {
        numMatches = crossCheck;
    }

    // clang-format off

    auto matches = native::CreateTensor({{numSamples, maxMatches, 2}, "NMA"}, nvcv::TYPE_S32);

    std::optional<std::shared_ptr<Tensor>> numMatchesTensor;
    std::optional<std::shared_ptr<Tensor>> distancesTensor;

    if (*numMatches)
    {
        numMatchesTensor = native::CreateTensor({{numSamples}, "N"}, nvcv::TYPE_S32);
    }
    if (distances)
    {
        distancesTensor = native::CreateTensor({{numSamples, maxMatches}, "NM"}, nvcv::TYPE_F32);
    }

    // clang-format on

    PairwiseMatcherInto(*matches, numMatchesTensor ? std::optional<Tensor *>{numMatchesTensor->get()} : std::nullopt,
                        distancesTensor ? std::optional<Tensor *>{distancesTensor->get()} : std::nullopt, set1, set2,
                        numSet1, numSet2, crossCheck, matchesPerPoint, normType, algoChoice, pstream);
    return {std::move(matches), std::move(numMatchesTensor), std::move(distancesTensor)};
}

} // namespace

void ExportOpPairwiseMatcher(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("match", NvtxTrace("cvcuda.match", &PairwiseMatcher), "set1"_a, "set2"_a, "num_set1"_a = nullptr,
          "num_set2"_a = nullptr, "num_matches"_a = nullptr, "distances"_a = false, "cross_check"_a = false,
          "matches_per_point"_a = 1, "norm_type"_a = nullptr, "algo_choice"_a = NVCV_BRUTE_FORCE, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Pairwise matcher operation on the given CUDA stream.


        Args:
            set1 (cvcuda.Tensor): Input tensor with 1st set of points.
            set2 (cvcuda.Tensor): Input tensor with 2nd set of points.
            num_set1 (cvcuda.Tensor, optional): Input tensor with number of valid points in the 1st set.  If not provided,
                                         consider the entire set1 containing valid points.
            num_set2 (cvcuda.Tensor, optional): Input tensor with number of valid points in the 2nd set.  If not provided,
                                         consider the entire set2 containing valid points.
            num_matches (bool, optional): Use True to return the number of matches.  If not provided, it is set
                                          to True if crossCheck=True and False otherwise.
            distances (bool, optional): Use True to return the match distances.
            cross_check (bool, optional): Use True to cross check best matches, a best match is only returned if it is
                                          the best match (minimum distance) from 1st set to 2nd set and vice versa.
            matches_per_point (Number, optional): Number of best matches to return per point.
            norm_type (cvcuda.Norm, optional): Choice on how distances are normalized.  Defaults to cvcuda.Norm.L2.
            algo_choice (cvcuda.Matcher, optional): Choice of the algorithm to perform the match.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with output matches, number of matches and their distances.
                                           The number of matches tensor may be None if its argument is False.
                                           The distances tensor may be None if its argument is False.

    )pbdoc");

    m.def("match_into", NvtxTrace("cvcuda.match_into", &PairwiseMatcherInto), "matches"_a, "num_matches"_a = nullptr,
          "distances"_a = nullptr, "set1"_a, "set2"_a, "num_set1"_a = nullptr, "num_set2"_a = nullptr,
          "cross_check"_a = false, "matches_per_point"_a = 1, "norm_type"_a = nullptr,
          "algo_choice"_a = NVCV_BRUTE_FORCE, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Executes the Pairwise matcher operation on the given CUDA stream.


        Args:
            matches (cvcuda.Tensor): Output tensor with matches.
            num_matches (cvcuda.Tensor, optional): Output tensor with number of matches.
            distances (cvcuda.Tensor, optional): Output tensor with match distances.
            set1 (cvcuda.Tensor): Input tensor with 1st set of points.
            set2 (cvcuda.Tensor): Input tensor with 2nd set of points.
            num_set1 (cvcuda.Tensor, optional): Input tensor with number of valid points in the 1st set.  If not provided,
                                         consider the entire set1 containing valid points.
            num_set2 (cvcuda.Tensor, optional): Input tensor with number of valid points in the 2nd set.  If not provided,
                                         consider the entire set2 containing valid points.
            cross_check (bool, optional): Use True to cross check best matches, a best match is only returned if it is
                                          the best match (minimum distance) from 1st set to 2nd set and vice versa.
            matches_per_point (Number, optional): Number of best matches to return per point.
            norm_type (cvcuda.Norm, optional): Choice on how distances are normalized.  Defaults to cvcuda.Norm.L2.
            algo_choice (cvcuda.Matcher, optional): Choice of the algorithm to perform the match.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with output matches, number of matches and their distances.
                                           The number of matches tensor may be None if its argument is None.
                                           The distances tensor may be None if its argument is None.

    )pbdoc");
}

} // namespace cvcudapy
