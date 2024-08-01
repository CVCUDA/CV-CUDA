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

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpPairwiseMatcher.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <optional>

namespace cvcudapy {

using TupleTensor3 = std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>>;

namespace {

TupleTensor3 PairwiseMatcherInto(Tensor &matches, std::optional<Tensor> numMatches, std::optional<Tensor> distances,
                                 Tensor &set1, Tensor &set2, std::optional<Tensor> numSet1,
                                 std::optional<Tensor> numSet2, bool crossCheck, int matchesPerPoint,
                                 std::optional<NVCVNormType> normType, NVCVPairwiseMatcherType algoChoice,
                                 std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (!normType)
    {
        normType = NVCV_NORM_L2;
    }

    auto op = CreateOperator<cvcuda::PairwiseMatcher>(algoChoice);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {set1, set2});
    guard.add(LockMode::LOCK_MODE_WRITE, {matches});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    if (numSet1)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*numSet1});
    }
    if (numSet2)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*numSet2});
    }
    if (numMatches)
    {
        guard.add(LockMode::LOCK_MODE_WRITE, {*numMatches});
    }
    if (distances)
    {
        guard.add(LockMode::LOCK_MODE_WRITE, {*distances});
    }

    op->submit(pstream->cudaHandle(), set1, set2, (numSet1 ? *numSet1 : nvcv::Tensor{nullptr}),
               (numSet2 ? *numSet2 : nvcv::Tensor{nullptr}), matches,
               (numMatches ? *numMatches : nvcv::Tensor{nullptr}), (distances ? *distances : nvcv::Tensor{nullptr}),
               crossCheck, matchesPerPoint, *normType);

    return TupleTensor3(std::move(matches), numMatches, distances);
}

TupleTensor3 PairwiseMatcher(Tensor &set1, Tensor &set2, std::optional<Tensor> numSet1, std::optional<Tensor> numSet2,
                             std::optional<bool> numMatches, bool distances, bool crossCheck, int matchesPerPoint,
                             std::optional<NVCVNormType> normType, NVCVPairwiseMatcherType algoChoice,
                             std::optional<Stream> pstream)
{
    nvcv::TensorShape set1Shape = set1.shape();
    nvcv::TensorShape set2Shape = set2.shape();

    if (set1Shape.rank() != 3 || set2Shape.rank() != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input sets must be rank-3 tensors");
    }

    int64_t numSamples = set1Shape[0];
    int64_t maxMatches = std::max(set1Shape[1], set2Shape[1]) * matchesPerPoint;

    if (!numMatches)
    {
        numMatches = crossCheck;
    }

    // clang-format off

    Tensor matches = Tensor::Create({{numSamples, maxMatches, 2}, "NMA"}, nvcv::TYPE_S32);

    std::optional<Tensor> numMatchesTensor, distancesTensor;

    if (*numMatches)
    {
        numMatchesTensor = Tensor::Create({{numSamples}, "N"}, nvcv::TYPE_S32);
    }
    if (distances)
    {
        distancesTensor = Tensor::Create({{numSamples, maxMatches}, "NM"}, nvcv::TYPE_F32);
    }

    // clang-format on

    return PairwiseMatcherInto(matches, numMatchesTensor, distancesTensor, set1, set2, numSet1, numSet2, crossCheck,
                               matchesPerPoint, normType, algoChoice, pstream);
}

} // namespace

void ExportOpPairwiseMatcher(py::module &m)
{
    using namespace pybind11::literals;

    m.def("match", &PairwiseMatcher, "set1"_a, "set2"_a, "num_set1"_a = nullptr, "num_set2"_a = nullptr,
          "num_matches"_a = nullptr, "distances"_a = false, "cross_check"_a = false, "matches_per_point"_a = 1,
          "norm_type"_a = nullptr, "algo_choice"_a = NVCV_BRUTE_FORCE, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Pairwise matcher operation on the given CUDA stream.

        See also:
            Refer to the CV-CUDA C API reference for this operator for more details and usage examples.

        Args:
            set1 (nvcv.Tensor): Input tensor with 1st set of points.
            set2 (nvcv.Tensor): Input tensor with 2nd set of points.
            num_set1 (nvcv.Tensor, optional): Input tensor with number of valid points in the 1st set.  If not provided,
                                         consider the entire set1 containing valid points.
            num_set2 (nvcv.Tensor, optional): Input tensor with number of valid points in the 2nd set.  If not provided,
                                         consider the entire set2 containing valid points.
            num_matches (bool, optional): Use True to return the number of matches.  If not provided, it is set
                                          to True if crossCheck=True and False otherwise.
            distances (bool, optional): Use True to return the match distances.
            cross_check (bool, optional): Use True to cross check best matches, a best match is only returned if it is
                                          the best match (minimum distance) from 1st set to 2nd set and vice versa.
            matches_per_point (Number, optional): Number of best matches to return per point.
            norm_type (cvcuda.Norm, optional): Choice on how distances are normalized.  Defaults to cvcuda.Norm.L2.
            algo_choice (cvcuda.Matcher, optional): Choice of the algorithm to perform the match.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[nvcv.Tensor, nvcv.Tensor, nvcv.Tensor]: A tuple with output matches, number of matches and their distances.
                                           The number of matches tensor may be None if its argument is False.
                                           The distances tensor may be None if its argument is False.

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA operator.
    )pbdoc");

    m.def("match_into", &PairwiseMatcherInto, "matches"_a, "num_matches"_a = nullptr, "distances"_a = nullptr, "set1"_a,
          "set2"_a, "num_set1"_a = nullptr, "num_set2"_a = nullptr, "cross_check"_a = false, "matches_per_point"_a = 1,
          "norm_type"_a = nullptr, "algo_choice"_a = NVCV_BRUTE_FORCE, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the Pairwise matcher operation on the given CUDA stream.

        See also:
            Refer to the CV-CUDA C API reference for this operator for more details and usage examples.

        Args:
            matches (nvcv.Tensor): Output tensor with matches.
            num_matches (nvcv.Tensor, optional): Output tensor with number of matches.
            distances (nvcv.Tensor, optional): Output tensor with match distances.
            set1 (nvcv.Tensor): Input tensor with 1st set of points.
            set2 (nvcv.Tensor): Input tensor with 2nd set of points.
            num_set1 (nvcv.Tensor, optional): Input tensor with number of valid points in the 1st set.  If not provided,
                                         consider the entire set1 containing valid points.
            num_set2 (nvcv.Tensor, optional): Input tensor with number of valid points in the 2nd set.  If not provided,
                                         consider the entire set2 containing valid points.
            cross_check (bool, optional): Use True to cross check best matches, a best match is only returned if it is
                                          the best match (minimum distance) from 1st set to 2nd set and vice versa.
            matches_per_point (Number, optional): Number of best matches to return per point.
            norm_type (cvcuda.Norm, optional): Choice on how distances are normalized.  Defaults to cvcuda.Norm.L2.
            algo_choice (cvcuda.Matcher, optional): Choice of the algorithm to perform the match.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[nvcv.Tensor, nvcv.Tensor, nvcv.Tensor]: A tuple with output matches, number of matches and their distances.
                                           The number of matches tensor may be None if its argument is None.
                                           The distances tensor may be None if its argument is None.

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
