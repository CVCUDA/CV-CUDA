# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cvcuda
import pytest as t
import numpy as np
import cvcuda_util as util


RNG = np.random.default_rng(0)


class ref:
    """Python reference class to store constants and test output content"""

    num_dtype = np.int32
    out_dtype = np.int32
    dist_dtype = np.float32

    def absdiff(a, b):
        if type(a) == float:
            return abs(a - b)
        else:
            return b - a if a < b else a - b

    def distance(p1, p2, norm_type):
        if norm_type == cvcuda.Norm.HAMMING:
            return sum([bin(c1 ^ c2).count("1") for c1, c2 in zip(p1, p2)])
        elif norm_type == cvcuda.Norm.L1:
            return sum([abs(ref.absdiff(c1, c2)) for c1, c2 in zip(p1, p2)])
        elif norm_type == cvcuda.Norm.L2:
            return np.sqrt(sum([ref.absdiff(c1, c2) ** 2 for c1, c2 in zip(p1, p2)]))

    def brute_force_matcher(batch_set1, batch_set2, cross_check, norm_type):
        batch_matches = []
        batch_num_matches = []
        batch_distances = []
        for set1, set2 in zip(batch_set1, batch_set2):
            batch_matches.append([])
            batch_num_matches.append(0)
            batch_distances.append([])
            for set1_idx, p1 in enumerate(set1):
                dist1to2 = []
                for set2_idx, p2 in enumerate(set2):
                    dist1to2.append((ref.distance(p1, p2, norm_type), set2_idx))
                sorted_dist_ids = sorted(dist1to2)
                if cross_check:
                    p2 = set2[sorted_dist_ids[0][1]]
                    dist2to1 = []
                    for q1_idx, q1 in enumerate(set1):
                        dist2to1.append((ref.distance(q1, p2, norm_type), q1_idx))
                    cc_sorted_dist_ids = sorted(dist2to1)
                    if cc_sorted_dist_ids[0][1] == set1_idx:
                        batch_matches[-1].append([set1_idx, sorted_dist_ids[0][1]])
                        batch_distances[-1].append(sorted_dist_ids[0][0])
                        batch_num_matches[-1] += 1
                else:
                    batch_matches[-1].append([set1_idx, sorted_dist_ids[0][1]])
                    batch_distances[-1].append(sorted_dist_ids[0][0])
                    batch_num_matches[-1] += 1
        return batch_matches, batch_num_matches, batch_distances

    def sort(matches, num_matches, distances):
        output = []
        for sample_idx in range(len(matches)):
            for match_idx in range(num_matches[sample_idx]):
                set1_idx = matches[sample_idx][match_idx][0]
                set2_idx = matches[sample_idx][match_idx][1]
                distance = distances[sample_idx][match_idx]
                output.append((sample_idx, set1_idx, set2_idx, distance))
        return sorted(output)


@t.mark.parametrize(
    "set_shape, set_dtype",
    [
        ((1, 11, 1), np.uint8),
        ((2, 12, 2), np.uint32),
        ((3, 22, 3), np.float32),
        ((4, 123, 32), np.uint8),
        ((3, 234, 26), np.uint32),
        ((2, 345, 13), np.float32),
    ],
)
def test_op_match_api(set_shape, set_dtype):
    set1 = cvcuda.Tensor(set_shape, set_dtype, "NMD")
    set2 = cvcuda.Tensor(set_shape, set_dtype, "NMD")

    matches, num_matches, distances = cvcuda.match(set1, set2)
    assert num_matches is None and distances is None
    assert matches.shape == (set_shape[0], set_shape[1], 2)
    assert matches.layout == "NMA"
    assert matches.dtype == ref.out_dtype

    _, num_matches, _ = cvcuda.match(set1, set2, num_matches=True)
    assert num_matches.shape == (set_shape[0],)
    assert num_matches.layout == "N"
    assert num_matches.dtype == ref.out_dtype

    _, _, distances = cvcuda.match(set1, set2, distances=True)
    assert distances.shape == (set_shape[0], set_shape[1])
    assert distances.layout == "NM"
    assert distances.dtype == ref.dist_dtype

    _, num_matches, _ = cvcuda.match(set1, set2, cross_check=True)
    assert num_matches is not None

    _, num_matches, distances = cvcuda.match(
        set1, set2, num_matches=True, distances=True
    )
    assert num_matches is not None and distances is not None

    num_set1 = cvcuda.Tensor(set_shape[:1], ref.num_dtype, "N")
    num_set2 = cvcuda.Tensor(set_shape[:1], ref.num_dtype, "N")

    big_matches, _, _ = cvcuda.match(
        set1,
        set2,
        num_set1,
        num_set2,
        cross_check=False,
        norm_type=cvcuda.Norm.L2,
        matches_per_point=64,
        algo_choice=cvcuda.Matcher.BRUTE_FORCE,
    )
    assert big_matches.shape == (set_shape[0], set_shape[1] * 64, 2)

    tmp = cvcuda.match_into(
        matches,
        num_matches,
        distances,
        set1,
        set2,
        num_set1,
        num_set2,
    )
    assert tmp[0] is matches and tmp[1] is num_matches and tmp[2] is distances

    stream = cvcuda.Stream()
    matches, _, _ = cvcuda.match(set1, set2, num_set1, num_set2, stream=stream)
    assert matches.shape == (set_shape[0], set_shape[1], 2)
    assert matches.layout == "NMA"
    assert matches.dtype == ref.out_dtype

    tmp = cvcuda.match_into(
        matches,
        None,
        None,
        set1,
        set2,
        None,
        None,
        False,
        1,
        cvcuda.Norm.L1,
        cvcuda.Matcher.BRUTE_FORCE,
        stream=stream,
    )
    assert tmp[0] is matches and tmp[1] is None and tmp[2] is None


@t.mark.parametrize(
    "set_shape, set_dtype, cross_check, norm_type",
    [
        ((1, 18, 32), np.uint8, False, cvcuda.Norm.HAMMING),
        ((2, 28, 21), np.uint32, False, cvcuda.Norm.L1),
        ((3, 36, 10), np.float32, False, cvcuda.Norm.L2),
        ((2, 17, 33), np.uint8, True, cvcuda.Norm.L1),
        ((3, 57, 13), np.float32, True, cvcuda.Norm.L2),
    ],
)
def test_op_match_content(set_shape, set_dtype, cross_check, norm_type):
    h_set1 = util.generate_data(set_shape, set_dtype, max_random=255, rng=RNG)
    h_set2 = util.generate_data(set_shape, set_dtype, max_random=255, rng=RNG)

    set1 = util.to_nvcv_tensor(h_set1, "NMD")
    set2 = util.to_nvcv_tensor(h_set2, "NMD")

    matches, num_matches, distances = cvcuda.match(
        set1,
        set2,
        num_matches=True,
        distances=True,
        cross_check=cross_check,
        norm_type=norm_type,
        algo_choice=cvcuda.Matcher.BRUTE_FORCE,
    )

    h_test_matches = util.to_cpu_numpy_buffer(matches.cuda())
    h_test_num_matches = util.to_cpu_numpy_buffer(num_matches.cuda())
    h_test_distances = util.to_cpu_numpy_buffer(distances.cuda())

    h_gold_matches, h_gold_num_matches, h_gold_distances = ref.brute_force_matcher(
        h_set1, h_set2, cross_check, norm_type
    )

    h_test_output = ref.sort(h_test_matches, h_test_num_matches, h_test_distances)
    h_gold_output = ref.sort(h_gold_matches, h_gold_num_matches, h_gold_distances)

    np.testing.assert_allclose(h_test_output, h_gold_output, rtol=1e-5, atol=1e-5)
