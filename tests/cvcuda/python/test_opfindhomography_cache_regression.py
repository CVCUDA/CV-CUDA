# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Regression for the PyOpFindHomography cache-key bug:
#
# python/mod_cvcuda/operators/OpFindHomography.cpp declares a cache Key whose
# ``doGetHash()`` returns 0, ``payloadSize()`` returns 0, and
# ``doIsCompatible()`` is unconditionally true. Combined with ``fetch()``
# picking by ``payloadSize > maxPayloadSize`` starting at 0, this makes the
# cache hand back the first inserted op for any subsequent request,
# regardless of ``batchSize`` or ``maxNumPoints``.
#
# When a smaller op is reused for a larger request, ``RunFindHomography``
# (src/cvcuda/priv/OpFindHomography.cu) reads ``batchSize = models.shape(0)``
# and issues cudaMemsetAsync + kernel launches sized by the larger batch —
# but ``DeviceState`` buffers were allocated for the smaller one. The OOB
# writes corrupt adjacent device memory and the resulting homographies are
# garbage.

import cupy
import numpy as np

import cvcuda


def test_findhomography_cache_reuse_regression():
    # Start from a clean cache so the test does not depend on pytest
    # collection order.
    cvcuda.clear_cache()

    num_points = 1024
    rng = np.random.default_rng(0)
    # Well-spread, non-collinear 2D points so the solver is non-degenerate.
    pts = rng.uniform(-100.0, 100.0, size=(num_points, 2)).astype(np.float32)

    def make_input(n):
        # Identical points across the batch dim; src == dst => identity H.
        batch = np.broadcast_to(pts, (n, num_points, 2)).copy()
        return cvcuda.as_tensor(cupy.asarray(batch), layout="NWC")

    # 1. Prime the op cache with a (batchSize=1, maxNumPoints=num_points) op.
    src_small = make_input(1)
    dst_small = make_input(1)
    cvcuda.findhomography(src_small, dst_small)

    # 2. Larger batch. A correct cache would create a new op sized for 64;
    #    the buggy cache returns the cached small op, and RunFindHomography
    #    writes past the end of the undersized DeviceState buffers.
    n = 64
    src_big = make_input(n)
    dst_big = make_input(n)
    out = cvcuda.findhomography(src_big, dst_big)

    h = cupy.asarray(out.cuda()).get()
    assert h.shape == (n, 3, 3)

    # Expected homography: identity, up to scale.
    expected = np.eye(3, dtype=np.float32)
    for i in range(n):
        hi = h[i]
        if abs(hi[2, 2]) > 1e-8:
            hi = hi / hi[2, 2]
        assert np.allclose(hi, expected, atol=1e-2), (
            f"homography for sample {i} is not identity (src == dst). "
            "PyOpFindHomography cache reuse bug suspected — the cached "
            f"(batchSize=1, maxNumPoints={num_points}) op was returned for "
            f"a batchSize={n} request, and RunFindHomography's memsets + "
            f"kernel launches wrote past the end of the undersized buffers.\n"
            f"Got:\n{hi}\nExpected:\n{expected}"
        )

    cvcuda.clear_cache()
