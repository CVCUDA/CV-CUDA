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

import gc
import subprocess
import sys
import textwrap

import numpy as np

import cvcuda


_WARP_AFFINE_ZERO_CAPACITY_CACHE_SCRIPT = textwrap.dedent(
    """\
    import numpy as np
    import cvcuda

    cvcuda.clear_cache()
    src = cvcuda.Tensor((1, 4, 4, 1), np.uint8, "NHWC")
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    # Tensor WarpAffine deliberately uses constructor capacity zero. Its
    # generic cache state must remain distinct from the leased VarShape state.
    cvcuda.warp_affine(
        src, matrix, cvcuda.Interp.NEAREST, border_value=[0]
    )
    cvcuda.Stream.current.sync()

    empty = cvcuda.ImageBatchVarShape(0)
    transforms = cvcuda.Tensor((1, 6), np.float32, "NC")
    try:
        cvcuda.warp_affine(
            empty,
            transforms,
            cvcuda.Interp.NEAREST,
            border_value=[0],
        )
    except RuntimeError as exc:
        assert "not initialized properly" in str(exc), str(exc)
    else:
        raise AssertionError("zero-capacity VarShape WarpAffine unexpectedly succeeded")

    # Re-fetch the capacity-zero Tensor operator after the VarShape lookup to
    # prove neither cache entry can be cast as the other's metadata type.
    cvcuda.warp_affine(
        src, matrix, cvcuda.Interp.NEAREST, border_value=[0]
    )
    cvcuda.Stream.current.sync()
    print("PASS", flush=True)
    """
)


def test_cache_metadata_is_not_part_of_the_public_mro():
    assert not hasattr(cvcuda, "_CacheItem")
    assert not hasattr(cvcuda.internal, "_ExternalCacheItem")

    expected_bases = {
        cvcuda.Resource: (object,),
        cvcuda.Stream: (object,),
        cvcuda.Container: (cvcuda.Resource,),
        cvcuda.Tensor: (cvcuda.Container,),
        cvcuda.Image: (cvcuda.Container,),
        cvcuda.TensorBatch: (cvcuda.Container,),
        cvcuda.ImageBatchVarShape: (cvcuda.Container,),
    }
    for cls, bases in expected_bases.items():
        assert cls.__bases__ == bases
        assert all(base.__name__ != "_CacheItem" for base in cls.__mro__)


def test_composed_cache_reuses_resource_and_stream_objects():
    cvcuda.clear_cache()

    tensor = cvcuda.Tensor((1, 7, 11, 3), np.uint8, "NHWC")
    stream = cvcuda.Stream()
    tensor_id = tensor.id
    stream_id = stream.id

    assert cvcuda.internal.nbytes_in_cache(tensor) > 0
    assert cvcuda.internal.nbytes_in_cache(stream) == 0
    assert cvcuda.cache_size(cvcuda.ThreadScope.LOCAL) == 2

    del tensor, stream
    gc.collect()

    assert cvcuda.Tensor((1, 7, 11, 3), np.uint8, "NHWC").id == tensor_id
    assert cvcuda.Stream().id == stream_id
    assert cvcuda.cache_size(cvcuda.ThreadScope.LOCAL) == 2


def test_composed_cache_tracks_the_wrapped_resource_lifetime():
    cvcuda.clear_cache()

    tensor = cvcuda.Tensor((1, 5, 13, 2), np.uint8, "NHWC")
    held_id = tensor.id
    live_owner = tensor
    del tensor
    gc.collect()

    replacement = cvcuda.Tensor((1, 5, 13, 2), np.uint8, "NHWC")
    assert replacement.id != held_id

    del live_owner
    gc.collect()

    # The first wrapper becomes reusable while the replacement remains live.
    assert cvcuda.Tensor((1, 5, 13, 2), np.uint8, "NHWC").id == held_id


def test_external_cache_capi_round_trip_is_preserved():
    cvcuda._test.external_cache_capi_round_trip()


def test_external_resource_capi_round_trip_uses_generic_holder_cast():
    assert cvcuda._test.external_resource_capi_round_trip() >= 0


def test_warp_affine_tensor_and_zero_capacity_varshape_cache_keys_do_not_alias():
    result = subprocess.run(
        [sys.executable, "-c", _WARP_AFFINE_ZERO_CAPACITY_CACHE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"WarpAffine cache key types aliased (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )
