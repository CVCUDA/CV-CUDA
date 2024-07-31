# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvcv
import gc
import numpy as np
import torch
import sys

# TODO: These tests technically belong to nvcv, but since it doesn't expose any
# operator (or anything that we could submit to a stream), we need to add this
# to cvcuda instead. Ideally nvcv should allows us to write proper stream tests
# using only the facilities it provides. Maybe a "noop" operator, or some mocks,
# maybe container copy, etc.


def test_clear_cache_inside_op():
    tensor = nvcv.Tensor((100, 1500, 1500, 3), nvcv.Type.U8, nvcv.TensorLayout.NHWC)
    map = nvcv.Tensor((100, 1500, 1500, 2), nvcv.Type.F32, nvcv.TensorLayout.NHWC)
    with cvcuda.Stream():
        out = cvcuda.remap(tensor, map)
        nvcv.clear_cache()
    del tensor
    del map
    del out
    gc.collect()


def test_gcbag_is_being_emptied():
    # Make sure there's no work scheduled on the stream, it's all ours.
    workstream = nvcv.cuda.Stream()

    # In order to test if the GCBag was really emptied,

    # we create a torch tensor,
    ttensor = torch.as_tensor(np.ndarray([100, 1500, 1500, 3], np.uint8), device="cuda")
    # keep track of its initial refcount.
    orig_ttensor_refcount = sys.getrefcount(ttensor)
    # and wrap it in a nvcv tensor 'cvwrapper'
    cvwrapper = nvcv.as_tensor(ttensor, nvcv.TensorLayout.NHWC)

    # We can then indirectly tell if 'cvwrapper' was destroyed by
    # monitoring 'ttensor's refcount.
    # This works because we know 'cvwrapper' holds a reference to
    # 'ttensor', as proved by the following assert:
    wrapped_ttensor_refcount = sys.getrefcount(ttensor)
    assert wrapped_ttensor_refcount > orig_ttensor_refcount

    # We need now to make sure cvwrapper is in the GCBag.
    # For that, we need to use it in operator
    with workstream:
        cvcuda.median_blur(cvwrapper, [3, 3], stream=workstream)
        # And make sure it finishes.
        workstream.sync()
    # Make sure the auxiliary stream has finished extending cvwrapper's lifetime
    nvcv.cuda.internal.syncAuxStream()

    # cvwrapper being referenced by others shouldn't change ttensor's refcount.
    assert sys.getrefcount(ttensor) == wrapped_ttensor_refcount

    # Now remove cvwrapper from the cache by clearing it.
    nvcv.clear_cache()

    # We can now release it from python side. We can't track its lifetime
    # directly anymore.
    del cvwrapper

    # But we know indirectly that it is still alive
    assert sys.getrefcount(ttensor) == wrapped_ttensor_refcount

    # To finally destroy cvwrapper, we empty the GCBag by executing a
    # cvcuda operator, any would do.
    with workstream:
        cvcuda.median_blur(
            nvcv.Tensor((3, 64, 32, 3), nvcv.Type.U8, nvcv.TensorLayout.NHWC), [3, 3]
        )
        workstream.sync()
    nvcv.cuda.internal.syncAuxStream()

    # Lo and behold, cvwrapper is no more.
    # The wrapped tensor torch has the same refcount it had when we've created it.
    assert sys.getrefcount(ttensor) == orig_ttensor_refcount
