# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy
import numpy as np
import cvcuda

import pytest as t

NUM_GPUS = cupy.cuda.runtime.getDeviceCount()

requires_multi_gpu = t.mark.skipif(
    NUM_GPUS < 2,
    reason="Multi-GPU tests require at least 2 GPUs",
)


@t.fixture(autouse=True)
def _restore_default_device():
    """Restore CUDA device 0 after each test so later tests are not polluted."""
    yield
    cupy.cuda.Device(0).use()


# ---------------------------------------------------------------------------
# Single-GPU smoke test (always runs)
# ---------------------------------------------------------------------------


def test_dlpack_device_id_on_default_gpu():
    """Verify DLPack export reports the correct device, data, and zero-copy semantics."""
    cupy.cuda.Device(0).use()

    src = cupy.full((1, 4, 4, 3), fill_value=200, dtype=cupy.uint8)
    src_nv = cvcuda.as_tensor(src, "NHWC")

    stream = cvcuda.Stream()
    with stream:
        out_nv = cvcuda.cvtcolor(src_nv, cvcuda.ColorConversion.BGR2GRAY, stream=stream)
    stream.sync()

    cuda_buf = out_nv.cuda()
    result = cupy.asarray(cuda_buf)

    assert result.device.id == 0
    assert result.shape == (1, 4, 4, 1)

    assert (
        result == 200
    ).all(), f"Expected all 200, got {cupy.unique(result).tolist()}"

    # Verify zero-copy: the cupy array shares the same device pointer.
    cai = cuda_buf.__cuda_array_interface__
    assert result.data.ptr == cai["data"][0]


# ---------------------------------------------------------------------------
# Multi-GPU tests (skipped when fewer than 2 GPUs are available)
# ---------------------------------------------------------------------------


@requires_multi_gpu
@t.mark.parametrize("gpu_id", range(NUM_GPUS))
def test_dlpack_reports_correct_device(gpu_id):
    """DLPack export must report the actual owning device, not hardcoded 0."""
    cupy.cuda.Device(gpu_id).use()

    src = cupy.full((1, 4, 4, 3), fill_value=200, dtype=cupy.uint8)
    src_nv = cvcuda.as_tensor(src, "NHWC")

    stream = cvcuda.Stream()
    with stream:
        out_nv = cvcuda.cvtcolor(src_nv, cvcuda.ColorConversion.BGR2GRAY, stream=stream)
    stream.sync()

    result = cupy.asarray(out_nv.cuda())

    assert result.device.id == gpu_id
    assert result.shape == (1, 4, 4, 1)
    assert (
        result == 200
    ).all(), f"Expected all 200, got {cupy.unique(result).tolist()}"


def _run_cvtcolor_on_gpu(gpu_id):
    """Run the cvtcolor operator on the given GPU & return the result as a cupy array."""
    cupy.cuda.Device(gpu_id).use()
    stream = cvcuda.Stream()

    src = cupy.zeros((1, 16, 16, 3), dtype=cupy.uint8)
    src_nv = cvcuda.as_tensor(src, "NHWC")

    with stream:
        out_nv = cvcuda.cvtcolor(
            src_nv,
            cvcuda.ColorConversion.BGR2GRAY,
            stream=stream,
        )
    stream.sync()
    return cupy.asarray(out_nv.cuda())


@requires_multi_gpu
@t.mark.parametrize("gpu_id", range(NUM_GPUS))
def test_operator_on_each_gpu(gpu_id):
    result = _run_cvtcolor_on_gpu(gpu_id)
    assert result.device.id == gpu_id
    assert result.shape == (1, 16, 16, 1)


@requires_multi_gpu
def test_operator_across_gpus_sequentially():
    for gpu_id in range(NUM_GPUS):
        result = _run_cvtcolor_on_gpu(gpu_id)
        assert result.device.id == gpu_id
        assert result.shape == (1, 16, 16, 1)


@requires_multi_gpu
def test_resource_reuse_across_gpus():
    cupy.cuda.Device(0).use()
    stream0 = cvcuda.Stream()

    rng = np.random.default_rng(0)
    src = cupy.asarray(rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8))
    src_nv = cvcuda.as_tensor(src, "NHWC")

    with stream0:
        intermediate = cvcuda.cvtcolor(
            src_nv, cvcuda.ColorConversion.BGR2GRAY, stream=stream0
        )
    stream0.sync()

    cupy.cuda.Device(1).use()
    stream1 = cvcuda.Stream()

    gray_on_0 = cupy.asarray(intermediate.cuda())
    gray_on_1 = cupy.array(gray_on_0)
    gray_rgb = cupy.concatenate([gray_on_1] * 3, axis=-1)
    gray_nv = cvcuda.as_tensor(gray_rgb, "NHWC")

    with stream1:
        result = cvcuda.cvtcolor(
            gray_nv, cvcuda.ColorConversion.BGR2GRAY, stream=stream1
        )
    stream1.sync()

    result_cupy = cupy.asarray(result.cuda())
    assert result_cupy.device.id == 1
    assert result_cupy.shape == (1, 32, 32, 1)


@requires_multi_gpu
def test_cache_independent_across_gpus():
    cvcuda.clear_cache()

    results = {}
    for gpu_id in range(NUM_GPUS):
        cupy.cuda.Device(gpu_id).use()
        stream = cvcuda.Stream()

        with cupy.cuda.Device(gpu_id):
            src = cupy.full((1, 8, 8, 3), fill_value=100, dtype=cupy.uint8)
        src_nv = cvcuda.as_tensor(src, "NHWC")

        with stream:
            out_nv = cvcuda.cvtcolor(
                src_nv, cvcuda.ColorConversion.BGR2GRAY, stream=stream
            )
        stream.sync()

        result_cupy = cupy.asarray(out_nv.cuda())
        assert result_cupy.device.id == gpu_id
        results[gpu_id] = result_cupy.get()

    for i in range(1, NUM_GPUS):
        np.testing.assert_array_equal(results[0], results[i])


# ---------------------------------------------------------------------------
# Resource creation on non-default device
# ---------------------------------------------------------------------------


def _image_device(img):
    """Return the CUDA device ordinal that an Image's buffer lives on."""
    return cupy.asarray(img.cuda()).device.id


@requires_multi_gpu
def test_image_zeros_on_non_default_device():
    """Image.zeros() must allocate on the current device, not device 0."""
    cvcuda.clear_cache()
    cupy.cuda.Device(1).use()

    img = cvcuda.Image.zeros((32, 32), cvcuda.Format.RGB8)
    assert _image_device(img) == 1


@requires_multi_gpu
def test_image_create_on_non_default_device():
    """Image() (uninitialized) must allocate on the current device."""
    cvcuda.clear_cache()
    cupy.cuda.Device(1).use()

    img = cvcuda.Image((32, 32), cvcuda.Format.RGB8)
    assert _image_device(img) == 1


@requires_multi_gpu
def test_tensor_create_on_non_default_device():
    """Tensor() must allocate on the current device."""
    cvcuda.clear_cache()
    cupy.cuda.Device(1).use()

    tensor = cvcuda.Tensor((1, 16, 16, 3), dtype=cvcuda.Type.U8, layout="NHWC")
    result = cupy.asarray(tensor.cuda())
    assert result.device.id == 1


# ---------------------------------------------------------------------------
# Cross-device cache isolation
# ---------------------------------------------------------------------------


@requires_multi_gpu
def test_cache_no_cross_device_image():
    """Cached Image on device 0 must not be returned when creating on device 1."""
    cvcuda.clear_cache()
    size = (64, 64)
    fmt = cvcuda.Format.RGB8

    cupy.cuda.Device(0).use()
    img0 = cvcuda.Image.zeros(size, fmt)
    ptr0 = cupy.asarray(img0.cuda()).data.ptr
    assert _image_device(img0) == 0
    del img0

    cupy.cuda.Device(1).use()
    img1 = cvcuda.Image.zeros(size, fmt)
    ptr1 = cupy.asarray(img1.cuda()).data.ptr
    assert _image_device(img1) == 1
    assert ptr1 != ptr0


@requires_multi_gpu
def test_cache_no_cross_device_tensor():
    """Cached Tensor on device 0 must not be returned when creating on device 1."""
    cvcuda.clear_cache()
    shape = (1, 32, 32, 3)

    cupy.cuda.Device(0).use()
    t0 = cvcuda.Tensor(shape, dtype=cvcuda.Type.U8, layout="NHWC")
    ptr0 = cupy.asarray(t0.cuda()).data.ptr
    del t0

    cupy.cuda.Device(1).use()
    t1 = cvcuda.Tensor(shape, dtype=cvcuda.Type.U8, layout="NHWC")
    ptr1 = cupy.asarray(t1.cuda()).data.ptr
    assert cupy.asarray(t1.cuda()).device.id == 1
    assert ptr1 != ptr0


# ---------------------------------------------------------------------------
# Operator device safety — operators with persistent GPU buffers must work
# on non-default devices without crashing (cudaErrorIllegalAddress).
# Each test runs an operator on GPU 0, then on GPU 1, verifying that the
# per-device internal state is allocated correctly.
# ---------------------------------------------------------------------------


def _run_op_on_each_gpu(op_fn):
    """Run op_fn(gpu_id) on GPU 0 then GPU 1 and return both results."""
    results = {}
    for gpu_id in range(min(NUM_GPUS, 2)):
        cupy.cuda.Device(gpu_id).use()
        results[gpu_id] = op_fn(gpu_id)
    return results


@requires_multi_gpu
def test_gaussian_on_non_default_device():
    """Gaussian filter must work across GPUs (has persistent m_kernel buffer)."""

    def run(gpu_id):
        with cupy.cuda.Device(gpu_id):
            rng = np.random.default_rng(0)
            src = cupy.asarray(rng.integers(0, 256, (1, 16, 16, 1), dtype=np.uint8))
        src_nv = cvcuda.as_tensor(src, "NHWC")
        out_nv = cvcuda.gaussian(src_nv, (3, 3), (1.0, 1.0))
        result = cupy.asarray(out_nv.cuda())
        assert result.device.id == gpu_id
        return result.shape

    results = _run_op_on_each_gpu(run)
    assert results[0] == results[1]


@requires_multi_gpu
def test_rotate_on_non_default_device():
    """Rotate must work across GPUs (has persistent d_aCoeffs buffer)."""

    def run(gpu_id):
        with cupy.cuda.Device(gpu_id):
            rng = np.random.default_rng(0)
            src = cupy.asarray(rng.integers(0, 256, (1, 16, 16, 3), dtype=np.uint8))
        src_nv = cvcuda.as_tensor(src, "NHWC")
        out_nv = cvcuda.rotate(src_nv, 45.0, [0, 0], cvcuda.Interp.NEAREST)
        result = cupy.asarray(out_nv.cuda())
        assert result.device.id == gpu_id
        return result.shape

    results = _run_op_on_each_gpu(run)
    assert results[0] == results[1]


@requires_multi_gpu
def test_inpaint_on_non_default_device():
    """Inpaint must work across GPUs (has persistent m_kernel_ptr, m_workspace)."""

    def run(gpu_id):
        with cupy.cuda.Device(gpu_id):
            rng = np.random.default_rng(0)
            src = cupy.asarray(rng.integers(0, 256, (1, 16, 16, 3), dtype=np.uint8))
            mask = cupy.zeros((1, 16, 16, 1), dtype=cupy.uint8)
        src_nv = cvcuda.as_tensor(src, "NHWC")
        mask_nv = cvcuda.as_tensor(mask, "NHWC")
        out_nv = cvcuda.inpaint(src_nv, mask_nv, 3.0)
        result = cupy.asarray(out_nv.cuda())
        assert result.device.id == gpu_id
        return result.shape

    results = _run_op_on_each_gpu(run)
    assert results[0] == results[1]


@requires_multi_gpu
def test_hqresize_on_non_default_device():
    """HQ Resize must work across GPUs (has persistent filter coefficients)."""

    def run(gpu_id):
        with cupy.cuda.Device(gpu_id):
            src = cupy.arange(37 * 41, dtype=cupy.float32).reshape(1, 37, 41, 1)
            src_nv = cvcuda.as_tensor(src, "NHWC")
            stream = cvcuda.Stream()
            with stream:
                out_nv = cvcuda.hq_resize(
                    src_nv,
                    (74, 82),
                    interpolation=cvcuda.Interp.CUBIC,
                    stream=stream,
                )
            stream.sync()
            result = cupy.asarray(out_nv.cuda())
            assert result.device.id == gpu_id
            assert bool(cupy.isfinite(result).all())
            return result.shape

    results = _run_op_on_each_gpu(run)
    assert results[0] == results[1]


@requires_multi_gpu
def test_bndbox_on_non_default_device():
    """BndBox/OSD must work across GPUs (has persistent Memory<T> buffers)."""

    def run(gpu_id):
        with cupy.cuda.Device(gpu_id):
            rng = np.random.default_rng(0)
            src = cupy.asarray(rng.integers(0, 256, (1, 64, 64, 4), dtype=np.uint8))
        src_nv = cvcuda.as_tensor(src, "NHWC")
        bboxes = cvcuda.BndBoxesI(
            boxes=[
                [
                    cvcuda.BndBoxI(
                        box=(5, 5, 20, 20),
                        thickness=1,
                        borderColor=(0, 255, 0, 255),
                        fillColor=(0, 128, 0, 128),
                    ),
                ],
            ],
        )
        out_nv = cvcuda.bndbox(src_nv, bboxes)
        result = cupy.asarray(out_nv.cuda())
        assert result.device.id == gpu_id
        return result.shape

    results = _run_op_on_each_gpu(run)
    assert results[0] == results[1]
