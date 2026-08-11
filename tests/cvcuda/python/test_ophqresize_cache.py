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

"""Regression tests for HQResize Python workspace-requirements caching."""

import subprocess
import sys
import textwrap

import pytest


_SCRIPT_PRELUDE = textwrap.dedent(
    """
    import gc

    import cupy
    import numpy

    import cvcuda


    def make_tensor(shape, value, layout="HWC"):
        array = cupy.full(shape, value, dtype=numpy.uint8)
        return array, cvcuda.as_tensor(array, layout)


    def make_planar_tensor(hw, values):
        array = cupy.empty((len(values), *hw), dtype=numpy.uint8)
        for channel, value in enumerate(values):
            array[channel].fill(value)
        return array, cvcuda.as_tensor(array, "CHW")


    def make_batch(capacity, samples, layout="HWC"):
        batch = cvcuda.TensorBatch(capacity)
        arrays = []
        for shape, value in samples:
            array, tensor = make_tensor(shape, value, layout)
            arrays.append(array)
            batch.pushback(tensor)
        return batch, arrays


    def make_planar_batch(capacity, hw, samples):
        batch = cvcuda.TensorBatch(capacity)
        arrays = []
        for values in samples:
            array, tensor = make_planar_tensor(hw, values)
            arrays.append(array)
            batch.pushback(tensor)
        return batch, arrays


    def run_resize(dst, src, stream, **kwargs):
        out = cvcuda.hq_resize_into(dst, src, stream=stream, **kwargs)
        stream.sync()
        assert out is dst


    def run_cubic(dst, src, stream, roi=None):
        kwargs = dict(interpolation=cvcuda.Interp.CUBIC, antialias=True)
        if roi is not None:
            kwargs["roi"] = roi
        run_resize(dst, src, stream, **kwargs)


    def assert_uniform(array, expected):
        assert bool(cupy.all(array == expected))


    cvcuda.clear_cache()
    """
)

_SCRIPT_EPILOGUE = textwrap.dedent(
    """
    cvcuda.internal.syncAuxStream()
    print("PASS", flush=True)
    """
)


def _run_isolated(script, **parameters):
    # WorkspaceCache is process-wide and is not cleared by cvcuda.clear_cache().
    # A fresh process ensures that an unrelated larger cached workspace cannot
    # hide an undersized stale requirements entry.
    assignments = "\n".join(f"{name} = {value!r}" for name, value in parameters.items())
    source = "\n".join(
        (_SCRIPT_PRELUDE, assignments, textwrap.dedent(script), _SCRIPT_EPILOGUE)
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"isolated HQResize cache regression failed (returncode={result.returncode})\n"
        f"stdout:\n{result.stdout[:2000]}\n"
        f"stderr:\n{result.stderr[:4000]}"
    )


@pytest.mark.parametrize("changed_batch", ["input_shape", "output_shape"])
def test_hqresize_tensorbatch_cache_invalidates_mutated_shapes(changed_batch):
    _run_isolated(
        """
        prime_src_size = 8 if changed_batch == "input_shape" else 256
        final_dst_size = 4 if changed_batch == "input_shape" else 128

        prime_src_array, prime_src = make_tensor(
            (prime_src_size, prime_src_size, 1), 37
        )
        prime_dst_array, prime_dst = make_tensor((4, 4, 1), 0)
        if changed_batch == "input_shape":
            final_src_array, final_src = make_tensor((256, 256, 1), 91)
            final_dst_array, final_dst = prime_dst_array, prime_dst
            expected_value = 91
        else:
            final_src_array, final_src = prime_src_array, prime_src
            final_dst_array, final_dst = make_tensor((128, 128, 1), 0)
            expected_value = 37

        src = cvcuda.TensorBatch(1)
        dst = cvcuda.TensorBatch(1)
        src.pushback(prime_src)
        dst.pushback(prime_dst)
        stream = cvcuda.Stream()

        run_cubic(dst, src, stream)
        assert_uniform(prime_dst_array, 37)

        # Preserve both TensorBatch identities while replacing exactly one live
        # shape. The larger contraction needs more intermediate workspace.
        if changed_batch == "input_shape":
            src[0] = final_src
        else:
            dst[0] = final_dst

        run_cubic(dst, src, stream)
        assert_uniform(final_dst_array, expected_value)

        # Replacing tensors without changing their signature should take the
        # requirements-cache hit path while still using only the live tensor
        # contents and handles.
        replacement_src_array, replacement_src = make_tensor((256, 256, 1), 123)
        replacement_dst_array, replacement_dst = make_tensor(
            (final_dst_size, final_dst_size, 1), 0
        )
        src[0] = replacement_src
        dst[0] = replacement_dst

        del prime_src_array, prime_src, prime_dst_array, prime_dst
        del final_src_array, final_src, final_dst_array, final_dst
        gc.collect()

        run_cubic(dst, src, stream)
        assert_uniform(replacement_dst_array, 123)
        """,
        changed_batch=changed_batch,
    )


def test_hqresize_tensorbatch_cache_invalidates_batch_length_growth():
    _run_isolated(
        """
        src, src_arrays = make_batch(2, [((8, 8, 1), 17)])
        dst, dst_arrays = make_batch(2, [((4, 4, 1), 0)])
        stream = cvcuda.Stream()

        run_cubic(dst, src, stream)
        assert_uniform(dst_arrays[0], 17)

        # Keep both batch identities while growing their live lengths. The
        # added contraction requires more workspace than the priming call.
        src_array, src_tensor = make_tensor((256, 256, 1), 93)
        dst_array, dst_tensor = make_tensor((4, 4, 1), 0)
        src.pushback(src_tensor)
        dst.pushback(dst_tensor)
        src_arrays.append(src_array)
        dst_arrays.append(dst_array)
        dst_arrays[0].fill(0)

        run_cubic(dst, src, stream)
        for output, expected in zip(dst_arrays, (17, 93)):
            assert_uniform(output, expected)
        """
    )


def test_hqresize_tensorbatch_cache_invalidates_ordered_heterogeneous_shapes():
    _run_isolated(
        """
        # The final call keeps the same input-shape multiset but reverses its
        # order relative to these heterogeneous output shapes.
        src, prime_src = make_batch(
            2, [((8, 8, 1), 19), ((256, 256, 1), 73)]
        )
        dst, outputs = make_batch(2, [((4, 4, 1), 0), ((128, 128, 1), 0)])
        stream = cvcuda.Stream()

        run_cubic(dst, src, stream)
        for output, expected in zip(outputs, (19, 73)):
            assert_uniform(output, expected)

        final_src = []
        for index, (shape, value) in enumerate(
            (((256, 256, 1), 101), ((8, 8, 1), 149))
        ):
            array, tensor = make_tensor(shape, value)
            final_src.append(array)
            src[index] = tensor
        for output in outputs:
            output.fill(0)

        run_cubic(dst, src, stream)
        for output, expected in zip(outputs, (101, 149)):
            assert_uniform(output, expected)
        """
    )


def test_hqresize_planar_tensorbatch_cache_invalidates_expanded_sample_count():
    _run_isolated(
        """
        src = cvcuda.TensorBatch(1)
        dst = cvcuda.TensorBatch(1)
        prime_src_array, prime_src = make_planar_tensor((256, 256), (13,))
        prime_dst_array, prime_dst = make_planar_tensor((4, 4), (0,))
        src.pushback(prime_src)
        dst.pushback(prime_dst)
        stream = cvcuda.Stream()

        run_cubic(dst, src, stream)
        assert_uniform(prime_dst_array, 13)

        # Only C changes. Planar expansion therefore grows from one workspace
        # sample to three while both TensorBatch lengths remain unchanged.
        expected_planes = (31, 67, 109)
        final_src_array, final_src = make_planar_tensor(
            (256, 256), expected_planes
        )
        final_dst_array, final_dst = make_planar_tensor((4, 4), (0, 0, 0))
        src[0] = final_src
        dst[0] = final_dst

        run_cubic(dst, src, stream)
        expected = cupy.asarray(expected_planes, dtype=numpy.uint8)[:, None, None]
        assert_uniform(final_dst_array, expected)
        """
    )


@pytest.mark.parametrize(
    "changed_parameter",
    ["roi", "min_interpolation", "mag_interpolation", "antialias"],
)
def test_hqresize_planar_tensorbatch_cache_invalidates_call_parameters(
    changed_parameter,
):
    _run_isolated(
        """
        if changed_parameter == "mag_interpolation":
            src_hw = (256, 64)
            dst_hw = (1024, 1024)
        elif changed_parameter in ("min_interpolation", "antialias"):
            src_hw = (1024, 1024)
            dst_hw = (256, 64)
        else:
            src_hw = (256, 256)
            dst_hw = (8, 8)

        expected_planes = ((11, 23, 47), (71, 89, 107))
        src, src_arrays = make_planar_batch(2, src_hw, expected_planes)
        dst, outputs = make_planar_batch(2, dst_hw, ((0, 0, 0),) * 2)
        stream = cvcuda.Stream()

        # Change exactly one requirements input. A single ROI is intentionally
        # broadcast because planar TensorBatch execution expands N tensors to
        # N*C planes internally. The asymmetric shapes force filter support to
        # change the intermediate processing order and workspace size.
        full_roi = [(0, 0, *src_hw)]
        prime = dict(
            min_interpolation=cvcuda.Interp.LANCZOS,
            mag_interpolation=cvcuda.Interp.LANCZOS,
            antialias=True,
            roi=full_roi,
        )
        final = dict(prime)

        if changed_parameter == "roi":
            prime["roi"] = [(120, 120, 136, 136)]
            final["roi"] = [(16, 16, 240, 240)]
        elif changed_parameter == "min_interpolation":
            prime["min_interpolation"] = cvcuda.Interp.NEAREST
            final["min_interpolation"] = cvcuda.Interp.LANCZOS
        elif changed_parameter == "mag_interpolation":
            prime["mag_interpolation"] = cvcuda.Interp.NEAREST
            final["mag_interpolation"] = cvcuda.Interp.LANCZOS
            prime["antialias"] = final["antialias"] = False
        elif changed_parameter == "antialias":
            prime["antialias"] = False
            final["antialias"] = True

        run_resize(dst, src, stream, **prime)
        run_resize(dst, src, stream, **final)

        for dst_array, plane_values in zip(outputs, expected_planes):
            expected = cupy.asarray(plane_values, dtype=numpy.uint8)[:, None, None]
            assert_uniform(dst_array, expected)
        """,
        changed_parameter=changed_parameter,
    )


@pytest.mark.parametrize("roi_change", ["count", "order"])
def test_hqresize_tensorbatch_cache_invalidates_roi_structure(roi_change):
    _run_isolated(
        """
        small_roi = (120, 120, 136, 136)
        large_roi = (16, 16, 240, 240)
        prime_rois = [small_roi] if roi_change == "count" else [small_roi, large_roi]
        final_rois = [large_roi, small_roi]

        src, src_arrays = make_batch(
            2, [((256, 256, 1), 29), ((256, 256, 1), 83)]
        )
        dst, outputs = make_batch(2, [((8, 8, 1), 0), ((128, 128, 1), 0)])
        stream = cvcuda.Stream()
        options = dict(
            min_interpolation=cvcuda.Interp.LANCZOS,
            mag_interpolation=cvcuda.Interp.LANCZOS,
            antialias=True,
        )

        run_resize(dst, src, stream, roi=prime_rois, **options)
        for output, expected in zip(outputs, (29, 83)):
            assert_uniform(output, expected)

        # Keep shapes and batch identities fixed. The count case changes a
        # broadcast ROI to per-sample ROIs; the order case changes only which
        # ROI is paired with each heterogeneous output shape.
        for output in outputs:
            output.fill(0)
        run_resize(dst, src, stream, roi=final_rois, **options)
        for output, expected in zip(outputs, (29, 83)):
            assert_uniform(output, expected)
        """,
        roi_change=roi_change,
    )


def test_hqresize_3d_tensorbatch_cache_hit_and_invalidation():
    _run_isolated(
        """
        src = cvcuda.TensorBatch(1)
        dst = cvcuda.TensorBatch(1)
        prime_src_array, prime_src = make_tensor((8, 8, 8, 1), 17, "DHWC")
        prime_dst_array, prime_dst = make_tensor((4, 4, 4, 1), 0, "DHWC")
        src.pushback(prime_src)
        dst.pushback(prime_dst)
        stream = cvcuda.Stream()
        small_roi = [(0, 0, 0, 8, 8, 8)]

        run_cubic(dst, src, stream, small_roi)
        assert_uniform(prime_dst_array, 17)

        # Same complete signature, but new handles and contents: this is the
        # 3D cache-hit path and must not retain live tensors from the prime.
        hit_src_array, hit_src = make_tensor((8, 8, 8, 1), 43, "DHWC")
        hit_dst_array, hit_dst = make_tensor((4, 4, 4, 1), 0, "DHWC")
        src[0] = hit_src
        dst[0] = hit_dst
        run_cubic(dst, src, stream, small_roi)
        assert_uniform(hit_dst_array, 43)

        # Change the 3D input/output shapes while preserving the six-component
        # ROI. The larger output also makes stale prime requirements unsafe.
        shape_src_array, shape_src = make_tensor((64, 64, 64, 1), 71, "DHWC")
        shape_dst_array, shape_dst = make_tensor((32, 32, 32, 1), 0, "DHWC")
        src[0] = shape_src
        dst[0] = shape_dst
        run_cubic(dst, src, stream, small_roi)
        assert_uniform(shape_dst_array, 71)

        # Keep the 3D shapes fixed and enlarge only the ROI. This also grows
        # the required contraction workspace substantially.
        large_roi = [(0, 0, 0, 64, 64, 64)]
        shape_dst_array.fill(0)
        run_cubic(dst, src, stream, large_roi)
        assert_uniform(shape_dst_array, 71)
        """
    )
