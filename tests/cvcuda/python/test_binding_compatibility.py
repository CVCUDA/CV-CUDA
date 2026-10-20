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

import subprocess
import sys
import textwrap
import weakref

import cupy
import numpy as np
import pytest

import cvcuda


class TruncOnly:
    def __trunc__(self):
        return 7


class IndexOnly:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class IntOnly:
    def __init__(self, value):
        self.value = value

    def __int__(self):
        return self.value


def _run_binding_repro(source):
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_binding_repro_passes(source):
    result = _run_binding_repro(source)
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"binding subprocess failed (returncode={result.returncode})\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


@pytest.mark.parametrize(
    ("type_name", "probe_name"),
    [
        ("Tensor", "facade_tensor_caster_probe"),
        ("Image", "facade_image_caster_probe"),
        ("Array", "facade_array_caster_probe"),
        ("TensorBatch", "facade_tensor_batch_caster_probe"),
        ("ImageBatchVarShape", "facade_image_batch_caster_probe"),
        ("Stream", "facade_stream_caster_probe"),
    ],
)
def test_facade_casters_reject_same_named_python_classes(type_name, probe_name):
    source = textwrap.dedent(
        f"""\
        import cvcuda

        hostile_type = type({type_name!r}, (), {{}})
        probe = getattr(cvcuda._test, {probe_name!r})

        try:
            probe(hostile_type())
        except TypeError as exc:
            assert "incompatible function arguments" in str(exc), str(exc)
        else:
            raise AssertionError("same-named Python class was accepted")

        print("PASS", flush=True)
        """
    )
    _assert_binding_repro_passes(source)


@pytest.mark.parametrize(
    ("probe_name", "expected_name", "is_singleton"),
    [
        ("datatype_caster_return_probe", "Type.U8", True),
        ("image_format_caster_return_probe", "Format.RGB8", False),
    ],
)
def test_public_value_caster_returns_owned_reference(
    probe_name, expected_name, is_singleton
):
    source = textwrap.dedent(
        f"""\
        import sys
        import cvcuda

        probe = getattr(cvcuda._test, {probe_name!r})
        owner_name, member_name = {expected_name!r}.split(".")
        expected = getattr(getattr(cvcuda, owner_name), member_name)
        before = sys.getrefcount(expected)
        control = expected
        owned_reference_delta = sys.getrefcount(expected) - before
        del control
        assert sys.getrefcount(expected) == before
        value = probe()
        if {is_singleton!r}:
            assert value is expected
            assert sys.getrefcount(expected) == before + owned_reference_delta
            del value
            assert sys.getrefcount(expected) == before
        else:
            assert value == expected
            # A freshly-created Format has one owner in this frame plus the
            # temporary reference held by getrefcount(). A borrowed C-API
            # result is already dangling before this point.
            assert sys.getrefcount(value) == 2

        print("PASS", flush=True)
        """
    )
    _assert_binding_repro_passes(source)


@pytest.mark.parametrize(
    "object_factory",
    [
        pytest.param(lambda: cvcuda.Format.U8, id="FormatMember"),
        pytest.param(lambda: cvcuda.Image((2, 2), cvcuda.Format.U8), id="Image"),
        pytest.param(lambda: cvcuda.ImageBatchVarShape(1), id="ImageBatchVarShape"),
        pytest.param(cvcuda.Stream, id="Stream"),
        pytest.param(
            lambda: cvcuda.Tensor((1, 2, 2, 1), cvcuda.Type.U8, "NHWC"),
            id="Tensor",
        ),
        pytest.param(lambda: cvcuda.TensorLayout("NHWC"), id="TensorLayout"),
    ],
)
def test_public_binding_objects_support_weakrefs(object_factory):
    value = object_factory()
    reference = weakref.ref(value)

    assert reference() is value


def test_type_member_weakref_contract_is_unchanged():
    with pytest.raises(TypeError):
        weakref.ref(cvcuda.Type.U8)


def test_as_image_preserves_empty_sequence_default():
    with pytest.raises(ValueError, match="Number of planes must be >= 1"):
        cvcuda.as_image()


@pytest.mark.parametrize("function", [cvcuda.as_image, cvcuda.as_tensor])
def test_external_buffer_entry_points_reject_unsupported_objects(function):
    with pytest.raises(TypeError, match="incompatible function arguments"):
        function(object())


def test_external_buffer_sequences_reject_none_without_crashing():
    source = textwrap.dedent(
        """\
        import cvcuda

        for function in (cvcuda.as_tensors, cvcuda.as_images):
            try:
                function([None])
            except RuntimeError as exc:
                assert "doesn't provide cuda_array_interface or DLPack interfaces" in str(exc)
            else:
                raise AssertionError(f"{function.__name__} unexpectedly accepted None")

        print("PASS", flush=True)
        """
    )
    _assert_binding_repro_passes(source)


def test_external_buffer_sequence_errors_remain_recoverable():
    source = textwrap.dedent(
        """\
        import cupy
        import cvcuda

        class NullCudaBuffer:
            __cuda_array_interface__ = {
                "shape": (1,),
                "typestr": "|u1",
                "data": (0, False),
                "version": 3,
            }

        for function in (cvcuda.as_tensors, cvcuda.as_images):
            try:
                function([NullCudaBuffer()])
            except RuntimeError as exc:
                assert "NULL CUDA buffer not accepted" in str(exc)
            else:
                raise AssertionError(f"{function.__name__} accepted a NULL CUDA buffer")

            valid = function([cupy.zeros((2, 2), dtype=cupy.uint8)])
            assert len(valid) == 1

        print("PASS", flush=True)
        """
    )
    _assert_binding_repro_passes(source)


def test_stack_tensor_lists_reject_none_without_crashing():
    source = textwrap.dedent(
        """\
        import cvcuda

        dst = cvcuda.Tensor((1, 2, 2, 1), cvcuda.Type.U8, "NHWC")
        for call in (lambda: cvcuda.stack([None]), lambda: cvcuda.stack_into(dst, [None])):
            try:
                call()
            except TypeError:
                pass
            else:
                raise AssertionError("stack tensor list unexpectedly accepted None")

        print("PASS", flush=True)
        """
    )
    _assert_binding_repro_passes(source)


@pytest.mark.parametrize("operator_name", ["warp_affine", "warp_perspective"])
def test_none_transform_raises_instead_of_crashing(operator_name):
    source = textwrap.dedent(
        f"""\
        import cvcuda

        src = cvcuda.Tensor((1, 2, 2, 1), cvcuda.Type.U8, "NHWC")
        operator = cvcuda.{operator_name}
        try:
            operator(
                src,
                None,
                cvcuda.Interp.NEAREST,
                border_mode=cvcuda.Border.CONSTANT,
                border_value=0,
            )
        except IndexError:
            pass
        else:
            raise AssertionError("None transform unexpectedly accepted")

        print("PASS", flush=True)
        """
    )
    _assert_binding_repro_passes(source)


@pytest.mark.parametrize(
    "operator_name,xform_width",
    [("warp_affine", 6), ("warp_perspective", 9)],
)
def test_none_varshape_border_preserves_scalar_shape(operator_name, xform_width):
    source = textwrap.dedent(
        f"""\
        import cvcuda

        src = cvcuda.ImageBatchVarShape(1)
        src.pushback(cvcuda.Image((2, 2), cvcuda.Format.RGB8))
        xform = cvcuda.Tensor((1, {xform_width}), cvcuda.Type.F32, "NC")
        operator = cvcuda.{operator_name}
        try:
            operator(
                src,
                xform,
                cvcuda.Interp.NEAREST,
                border_mode=cvcuda.Border.CONSTANT,
                border_value=None,
            )
        except ValueError as exc:
            assert "current is '1', '0' respectively" in str(exc), str(exc)
        else:
            raise AssertionError("scalar None border unexpectedly accepted")

        print("PASS", flush=True)
        """
    )
    _assert_binding_repro_passes(source)


@pytest.mark.parametrize(
    "operator,xform",
    [
        pytest.param(
            cvcuda.warp_affine,
            np.array([[1, 0, 100], [0, 1, 100]], dtype=np.float32),
            id="warp-affine",
        ),
        pytest.param(
            cvcuda.warp_perspective,
            np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]], dtype=np.float32),
            id="warp-perspective",
        ),
    ],
)
def test_none_border_matches_scalar_nan(operator, xform):
    source = cupy.zeros((1, 2, 2, 1), dtype=cupy.float32)
    src = cvcuda.as_tensor(source, "NHWC")
    kwargs = {
        "border_mode": cvcuda.Border.CONSTANT,
        "flags": cvcuda.Interp.NEAREST,
    }

    none_output = operator(src, xform, border_value=None, **kwargs)
    nan_output = operator(
        src, xform, border_value=np.array(np.nan, dtype=np.float32), **kwargs
    )

    none_result = cupy.asarray(none_output.cuda()).get()
    nan_result = cupy.asarray(nan_output.cuda()).get()
    assert np.isnan(none_result).all()
    np.testing.assert_allclose(none_result, nan_result, equal_nan=True)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("7", id="str"),
        pytest.param(b"7", id="bytes"),
        pytest.param(bytearray(b"7"), id="bytearray"),
        pytest.param(memoryview(b"7"), id="memoryview"),
        pytest.param(TruncOnly(), id="trunc-only"),
        pytest.param(7.0, id="float"),
    ],
)
@pytest.mark.parametrize("function", [cvcuda.as_stream, cvcuda.set_cache_limit_inbytes])
def test_integral_entry_points_reject_incompatible_conversions(function, value):
    with pytest.raises(TypeError, match="incompatible function arguments"):
        function(value)


@pytest.mark.parametrize("value_type", [IndexOnly, IntOnly])
def test_integral_entry_points_accept_numeric_protocols(value_type):
    stream = cvcuda.as_stream(value_type(1))
    assert stream.handle == 1

    cache_limit = cvcuda.get_cache_limit_inbytes()
    cvcuda.set_cache_limit_inbytes(value_type(cache_limit))
    assert cvcuda.get_cache_limit_inbytes() == cache_limit
