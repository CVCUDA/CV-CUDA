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

"""Shared machinery for cvcuda.tv's Torch tensor parity tests."""

import enum
import gc
import inspect
import itertools

import numpy as np
import pytest

import cvcuda


def image_params(
    dtypes,
    channels,
    *,
    sizes=((17, 11),),
    batches=(None, 1, 3),
    marks=(),
):
    """Return deterministic planar image specifications for pytest."""
    rows = []
    for dtype, channel, (height, width), batch in itertools.product(
        dtypes, channels, sizes, batches
    ):
        dtype_marks = ()
        if not isinstance(dtype, str):
            dtype_marks = dtype.marks
            (dtype,) = dtype.values
        rows.append(
            pytest.param(
                (batch, channel, height, width, dtype),
                id=(
                    f"{'chw' if batch is None else f'n{batch}chw'}-c{channel}-"
                    f"{dtype}-{height}x{width}"
                ),
                marks=(
                    (*marks, *dtype_marks)
                    if isinstance(marks, (tuple, list))
                    else (marks, *dtype_marks)
                ),
            )
        )
    return rows


def images(*, dtypes, channels, sizes=((17, 11),), batches=(None, 1, 3)):
    """Declare the image matrix while leaving argument scenarios on the test."""
    return pytest.mark.parametrize(
        "image",
        image_params(dtypes, channels, sizes=sizes, batches=batches),
        indirect=True,
    )


@pytest.fixture(scope="session")
def torch():
    torch_module = pytest.importorskip("torch")
    if not torch_module.cuda.is_available():
        pytest.skip("cvcuda.tv parity needs a CUDA-enabled torch build")
    yield torch_module
    torch_module.cuda.synchronize()
    gc.collect()
    cvcuda.clear_cache()
    gc.collect()
    torch_module.cuda.synchronize()


@pytest.fixture(scope="session")
def tvf():
    return pytest.importorskip("torchvision.transforms.v2.functional")


@pytest.fixture
def image(request, torch):
    """Create the requested contiguous CUDA image in CHW or NCHW order.

    Integer images span the full dtype range; special/test_parity_regressions.py
    additionally covers narrower ranges that expose otherwise hidden rounding.
    """
    batch, channels, height, width, dtype_name = request.param
    dtype = getattr(torch, dtype_name)
    shape = (channels, height, width)
    if batch is not None:
        shape = (batch, *shape)
    high = 1 if dtype.is_floating_point else torch.iinfo(dtype).max
    torch.manual_seed(0)
    return torch.testing.make_tensor(
        shape, low=0, high=high, dtype=dtype, device="cuda"
    ).contiguous()


def _planar_layout(ndim):
    if ndim == 3:
        return "CHW"
    if ndim == 4:
        return "NCHW"
    raise ValueError(f"cvcuda.tv parity expects a 3D or 4D image, got {ndim}D")


def layout_pair(ndim):
    planar = _planar_layout(ndim)
    return planar, planar.replace("CHW", "HWC")


def _from_planar(tensor, layout):
    planar = _planar_layout(tensor.ndim)
    return tensor.permute(*(planar.index(axis) for axis in layout)).contiguous()


def _to_planar(tensor, layout):
    planar = _planar_layout(tensor.ndim)
    return tensor.permute(*(layout.index(axis) for axis in planar))


def layout_source(torch, layout):
    """A random float image in ``layout`` and the cvcuda tensor aliasing it."""
    planar = torch.rand((2, 3, 7, 9), device="cuda")
    if "N" not in layout:
        planar = planar[0]
    owner = _from_planar(planar, layout)
    return owner, cvcuda.as_tensor(owner, layout=layout)


def as_torch(torch, output):
    tensor = torch.as_tensor(output.cuda())
    # Some Torch builds (e.g. Jetson nv24.03) copy a CAI object to the CPU.
    return tensor if tensor.is_cuda else tensor.cuda()


def to_torch(torch, output, layout):
    assert str(output.layout) == layout
    return _to_planar(as_torch(torch, output), layout)


def _normalize_output(torch, output, layout, native):
    if isinstance(output, (tuple, list)):
        return type(output)(
            _normalize_output(torch, item, layout, native) for item in output
        )
    assert isinstance(output, cvcuda.Tensor if native else torch.Tensor)
    return to_torch(torch, output, layout) if native else output


def _output_tensors(output):
    if isinstance(output, (tuple, list)):
        return [tensor for item in output for tensor in _output_tensors(item)]
    return [output]


def _assert_fresh(output, inputs):
    for tensor in _output_tensors(output):
        assert tensor.numel() > 0
        storage = tensor.untyped_storage()
        start, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        # Storage spans include offset views and stride gaps, unlike numel * itemsize.
        for source in inputs:
            storage = source.untyped_storage()
            low, high = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
            assert end <= low or high <= start, "output aliases input storage"


def _clone_arg(torch, value):
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


def _parity_arg(torch, value, cv_op, image, native, owners):
    value = _clone_arg(torch, value)
    if not isinstance(value, torch.Tensor):
        return value
    if native:
        value = value.to(image.device)
        if cv_op.__name__ == "erase":
            value = value.to(image.dtype)
        if value.ndim == 0:
            value = value.reshape(1)
        value = value.contiguous()
    owners.append(value)
    if not native:
        return value
    return cvcuda.as_tensor(
        value, layout="NHWC" if cv_op.__name__ == "elastic" else None
    )


def _parity_output(torch, cv_op, image, args, kwargs, layout, native, fresh):
    owner = _from_planar(image, layout).clone()
    argument_owners = []
    op_args = [
        _parity_arg(torch, value, cv_op, image, native, argument_owners)
        for value in args
    ]
    op_kwargs = {
        name: _parity_arg(torch, value, cv_op, image, native, argument_owners)
        for name, value in kwargs.items()
    }
    snapshots = [value.clone() for value in argument_owners] if fresh else []
    inpt = cvcuda.as_tensor(owner, layout=layout) if native else owner
    if native:
        op_kwargs["stream"] = cvcuda.as_stream(torch.cuda.current_stream().cuda_stream)
    output = _normalize_output(
        torch, cv_op(inpt, *op_args, **op_kwargs), layout, native
    )
    if fresh:
        _assert_fresh(output, [image, owner, *argument_owners])
        torch.testing.assert_close(owner, _from_planar(image, layout), atol=0, rtol=0)
        for value, snapshot in zip(argument_owners, snapshots):
            torch.testing.assert_close(value, snapshot, atol=0, rtol=0)
    return output


def _assert_oracle(torch, result, expected, atol, rtol, max_bad_frac):
    if not max_bad_frac or isinstance(result, (tuple, list)):
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)
        return
    bad_fraction = (
        (~torch.isclose(result, expected, atol=atol, rtol=rtol)).float().mean()
    )
    assert bad_fraction <= max_bad_frac


@pytest.fixture(scope="session")
def assert_tv_parity(torch, request):
    """Check Torch on planar rows and native inputs in every requested layout."""
    requested_layout = getattr(request, "param", None)
    caller = torch.cuda.Stream()

    def assert_parity(
        cv_op,
        tv_op,
        image,
        *args,
        atol=0,
        rtol=0,
        max_bad_frac=0,
        fresh=False,
        **kwargs,
    ):
        assert image.is_cuda and image.is_contiguous()
        layouts = layout_pair(image.ndim)
        planar_layout = layouts[0]
        if requested_layout is not None:
            layouts = (layouts[("planar", "interleaved").index(requested_layout)],)

        caller.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(caller):
            expected = tv_op(
                image.clone(),
                *(_clone_arg(torch, value) for value in args),
                **{name: _clone_arg(torch, value) for name, value in kwargs.items()},
            )
            outputs = []
            via_torch = None
            if planar_layout in layouts:
                via_torch = _parity_output(
                    torch, cv_op, image, args, kwargs, planar_layout, False, fresh
                )
                _assert_oracle(torch, via_torch, expected, atol, rtol, max_bad_frac)
                outputs.extend(_output_tensors(via_torch))
            for layout in layouts:
                actual = _parity_output(
                    torch, cv_op, image, args, kwargs, layout, True, fresh
                )
                if fresh:
                    _assert_fresh(actual, outputs)
                if layout == planar_layout:
                    torch.testing.assert_close(actual, via_torch, atol=0, rtol=0)
                _assert_oracle(torch, actual, expected, atol, rtol, max_bad_frac)
                outputs.extend(_output_tensors(actual))
        torch.cuda.current_stream().wait_stream(caller)

    return assert_parity


def _norm_default(value):
    if isinstance(value, enum.Enum):
        return value.value
    text = str(value)
    if text.startswith("torch."):
        return text.split(".", 1)[1]
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, type) and getattr(value, "__module__", "") == "numpy":
        return np.dtype(value).name
    return value


def param_spec(sig, exclude=()):
    return [
        (parameter.name, parameter.kind, _norm_default(parameter.default))
        for parameter in sig.parameters.values()
        if parameter.name not in exclude
    ]


def _near(value, eps, *, half):
    """Where ``value`` is within ``eps`` of a .5 (``half``) or an integer boundary."""
    frac = value - value.floor()
    if half:
        return (frac - 0.5).abs() < eps
    return (frac < eps) | (frac > 1 - eps)


def tie_exempt(op, ref, got, inpt, args=(), kwargs=None):
    """Return the mismatching pixels of ``op`` that a documented tie explains.

    ``ref`` is torchvision's output and ``got`` cvcuda.tv's for the same call
    ``F.<op>(inpt, *args, **kwargs)``. Each residual class has a float64 or
    exact-integer oracle for the quantity whose rounding decides the pixel; a
    pixel is exempt only where that quantity sits on a rounding boundary.
    """
    import torch
    from torchvision.transforms.v2 import InterpolationMode
    from torchvision.transforms.v2 import functional as F
    from torchvision.transforms.v2.functional._geometry import _check_interpolation

    function = getattr(F, op)
    bound = inspect.signature(function).bind(inpt, *args, **(kwargs or {}))
    bound.apply_defaults()
    params = dict(bound.arguments)
    first = next(iter(params))
    none = torch.zeros(ref.shape, dtype=torch.bool, device=ref.device)

    interpolation = params.get("interpolation")
    if (
        interpolation is not None
        and op in ("rotate", "affine", "perspective", "elastic")
        and _check_interpolation(interpolation) == InterpolationMode.NEAREST
    ):
        # Nearest sampling rounds the source coordinate; resampling float64
        # coordinate ramps bilinearly gives that coordinate exactly.
        height, width = inpt.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=inpt.device),
            torch.arange(width, dtype=torch.float64, device=inpt.device),
            indexing="ij",
        )
        ties = []
        for ramp in (xx, yy):
            ramp_params = dict(
                params,
                **{first: ramp[None]},
                interpolation=InterpolationMode.BILINEAR,
                fill=None,
            )
            ties.append(_near(function(**ramp_params)[0], 1e-3, half=True))
        return (ties[0] | ties[1]).expand(ref.shape)

    if inpt.is_floating_point():
        return none
    delta = got.to(torch.int64) - ref.to(torch.int64)

    if op == "autocontrast":
        # cvcuda.tv is the exact floor of (x - min) * bound / (max - min);
        # torchvision's fp32 division can land 1 lower on a whole quotient.
        bound_value = torch.iinfo(inpt.dtype).max
        image = inpt.to(torch.int64)
        low = image.amin((-2, -1), keepdim=True)
        high = image.amax((-2, -1), keepdim=True)
        num = (image - low) * bound_value
        den = torch.where(high == low, 1, high - low)
        if inpt.dtype == torch.uint8:
            return (num % den == 0) & (delta == 1)
        # Beyond fp32's mantissa both sides are the exact floor up to fp32
        # resolution of an integer quotient.
        floor = num // den
        quotient = num.double() / den.double()
        near = (quotient - quotient.round()).abs() <= 2**-21 * quotient.clamp(min=1)
        ours = (got.to(torch.int64) - floor).abs() <= 1
        theirs = (ref.to(torch.int64) - floor).abs() <= 1
        return near & ours & theirs

    if op == "adjust_hue":
        # torchvision truncates v * 255.999 after its fp32 HSV round trip.
        exact = function(inpt.double() / 255, params["hue_factor"]) * (256 - 1e-3)
        return _near(exact, 1e-3, half=False) & (delta.abs() == 1)

    # Round-to-nearest resampling and filtering (resize antialias, blur,
    # bilinear warps): the float64 result sits on a .5 boundary.
    exact = function(**dict(params, **{first: inpt.double()}))
    return _near(exact, 5e-4, half=True) & (delta.abs() == 1)
