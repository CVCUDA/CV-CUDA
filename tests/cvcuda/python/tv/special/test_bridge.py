# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv bridge contracts beyond native tensor parity."""

import importlib
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv
from cvcuda.tv import _torch

from conftest import as_torch


@pytest.mark.parametrize("torch_input", [False, True])
def test_jpeg_first_argument_keyword(torch, torch_input):
    image = torch.arange(3 * 8 * 8, device="cuda", dtype=torch.uint8).reshape(3, 8, 8)
    if not torch_input:
        image = cvcuda.as_tensor(image, "CHW")
    expected = cvcuda_tv.jpeg(image, quality=80)
    actual = cvcuda_tv.jpeg(image=image, quality=80)
    if not torch_input:
        actual, expected = as_torch(torch, actual), as_torch(torch, expected)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(TypeError):
        cvcuda_tv.jpeg(image, image=image, quality=80)


@pytest.mark.parametrize("dtype_name", ["uint8", "uint16", "float16", "float32"])
@pytest.mark.parametrize("channels", [1, 3])
def test_native_grayscale_without_torch(monkeypatch, dtype_name, channels):
    # A None entry makes ``import torch`` raise, so a fresh cvcuda.tv import and a
    # native call that reached for torch, cupy, or cuda-python would fail here.
    for name in ("torch", "cupy", "cuda"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(cvcuda, "tv", cvcuda.tv)
    for name in [
        m for m in sys.modules if m == "cvcuda.tv" or m.startswith("cvcuda.tv.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    fresh_tv = importlib.import_module("cvcuda.tv")
    assert fresh_tv is not cvcuda_tv

    dtype = np.dtype(dtype_name)
    integer = np.issubdtype(dtype, np.integer)
    source = np.zeros((4, 5, 3), dtype=dtype)
    source[..., 0] = 100 if integer else 1
    owner = cvcuda.Image(source)
    stream = cvcuda.Stream()
    actual = fresh_tv.rgb_to_grayscale(
        cvcuda.as_tensor(owner), num_output_channels=channels, stream=stream
    )
    stream.sync()
    actual = cvcuda.as_image(actual.cuda()).cpu()
    np.testing.assert_array_equal(
        actual, np.full_like(actual, 29 if integer else 0.2989)
    )


def _assert_bridged(torch, via_torch, via_native):
    if isinstance(via_native, (tuple, list)):
        assert isinstance(via_torch, (tuple, list))
        assert len(via_torch) == len(via_native)
        for torch_item, native_item in zip(via_torch, via_native):
            _assert_bridged(torch, torch_item, native_item)
        return
    assert type(via_native) is cvcuda.Tensor
    assert isinstance(via_torch, torch.Tensor)
    expected = as_torch(torch, via_native)
    torch.testing.assert_close(via_torch, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "name,keyword,kwargs",
    [
        pytest.param("invert", "inpt", {}, id="invert-inpt"),
        pytest.param("jpeg", "image", {"quality": 50}, id="jpeg-image"),
    ],
)
def test_torch_tensor_by_keyword(torch, name, keyword, kwargs):
    generator = torch.Generator(device="cuda").manual_seed(0)
    image = torch.randint(
        0, 256, (3, 16, 20), generator=generator, device="cuda", dtype=torch.uint8
    )
    operator = getattr(cvcuda_tv, name)
    via_native = operator(cvcuda.as_tensor(image, "CHW"), **kwargs)
    via_torch = operator(**{keyword: image}, **kwargs)
    _assert_bridged(torch, via_torch, via_native)


@pytest.mark.parametrize(
    "call",
    [
        lambda x: cvcuda_tv.invert(x),
        lambda x: cvcuda_tv.jpeg(image=x, quality=100),
        lambda x: cvcuda_tv.five_crop(x, [8, 8])[4],
    ],
    ids=["positional", "keyword", "tuple"],
)
def test_torch_input_runs_on_caller_stream(torch, call):
    # Busy the legacy stream so work queued there finishes long after the
    # caller's non-blocking stream reads the result.
    caller = torch.cuda.Stream()
    for value in (11, 37, 101):
        image = torch.full((3, 256, 256), value, device="cuda", dtype=torch.uint8)
        expected = call(cvcuda.as_tensor(image, "CHW"))
        expected = as_torch(torch, expected).clone()
        torch.cuda.synchronize()
        torch.cuda._sleep(200_000_000)
        with torch.cuda.stream(caller):
            actual = call(image).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_torch_bridge_stream_selection(torch):
    wrapped = _torch.accept_torch(lambda inpt, *, stream=None: (inpt, stream))
    image = torch.zeros((3, 4, 4), device="cuda")
    caller = torch.cuda.Stream()
    with torch.cuda.stream(caller):
        _, implicit = wrapped(image)
    assert implicit.handle == caller.cuda_stream

    explicit = cvcuda.Stream()
    with torch.cuda.stream(caller):
        _, chosen = wrapped(image, stream=explicit)
    assert chosen is explicit

    native = cvcuda.as_tensor(image, "CHW")
    assert wrapped(native) == (native, None)


@pytest.mark.parametrize(
    "name,call",
    [
        ("erase", lambda image, extra: cvcuda_tv.erase(image, 2, 3, 4, 5, extra)),
        ("elastic", lambda image, extra: cvcuda_tv.elastic(image, extra)),
    ],
)
def test_torch_secondary_runs_on_caller_stream(torch, monkeypatch, name, call):
    caller = torch.cuda.Stream()
    image = torch.full((3, 16, 20), 9, device="cuda", dtype=torch.uint8)
    extras = {
        "erase": lambda: torch.full((3, 1, 1), 200.0, device="cuda"),
        "elastic": lambda: torch.zeros((1, 16, 20, 2), device="cuda"),
    }
    expected = call(image, extras[name]()).clone()
    torch.cuda.synchronize()

    native = getattr(cvcuda, "remap" if name == "elastic" else "erase")
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs["stream"].handle)
        return native(*args, **kwargs)

    monkeypatch.setattr(cvcuda, native.__name__, spy)
    torch.cuda._sleep(200_000_000)
    with torch.cuda.stream(caller):
        actual = call(image, extras[name]()).clone()
    torch.cuda.synchronize()
    assert seen == [caller.cuda_stream]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("name", ["erase", "elastic"])
def test_non_tensor_image_with_torch_argument_uses_operator_error(torch, name):
    """A Torch tensor argument must not make the bridge touch a non-tensor image."""
    if name == "erase":
        value = torch.zeros(1, device="cuda")
        call = lambda v: cvcuda_tv.erase(None, 0, 0, 1, 1, v)  # noqa: E731
    else:
        value = torch.zeros((1, 4, 4, 2), device="cuda")
        call = lambda v: cvcuda_tv.elastic(None, v)  # noqa: E731
    with pytest.raises(Exception) as native:
        call(cvcuda.as_tensor(value))
    first_line = re.escape(str(native.value).splitlines()[0])
    with pytest.raises(type(native.value), match=first_line):
        call(value)


@pytest.mark.parametrize("device_index", [0, 3])
@pytest.mark.parametrize("stream_handle", [0, 1234])
def test_torch_import_uses_source_stream(monkeypatch, device_index, stream_handle):
    """Select the input device's current stream without caching its identity."""
    torch = pytest.importorskip("torch")

    interface = {"version": 2, "shape": (3, 8, 8)}
    source = SimpleNamespace(
        ndim=3,
        device=torch.device("cuda", device_index),
        get_device=lambda: device_index,
        __cuda_array_interface__=interface,
    )

    def current_stream(device):
        assert device == device_index
        return SimpleNamespace(cuda_stream=stream_handle)

    def as_tensor(owner, layout):
        assert owner._tensor is source
        assert layout == "CHW"
        assert owner.__cuda_array_interface__ == {
            **interface,
            "version": 3,
            "stream": stream_handle or -1,
        }
        return owner

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(cvcuda, "as_tensor", as_tensor)
    _torch._as_cvcuda(source)
    assert interface == {"version": 2, "shape": (3, 8, 8)}
