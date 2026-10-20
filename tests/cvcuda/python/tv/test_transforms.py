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

"""cvcuda.tv.Compose against torchvision.transforms.v2.Compose.

Per-operator parity is enforced against torchvision, allowing only documented
rounding ties and probed native gaps. Pipelines must draw the same random
parameters and convert the image exactly once in and once out; their output is
checked only roughly because rounding ties compound across transforms.
"""

import collections
import functools
import inspect
import pickle
import subprocess
import sys
import textwrap
import threading

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch as _torch_module  # noqa: E402
from torchvision import tv_tensors  # noqa: E402
from torchvision.transforms import v2  # noqa: E402
from torchvision.transforms.v2 import functional as F  # noqa: E402
from torchvision.transforms.v2.functional._utils import _KERNEL_REGISTRY  # noqa: E402

import cvcuda  # noqa: E402
from cvcuda import tv as cvcuda_tv  # noqa: E402
from cvcuda.tv import _torch as tv_torch  # noqa: E402
from cvcuda.tv import _compose  # noqa: E402
from cvcuda.tv._compose import CvcudaComposeError  # noqa: E402

from conftest import as_torch, tie_exempt  # noqa: E402

_META = {"get_size", "get_dimensions", "get_num_channels", "get_num_frames"}
_FUNCTIONALS = [name for name in cvcuda_tv.__all__ if name != "Compose"]


# ----------------------------------------------------------------- probes


class _Probe:
    """Counts image conversions and records the ops each side ran."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.conv = collections.Counter()
        self.cv = []
        self.tv = []
        self.record = set()
        self.calls = []
        imports, exports = tv_torch._as_cvcuda, tv_torch._to_torch

        def counted_import(tensor):
            self.conv["in"] += 1
            return imports(tensor)

        def counted_export(value):
            # _to_torch recurses through this name, so tuples count per leaf.
            self.conv["out"] += isinstance(value, cvcuda.Tensor)
            return exports(value)

        monkeypatch.setattr(tv_torch, "_as_cvcuda", counted_import)
        monkeypatch.setattr(tv_torch, "_to_torch", counted_export)

    def trace_torchvision(self, overrides=None):
        """Record torchvision's CUDA image kernels by functional name."""
        for functional, registry in _KERNEL_REGISTRY.items():
            if functional.__name__ in _META:
                continue
            for kind in (_torch_module.Tensor, tv_tensors.Image):
                if kind not in registry:
                    continue
                kernel = (overrides or {}).get(functional.__name__, registry[kind])

                def traced(*args, _kernel=kernel, _name=functional.__name__, **kw):
                    if args[0].is_cuda:
                        self.tv.append(_name)
                    return _kernel(*args, **kw)

                self.monkeypatch.setitem(registry, kind, traced)

    def trace_cvcuda(self):
        """Record every public cvcuda.tv call; keep the I/O of ops in ``record``."""
        for name in _FUNCTIONALS:
            public = getattr(cvcuda_tv, name)

            def traced(inpt, *args, _public=public, _name=name, **kwargs):
                self.cv.append(_name)
                out = _public(inpt, *args, **kwargs)
                if _name in self.record:
                    kwargs.pop("stream", None)
                    self.calls.append(
                        (_name, _planar(inpt), args, kwargs, _planar(out))
                    )
                return out

            self.monkeypatch.setattr(cvcuda_tv, name, traced)

    def clear(self):
        self.conv.clear()
        self.cv.clear()
        self.tv.clear()
        self.calls.clear()


@pytest.fixture
def probe(torch, monkeypatch):
    return _Probe(monkeypatch)


def _planar(value):
    return _torch_module.as_tensor(value.cuda()).clone()


@pytest.fixture(scope="session")
def native_gaps(torch):
    """Native ops that do not yet match torchvision bit for bit on uint8.

    Each gap is probed, not assumed, so a gate turns into a hard failure the moment
    the native fix lands: equalize (HistogramEq Pillow table), adjust_saturation
    (fused multiply-add blend order) and adjust_sharpness (double-precision factor).
    """
    torch.manual_seed(0)
    image = torch.randint(10, 240, (3, 64, 48), dtype=torch.uint8, device="cuda")
    probes = {
        "equalize": lambda: (cvcuda_tv.equalize(image), F.equalize(image)),
        "adjust_saturation": lambda: (
            cvcuda_tv.adjust_saturation(image, 1.3),
            F.adjust_saturation(image, 1.3),
        ),
        "adjust_sharpness": lambda: (
            cvcuda_tv.adjust_sharpness(image, 1.3),
            F.adjust_sharpness(image, 1.3),
        ),
    }
    return frozenset(op for op, run in probes.items() if not torch.equal(*run()))


def _native_gate(native_gaps, hit, check):
    """Run ``check``; xfail only if it fails in a run where a known native gap changed an output.

    ``hit`` lists the gap ops whose own call differed from torchvision in this run, so
    runs that drew an exact factor (or a float dtype) stay strict. The session probe
    empties ``native_gaps`` once the native fixes land, which removes the gate.
    """
    used = sorted(native_gaps & set(hit))
    if not used:
        check()
        return
    try:
        check()
    except AssertionError:
        pytest.xfail(f"native op not yet torchvision-exact: {', '.join(used)}")


def _gap_hits(native_gaps, probe):
    """Gap ops whose own call differed from torchvision in this run (needs ``record``)."""
    hits = []
    for op, inpt, args, kwargs, out in probe.calls:
        if (
            op in native_gaps
            and _mismatch(getattr(F, op)(inpt, *args, **kwargs), out).any()
        ):
            hits.append(op)
    return hits


def _rng():
    return _torch_module.get_rng_state(), _torch_module.cuda.get_rng_state()


def _rng_equal(a, b):
    return all(_torch_module.equal(p, q) for p, q in zip(a, b))


def _seed(seed):
    _torch_module.manual_seed(seed)
    _torch_module.cuda.manual_seed(seed)


def _types(value):
    if isinstance(value, (tuple, list)):
        return tuple(_types(item) for item in value)
    return type(value)


def _leaves(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def _mismatch(ref, got):
    if ref.is_floating_point():
        return (ref.double() - got.double()).abs() > 1e-4
    return ref != got


def _equal_or_raise(a, b):
    assert _equal(a, b)


def _equal(a, b):
    if isinstance(a, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_equal(p, q) for p, q in zip(a, b))
        )
    return type(a) is type(b) and a.shape == b.shape and _torch_module.equal(a, b)


def _seeded(transforms, inpt, seed=0):
    """Run both Composes from one seed; return (ref, got, rng_equal)."""
    _seed(seed)
    ref = v2.Compose(transforms)(inpt)
    after = _rng()
    _seed(seed)
    got = cvcuda_tv.Compose(transforms)(inpt)
    return ref, got, _rng_equal(after, _rng())


# ----------------------------------------------------------------- class table

U8, F32, BOTH = ("uint8",), ("float32",), ("uint8", "float32")
_AA = ("autocontrast", "rotate", "affine")


def _solarize(dtype):
    return 128 if dtype == "uint8" else 0.5


# (name, factory(dtype), dtypes, status, oracle ops). Status: S bit-exact,
# L residuals explained by the oracle ops' ties or a stated limit.
CLASSES = [
    ("AutoAugment", lambda d: v2.AutoAugment(), BOTH, "L", _AA),
    ("CenterCrop", lambda d: v2.CenterCrop(40), BOTH, "S", ()),
    (
        "ColorJitter",
        lambda d: v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
        BOTH,
        "L",
        ("adjust_hue",),
    ),
    (
        "Compose",
        lambda d: v2.Compose([v2.RandomHorizontalFlip(), v2.CenterCrop(40)]),
        BOTH,
        "S",
        (),
    ),
    (
        "ConvertImageDtype",
        lambda d: v2.ConvertImageDtype(_torch_module.float32),
        U8,
        "S",
        (),
    ),
    (
        "ElasticTransform",
        lambda d: v2.ElasticTransform(alpha=20.0),
        BOTH,
        "L",
        ("elastic",),
    ),
    ("FiveCrop", lambda d: v2.FiveCrop(24), BOTH, "S", ()),
    (
        "GaussianBlur",
        lambda d: v2.GaussianBlur(5, (0.1, 2.0)),
        BOTH,
        "L",
        ("gaussian_blur",),
    ),
    ("GaussianNoise", lambda d: v2.GaussianNoise(), F32, "L", None),
    ("Grayscale", lambda d: v2.Grayscale(3), BOTH, "S", ()),
    ("Identity", lambda d: v2.Identity(), BOTH, "S", ()),
    ("JPEG", lambda d: v2.JPEG((50, 90)), U8, "L", None),
    ("Normalize", lambda d: v2.Normalize([0.5] * 3, [0.2] * 3), F32, "S", ()),
    ("Pad", lambda d: v2.Pad([1, 2, 3, 4], fill=7), BOTH, "S", ()),
    ("RGB", lambda d: v2.RGB(), BOTH, "S", ()),
    ("RandAugment", lambda d: v2.RandAugment(num_ops=3), BOTH, "L", _AA),
    (
        "RandomAdjustSharpness",
        lambda d: v2.RandomAdjustSharpness(2.0, 0.7),
        BOTH,
        "S",
        (),
    ),
    (
        "RandomAffine",
        lambda d: v2.RandomAffine(20, (0.1, 0.1), (0.8, 1.2), (5, 10, 5, 10)),
        BOTH,
        "L",
        ("affine",),
    ),
    (
        "RandomApply",
        lambda d: v2.RandomApply(
            [v2.RandomInvert(1.0), v2.RandomVerticalFlip(1.0)], 0.6
        ),
        BOTH,
        "S",
        (),
    ),
    (
        "RandomAutocontrast",
        lambda d: v2.RandomAutocontrast(0.7),
        BOTH,
        "L",
        ("autocontrast",),
    ),
    (
        "RandomChannelPermutation",
        lambda d: v2.RandomChannelPermutation(),
        BOTH,
        "S",
        (),
    ),
    (
        "RandomChoice",
        lambda d: v2.RandomChoice(
            [v2.RandomInvert(1.0), v2.CenterCrop(30)], p=[0.3, 0.7]
        ),
        BOTH,
        "S",
        (),
    ),
    (
        "RandomCrop",
        lambda d: v2.RandomCrop(
            40, padding=[2, 3], pad_if_needed=True, padding_mode="edge"
        ),
        BOTH,
        "S",
        (),
    ),
    ("RandomEqualize", lambda d: v2.RandomEqualize(0.7), BOTH, "S", ()),
    ("RandomErasing", lambda d: v2.RandomErasing(0.8, value=0), BOTH, "S", ()),
    ("RandomGrayscale", lambda d: v2.RandomGrayscale(0.6), BOTH, "S", ()),
    ("RandomHorizontalFlip", lambda d: v2.RandomHorizontalFlip(), BOTH, "S", ()),
    ("RandomInvert", lambda d: v2.RandomInvert(0.6), BOTH, "S", ()),
    (
        "RandomOrder",
        lambda d: v2.RandomOrder(
            [v2.RandomInvert(1.0), v2.CenterCrop(30), v2.RandomVerticalFlip(1.0)]
        ),
        BOTH,
        "S",
        (),
    ),
    (
        "RandomPerspective",
        lambda d: v2.RandomPerspective(0.4, 0.8),
        BOTH,
        "L",
        ("perspective",),
    ),
    (
        "RandomPhotometricDistort",
        lambda d: v2.RandomPhotometricDistort(p=0.7),
        BOTH,
        "L",
        ("adjust_hue",),
    ),
    ("RandomPosterize", lambda d: v2.RandomPosterize(3, 0.7), U8, "S", ()),
    ("RandomResize", lambda d: v2.RandomResize(20, 60), BOTH, "L", ("resize",)),
    (
        "RandomResizedCrop",
        lambda d: v2.RandomResizedCrop(32),
        BOTH,
        "L",
        ("resized_crop",),
    ),
    (
        "RandomRotation",
        lambda d: v2.RandomRotation(30, expand=True),
        BOTH,
        "L",
        ("rotate",),
    ),
    (
        "RandomShortestSize",
        lambda d: v2.RandomShortestSize([30, 40, 50], max_size=70),
        BOTH,
        "L",
        ("resize",),
    ),
    ("RandomSolarize", lambda d: v2.RandomSolarize(_solarize(d), 0.7), BOTH, "S", ()),
    ("RandomVerticalFlip", lambda d: v2.RandomVerticalFlip(), BOTH, "S", ()),
    ("RandomZoomOut", lambda d: v2.RandomZoomOut(fill=3, p=0.8), BOTH, "S", ()),
    ("Resize", lambda d: v2.Resize(33, max_size=50), BOTH, "L", ("resize",)),
    ("ScaleJitter", lambda d: v2.ScaleJitter((60, 60)), BOTH, "L", ("resize",)),
    ("TenCrop", lambda d: v2.TenCrop(24, vertical_flip=True), BOTH, "S", ()),
    ("ToDtype", lambda d: v2.ToDtype(_torch_module.float16, scale=True), BOTH, "S", ()),
    ("ToImage", lambda d: v2.ToImage(), BOTH, "S", ()),
    ("ToPureTensor", lambda d: v2.ToPureTensor(), BOTH, "S", ()),
    ("TrivialAugmentWide", lambda d: v2.TrivialAugmentWide(), BOTH, "L", _AA),
]

REJECTED = {
    "AugMix": lambda: v2.AugMix(),
    "ClampBoundingBoxes": lambda: v2.ClampBoundingBoxes(),
    "ClampKeyPoints": lambda: v2.ClampKeyPoints(),
    "ConvertBoundingBoxFormat": lambda: v2.ConvertBoundingBoxFormat("XYXY"),
    "CutMix": lambda: v2.CutMix(num_classes=2),
    "Lambda": lambda: v2.Lambda(lambda t: t),
    "LinearTransformation": lambda: v2.LinearTransformation(
        _torch_module.eye(4), _torch_module.zeros(4)
    ),
    "MixUp": lambda: v2.MixUp(num_classes=2),
    "PILToTensor": lambda: v2.PILToTensor(),
    "RandomIoUCrop": lambda: v2.RandomIoUCrop(),
    "SanitizeBoundingBoxes": lambda: v2.SanitizeBoundingBoxes(),
    "SanitizeKeyPoints": lambda: v2.SanitizeKeyPoints(),
    "SetClampingMode": lambda: v2.SetClampingMode("soft"),
    "ToPILImage": lambda: v2.ToPILImage(),
    "ToTensor": lambda: v2.ToTensor(),
    "Transform": lambda: v2.Transform(),
    "UniformTemporalSubsample": lambda: v2.UniformTemporalSubsample(2),
}

_LAYOUTS = {"chw": (3, 48, 56), "nchw": (2, 3, 48, 56)}


def _class_params():
    params = []
    for name, factory, dtypes, status, oracles in CLASSES:
        assert (status == "S") == (oracles == ())
        if oracles is None and name != "JPEG":
            continue
        for dtype in dtypes:
            for layout in _LAYOUTS:
                for seed in range(3):
                    params.append(
                        pytest.param(
                            name,
                            factory,
                            dtype,
                            layout,
                            seed,
                            oracles,
                            id=f"{name}-{dtype}-{layout}-{seed}",
                        )
                    )
    return params


def _image(shape, dtype, seed):
    _torch_module.manual_seed(seed)
    x = _torch_module.randint(10, 240, shape, dtype=_torch_module.uint8, device="cuda")
    return x if dtype == "uint8" else x.float() / 255


def test_table_complete():
    classes = {
        name
        for name in dir(v2)
        if inspect.isclass(getattr(v2, name))
        and issubclass(getattr(v2, name), _torch_module.nn.Module)
    }
    table = {row[0] for row in CLASSES}
    assert not table & set(REJECTED)
    assert classes == table | set(REJECTED)
    assert len(CLASSES) + len(REJECTED) == len(classes)


def test_registry_complete():
    assert "Compose" in cvcuda_tv.__all__
    assert cvcuda_tv.Compose is _compose.Compose
    image = {
        f.__name__ for f, reg in _KERNEL_REGISTRY.items() if tv_tensors.Image in reg
    } - _META
    ours = {f.__name__ for f, reg in _KERNEL_REGISTRY.items() if _compose._Img in reg}
    assert image == ours == set(_FUNCTIONALS)


@pytest.mark.parametrize("name,factory,dtype,layout,seed,oracles", _class_params())
def test_class_parity(probe, native_gaps, name, factory, dtype, layout, seed, oracles):
    x = _image(_LAYOUTS[layout], dtype, 1000 + seed)
    transform = factory(dtype)
    _seed(seed)
    ref = transform(x.cpu()).cuda() if name == "JPEG" else transform(x)
    after = _rng()

    pipeline = cvcuda_tv.Compose([transform])
    probe.record = set(oracles or ("jpeg",)) | set(native_gaps)
    probe.trace_cvcuda()
    probe.clear()
    _seed(seed)
    got = pipeline(x)

    leaves = _leaves(got)
    assert _rng_equal(after, _rng())
    assert _types(got) == _types(ref)
    assert dict(probe.conv) == {"in": 1, "out": len(leaves)}

    if name == "JPEG":
        ((_, _, args, kwargs, out),) = probe.calls
        quality = kwargs.get("quality", args[0] if args else None)
        assert 50 <= quality <= 90
        assert _torch_module.equal(got, cvcuda_tv.jpeg(x, quality))
        assert _torch_module.equal(out, got)
        return

    gap_hits = _gap_hits(native_gaps, probe)
    # Gap ops are recorded only to detect gap hits; the oracle loop checks the rest.
    calls = [c for c in probe.calls if c[0] in (oracles or ())]

    for op, inpt, args, kwargs, out in calls:
        expected = getattr(F, op)(inpt, *args, **kwargs)
        assert expected.shape == out.shape, op
        mismatch = _mismatch(expected, out)
        ties = tie_exempt(op, expected, out, inpt, args, kwargs)
        if op in native_gaps and (mismatch & ~ties).any():
            gap_hits.append(op)
        else:
            assert not (mismatch & ~ties).any(), f"{op}: unexplained mismatch"

    def numerics():
        assert not gap_hits, f"native gap changed output: {gap_hits}"
        for r, g in zip(_leaves(ref), leaves):
            _pipeline_sanity(r, g)

    _native_gate(native_gaps, gap_hits, numerics)


def test_gaussian_noise(probe):
    # CV-CUDA generates the noise independently of Torch's CUDA generator;
    # CPU-sampled transform parameters still stay in sync.
    x = _torch_module.full((3, 256, 256), 0.5, device="cuda")
    noise = v2.GaussianNoise(mean=0.0, sigma=0.1, clip=False)
    transforms = [noise, v2.RandomErasing(1.0, value="random")]
    pipeline = cvcuda_tv.Compose(transforms)
    for seed in range(3):
        _seed(seed)
        v2.Compose(transforms)(x)
        cpu_after = _torch_module.get_rng_state()
        _seed(seed)
        cuda_before = _torch_module.cuda.get_rng_state()
        probe.clear()
        pipeline(x)
        assert _torch_module.equal(cpu_after, _torch_module.get_rng_state())
        assert _torch_module.equal(cuda_before, _torch_module.cuda.get_rng_state())
        assert dict(probe.conv) == {"in": 1, "out": 1}
        residual = cvcuda_tv.Compose([noise])(x) - x
        assert abs(residual.mean().item()) < 2e-3
        assert abs(residual.std().item() - 0.1) < 2e-3


@pytest.mark.parametrize("dtype", ["uint8", "float16"])
def test_gaussian_noise_dtypes(torch, dtype):
    x = torch.full((3, 64, 64), 128, dtype=torch.uint8, device="cuda")
    x = x if dtype == "uint8" else (x.float() / 255).half()
    got = cvcuda_tv.Compose([v2.GaussianNoise()])(x)
    assert got.dtype == x.dtype and got.shape == x.shape
    assert not torch.equal(got, x)


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_rejected(name):
    transform = REJECTED[name]()
    with pytest.raises(TypeError, match=f"does not support {name}"):
        cvcuda_tv.Compose([transform])
    with pytest.raises(TypeError, match=f"does not support {name}"):
        cvcuda_tv.Compose([v2.RandomApply([v2.Compose([transform])])])


def test_rejected_non_v2_callable():
    with pytest.raises(TypeError, match="only accepts torchvision.transforms.v2"):
        cvcuda_tv.Compose([lambda t: t])


# ----------------------------------------------------------------- oracles

_ORACLE_CASES = [
    *(
        (f"rotate{angle}", "rotate", (angle,), {}, 0.02)
        for angle in (45.0, -45.0, 135.0, 30.0, 13.5)
    ),
    ("affine", "affine", (17.0, [3, -2], 1.1, [5.0, -4.0]), {}, 0.001),
    (
        "perspective-nearest",
        "perspective",
        ([[0, 0], [79, 0], [79, 63], [0, 63]], [[4, 2], [75, 6], [77, 60], [2, 61]]),
        {"interpolation": v2.InterpolationMode.NEAREST},
        0.0012,
    ),
    ("resize-down", "resize", ([40, 52],), {"antialias": True}, 0.0005),
    ("resize-up", "resize", ([97, 113],), {"antialias": True}, 0.0005),
    ("resize-noaa", "resize", ([40, 52],), {"antialias": False}, 0.0005),
    ("autocontrast-u8", "autocontrast", (), {}, 0.0102),
    ("autocontrast-u16", "autocontrast", (), {}, 0.0038),
    *((f"hue{h}", "adjust_hue", (h,), {}, 0.0001) for h in (0.05, -0.1, 0.27, 0.5)),
]


@pytest.mark.parametrize("batched", [False, True], ids=["chw", "nchw"])
@pytest.mark.parametrize(
    "case_id,op,args,kwargs,max_frac",
    [pytest.param(case[0], *case[1:], id=case[0]) for case in _ORACLE_CASES],
)
def test_oracle(torch, case_id, op, args, kwargs, max_frac, batched):
    """Every per-op mismatch is a documented tie; their rate is not systematic.

    ``max_frac`` is twice the largest rate measured across the native builds.
    """
    torch.manual_seed(0)
    shape = (2, 3, 64, 80) if batched else (3, 64, 80)
    if op == "adjust_hue":
        shape = (32, 3, 64, 80)
        x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    elif case_id == "autocontrast-u16":
        x = torch.randint(3000, 60000, shape, dtype=torch.int32, device="cuda")
        x = x.to(torch.uint16)
    else:
        x = torch.randint(10, 240, shape, dtype=torch.uint8, device="cuda")
    ref = getattr(F, op)(x, *args, **kwargs)
    got = getattr(cvcuda_tv, op)(x, *args, **kwargs)
    mismatch = ref != got
    ties = tie_exempt(op, ref, got, x, args, kwargs)
    assert not (mismatch & ~ties).any()
    assert mismatch.float().mean().item() <= max_frac
    if op == "autocontrast" and x.dtype == torch.uint8:
        image = x.to(torch.int64)
        low = image.amin((-2, -1), keepdim=True)
        high = image.amax((-2, -1), keepdim=True)
        assert torch.equal(got.to(torch.int64), (image - low) * 255 // (high - low))


def _pipeline_sanity(ref, got):
    assert ref.shape == got.shape
    assert ref.dtype == got.dtype
    # Thresholds and histogram LUTs amplify earlier ties; parity is per operator.
    delta = (ref.float() - got.float()).abs().mean().item()
    assert delta <= (3.0 if ref.dtype == _torch_module.uint8 else 0.1), delta


# ----------------------------------------------------------------- mixed pipeline


def _mixed(dtype):
    return [
        v2.Resize(160),
        v2.RandomResizedCrop(144, scale=(0.5, 1.0)),
        v2.CenterCrop(136),
        v2.RandomCrop(128, padding=4),
        v2.Pad(2, padding_mode="reflect"),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(15),
        v2.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        v2.RandomPerspective(0.3, p=0.7),
        v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
        v2.RandomGrayscale(p=0.3),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.6),
        v2.RandomChoice(
            [
                v2.RandomInvert(1.0),
                v2.RandomPosterize(4, 1.0),
                v2.RandomSolarize(_solarize(dtype), 1.0),
            ]
        ),
        v2.RandomOrder(
            [
                v2.RandomAutocontrast(0.7),
                v2.RandomEqualize(0.7),
                v2.RandomAdjustSharpness(2.0, 0.7),
            ]
        ),
        v2.RandAugment(num_ops=2),
        v2.TrivialAugmentWide(),
        v2.AutoAugment(),
        v2.ElasticTransform(alpha=20.0),
        v2.ToDtype(_torch_module.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.7, value="random"),
    ]


def _exact_autocontrast(kernel):
    """torchvision's kernel with integer autocontrast as cvcuda.tv's exact floor."""

    def autocontrast(image, *args, **kwargs):
        if image.is_floating_point():
            return kernel(image, *args, **kwargs)
        bound = _torch_module.iinfo(image.dtype).max
        i = image.to(_torch_module.int64)
        low = i.amin((-2, -1), keepdim=True)
        high = i.amax((-2, -1), keepdim=True)
        flat = high == low
        q = (i - low) * bound // _torch_module.where(flat, 1, high - low)
        return _torch_module.where(flat, i, q).clamp(0, bound).to(image.dtype)

    return autocontrast


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("batched", [False, True], ids=["chw", "nchw"])
@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_mixed_pipeline(probe, native_gaps, dtype, batched, seed):
    shape = (4, 3, 200, 240) if batched else (3, 200, 240)
    x = _image(shape, dtype, 100)
    transforms = _mixed(dtype)
    registry = _KERNEL_REGISTRY[F.autocontrast]
    probe.trace_torchvision(
        {"autocontrast": _exact_autocontrast(registry[_torch_module.Tensor])}
    )
    _seed(seed)
    ref = v2.Compose(transforms)(x)
    after, tv_trace = _rng(), list(probe.tv)

    pipeline = cvcuda_tv.Compose(transforms)
    probe.record = set(native_gaps)
    probe.trace_cvcuda()
    probe.clear()
    _seed(seed)
    got = pipeline(x)

    assert dict(probe.conv) == {"in": 1, "out": 1}
    assert _rng_equal(after, _rng())
    assert probe.cv == tv_trace
    assert probe.tv == []
    assert _types(got) == _types(ref)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype

    def numerics():
        _pipeline_sanity(ref, got)

    _native_gate(native_gaps, _gap_hits(native_gaps, probe), numerics)


# ----------------------------------------------------------------- semantics


@pytest.fixture
def x(torch):
    torch.manual_seed(7)
    return torch.randint(10, 240, (3, 40, 48), dtype=torch.uint8, device="cuda")


@pytest.fixture
def proxy(x):
    state = _compose._TLS
    state.device, state.stream, state.pure = x.device, None, True
    yield _compose._proxy(cvcuda.as_tensor(x, "CHW"))
    del state.device


def test_proxy_keeps_image_through_aliases(proxy):
    assert repr(proxy).startswith("Image(cvcuda, shape=(3, 40, 48)")
    assert proxy.as_subclass(tv_tensors.Image)._cv is proxy._cv
    assert tv_tensors.wrap(proxy, like=proxy)._cv is proxy._cv
    assert proxy.detach()._cv is proxy._cv


@pytest.mark.parametrize(
    "use,match",
    [
        (lambda p: p.data_ptr(), "no torch storage"),
        (lambda p: p.__cuda_array_interface__, "no torch storage"),
        (lambda p: _torch_module.as_tensor(p, dtype=_torch_module.float32), "_to_copy"),
        (lambda p: p * 2, "aten.mul"),
    ],
    ids=["data_ptr", "cuda_array_interface", "as_tensor", "mul"],
)
def test_proxy_rejects_torch_access(proxy, use, match):
    with pytest.raises(CvcudaComposeError, match=match):
        use(proxy)
    assert issubclass(CvcudaComposeError, RuntimeError)


def test_proxy_unmapped_functional_raises(torch, proxy):
    image = torch.ones((2, 3, 4, 5), device="cuda")
    video = _compose._proxy(cvcuda.as_tensor(image, "NCHW"), _compose._Passive)
    with pytest.raises(CvcudaComposeError, match="not supported on a CV-CUDA image"):
        F.uniform_temporal_subsample(video, 2)


class _WrapLike(v2.Transform):
    def transform(self, inpt, params):
        return tv_tensors.wrap(F.horizontal_flip(inpt), like=inpt)


class _Printing(v2.Transform):
    def transform(self, inpt, params):
        # Printing a proxy must not touch its (absent) Torch storage.
        assert repr(inpt) and str(inpt)
        return F.vertical_flip(inpt)


class _Sampled(v2.Transform):
    def make_params(self, flat_inputs):
        return {"angle": float(_torch_module.empty(1).uniform_(-20, 20))}

    def transform(self, inpt, params):
        return F.rotate(inpt, params["angle"])


class _Fresh(v2.Transform):
    def transform(self, inpt, params):
        zeros = _torch_module.zeros(
            tuple(inpt.shape), dtype=inpt.dtype, device=inpt.device
        )
        return tv_tensors.wrap(zeros, like=inpt)


class _TorchMath(v2.Transform):
    def transform(self, inpt, params):
        return inpt * 2


class _OwnForward(v2.Transform):
    def forward(self, inpt):
        return inpt

    def transform(self, inpt, params):
        return inpt


@pytest.mark.parametrize("image", [False, True], ids=["tensor", "image"])
@pytest.mark.parametrize(
    "transforms",
    [
        lambda: [_WrapLike(), v2.RandomVerticalFlip(1.0)],
        lambda: [_Printing()],
        lambda: [_Sampled(), v2.RandomHorizontalFlip()],
    ],
    ids=["wrap_like", "repr", "make_params"],
)
def test_user_subclass_supported(x, transforms, image):
    inpt = tv_tensors.Image(x) if image else x
    ref, got, rng = _seeded(transforms(), inpt, 3)
    assert rng and _equal(ref, got)


def test_user_subclass_errors(x):
    with pytest.raises(CvcudaComposeError, match="_Fresh returned"):
        cvcuda_tv.Compose([_Fresh()])(x)
    with pytest.raises(CvcudaComposeError, match="aten.mul"):
        cvcuda_tv.Compose([_TorchMath()])(x)
    with pytest.raises(TypeError, match="overrides forward"):
        cvcuda_tv.Compose([_OwnForward()])


@pytest.mark.parametrize("image", [False, True], ids=["tensor", "image"])
@pytest.mark.parametrize(
    "transforms",
    [
        lambda: [v2.Pad(3, fill={tv_tensors.Image: 9, "others": 200})],
        lambda: [v2.Pad(3, fill={_torch_module.Tensor: 50, tv_tensors.Image: 9})],
        lambda: [v2.RandomRotation(30, fill={tv_tensors.Image: 7, "others": 99})],
        lambda: [
            v2.ToDtype(
                {
                    tv_tensors.Image: _torch_module.float32,
                    "others": _torch_module.float16,
                },
                scale=True,
            )
        ],
        lambda: [v2.ToDtype({tv_tensors.Image: _torch_module.float32, "others": None})],
        lambda: [v2.ToImage(), v2.Pad(2, fill={tv_tensors.Image: 9, "others": 200})],
        lambda: [
            v2.ToPureTensor(),
            v2.Pad(2, fill={tv_tensors.Image: 9, "others": 200}),
        ],
    ],
    ids=[
        "pad",
        "pad_tensor_key",
        "rotation",
        "to_dtype",
        "to_dtype_none",
        "to_image",
        "to_pure",
    ],
)
def test_per_type_dicts(x, transforms, image):
    inpt = tv_tensors.Image(x) if image else x
    ref, got, rng = _seeded(transforms(), inpt)
    assert rng and _equal(ref, got)


def test_per_type_dict_missing_key(x):
    pipeline = cvcuda_tv.Compose(
        [v2.ToDtype({tv_tensors.Image: _torch_module.float32})]
    )
    with pytest.raises(ValueError, match="No dtype"):
        pipeline(x)


@pytest.mark.parametrize("image", [False, True], ids=["tensor", "image"])
@pytest.mark.parametrize(
    "transforms",
    [
        lambda: [v2.FiveCrop(16), v2.RandomHorizontalFlip(1.0)],
        lambda: [
            v2.TenCrop(16),
            v2.Resize(8),
            v2.ToDtype(_torch_module.float32, scale=True),
        ],
        lambda: [v2.FiveCrop(16), v2.ColorJitter(0.4, 0.4, 0.4)],
    ],
    ids=["fivecrop_flip", "tencrop_resize", "fivecrop_jitter"],
)
def test_tuples(probe, native_gaps, x, transforms, image):
    inpt = tv_tensors.Image(x) if image else x
    probe.record = set(native_gaps)
    probe.trace_cvcuda()
    ref, got, rng = _seeded(transforms(), inpt)
    assert rng
    hits = _gap_hits(native_gaps, probe)
    _native_gate(native_gaps, hits, lambda: _equal_or_raise(ref, got))


@pytest.mark.parametrize("layout", ["HWC", "NHWC", "CHW", "NCHW"])
def test_layouts(torch, layout):
    sizes = {"N": 2, "C": 3, "H": 40, "W": 48}
    torch.manual_seed(2)
    owner = torch.randint(
        10, 240, tuple(sizes[a] for a in layout), dtype=torch.uint8, device="cuda"
    )
    planar = owner.permute(*(layout.index(a) for a in layout.replace("HWC", "CHW")))
    planar = planar.contiguous()
    transforms = [
        v2.RandomResizedCrop(24),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
    ]
    torch.manual_seed(1)
    ref = v2.Compose(transforms)(planar)
    torch.manual_seed(1)
    out = cvcuda_tv.Compose(transforms)(cvcuda.as_tensor(owner, layout))
    torch.manual_seed(1)
    ours_planar = cvcuda_tv.Compose(transforms)(planar)
    assert type(out) is cvcuda.Tensor and str(out.layout) == layout
    got = as_torch(torch, out)
    got = got.permute(*(layout.index(a) for a in layout.replace("HWC", "CHW")))
    assert torch.equal(got, ours_planar)
    assert (got.int() - ref.int()).abs().max().item() <= 1


def test_output_types(x):
    image = tv_tensors.Image(x)
    assert (
        type(cvcuda_tv.Compose([v2.RandomHorizontalFlip()])(image)) is tv_tensors.Image
    )
    assert (
        type(cvcuda_tv.Compose([v2.RandomHorizontalFlip()])(x)) is _torch_module.Tensor
    )
    assert type(cvcuda_tv.Compose([v2.Resize(20)])(image)) is tv_tensors.Image
    assert type(cvcuda_tv.Compose([v2.Resize(20)])(x)) is _torch_module.Tensor


def test_input_errors(x):
    pipeline = cvcuda_tv.Compose([v2.RandomHorizontalFlip()])
    with pytest.raises(TypeError, match="needs a CUDA tensor, got one on cpu"):
        pipeline(x.cpu())
    with pytest.raises(ValueError, match="3- or 4-dimensional"):
        pipeline(_torch_module.zeros(1, 2, 3, 8, 8, device="cuda"))
    with pytest.raises(TypeError, match="takes a cvcuda.Tensor or a CUDA torch.Tensor"):
        pipeline("image")


@pytest.mark.parametrize(
    "transforms,channels",
    [
        (lambda: [v2.Normalize([0.0] * 3, [1.0] * 3)], 3),
        (lambda: [v2.RandomCrop(100)], 3),
        (lambda: [v2.Grayscale()], 2),
        (lambda: [v2.Grayscale(), v2.RandomHorizontalFlip()], 2),
        (lambda: [v2.ColorJitter(brightness=0.3)], 2),
        (lambda: [v2.RandomGrayscale(1.0)], 2),
        (lambda: [v2.RGB()], 2),
    ],
    ids=[
        "normalize_u8",
        "crop_too_big",
        "gray",
        "gray_then_flip",
        "jitter",
        "random_gray",
        "rgb",
    ],
)
def test_error_parity(torch, transforms, channels):
    inpt = torch.randint(10, 240, (channels, 40, 48), dtype=torch.uint8, device="cuda")
    with pytest.raises(Exception) as expected:
        v2.Compose(transforms())(inpt)
    with pytest.raises(type(expected.value)) as actual:
        cvcuda_tv.Compose(transforms())(inpt)
    assert type(actual.value) is type(expected.value)


def test_construction_errors():
    with pytest.raises(ValueError, match="at least one"):
        cvcuda_tv.Compose([])
    with pytest.raises(TypeError, match="should be a sequence"):
        cvcuda_tv.Compose(iter([v2.Identity()]))
    transforms = [v2.Identity()]
    pipeline = cvcuda_tv.Compose(transforms)
    assert pipeline.transforms is transforms


# ----------------------------------------------------------------- streams, threads, state


def test_side_stream(torch):
    big = torch.rand(8, 3, 512, 512, device="cuda")
    transforms = [v2.GaussianBlur(31, 8.0), v2.RandomErasing(1.0, value="random")]
    torch.manual_seed(0)
    ref = v2.Compose(transforms)(big)
    torch.cuda.synchronize()
    pipeline = cvcuda_tv.Compose(transforms)
    stream = torch.cuda.Stream()
    for _ in range(10):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            torch.manual_seed(0)
            out = pipeline(big)
            # Computed on the side stream, so it only sees the output once
            # CV-CUDA's work is ordered on that stream.
            worst = (out - ref).abs().max()
        torch.cuda.current_stream().wait_stream(stream)
        assert worst.item() <= 1e-4


def test_static_side_stream(torch):
    torch.manual_seed(0)
    image = torch.randint(0, 256, (8, 3, 512, 512), dtype=torch.uint8, device="cuda")
    transforms = [
        v2.Resize(256),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.25] * 3),
    ]
    pipeline = cvcuda_tv.Compose(transforms)
    ref = pipeline(image)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    for _ in range(10):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            worst = (pipeline(image) - ref).abs().max()
        torch.cuda.current_stream().wait_stream(stream)
        assert worst.item() == 0


def test_two_threads_two_streams(torch):
    results = {}

    def worker(value):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            image = torch.full(
                (3, 64, 64), value * 10, dtype=torch.uint8, device="cuda"
            )
            pipeline = cvcuda_tv.Compose(
                [v2.RandomErasing(1.0, value=value), v2.Resize(32)]
            )
            outs = [pipeline(image) for _ in range(50)]
            stream.synchronize()
            results[value] = all(
                int(o.min()) >= value and int(o.max()) <= value * 10 for o in outs
            )

    threads = [threading.Thread(target=worker, args=(v,)) for v in (3, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {3: True, 7: True}


class _Nested(v2.Transform):
    def transform(self, inpt, params):
        inner = _torch_module.zeros(3, 8, 8, dtype=_torch_module.uint8, device="cuda")
        cvcuda_tv.Compose([v2.RandomHorizontalFlip(1.0)])(inner)
        return F.vertical_flip(inpt)


def test_nested_call_restores_state(x):
    out = cvcuda_tv.Compose([_Nested(), v2.FiveCrop(16)])(tv_tensors.Image(x))
    assert all(type(o) is tv_tensors.Image for o in out)


def test_state_restored_after_exception(x):
    with pytest.raises(ValueError):
        cvcuda_tv.Compose([v2.RandomCrop(100)])(tv_tensors.Image(x))
    out = cvcuda_tv.Compose([v2.RandomHorizontalFlip()])(x)
    assert type(out) is _torch_module.Tensor


def test_output_lifetime(torch):
    pipeline = cvcuda_tv.Compose(
        [
            v2.RandomResizedCrop(24),
            v2.ColorJitter(0.3, 0.3, 0.3),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    def image(value):
        return torch.full((3, 40, 48), value, dtype=torch.uint8, device="cuda")

    torch.manual_seed(5)
    refs = [v2.Compose(pipeline.transforms)(image(k)).clone() for k in range(16)]
    torch.manual_seed(5)
    outs = [pipeline(image(k)) for k in range(16)]
    junk = [torch.empty(1 << 22, device="cuda").fill_(-1) for _ in range(8)]
    del junk
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(outs, refs))


@pytest.mark.skipif(
    not _torch_module.cuda.is_available() or _torch_module.cuda.device_count() < 2,
    reason="needs two GPUs",
)
def test_second_device(torch):
    torch.manual_seed(0)
    image = torch.randint(10, 240, (3, 40, 48), dtype=torch.uint8, device="cuda:1")
    with torch.cuda.device(1):
        ref, got, rng = _seeded(
            [v2.RandomResizedCrop(24), v2.RandomHorizontalFlip()], image
        )
    assert got.device == image.device and rng
    assert (got.int() - ref.int()).abs().max().item() <= 1


# ----------------------------------------------------------------- pickling, DataLoader

_CHILD_GUARD = """
import collections, sys, torch
import cvcuda.tv as tv
from cvcuda.tv import _compose
from torchvision import tv_tensors
from torchvision.transforms.v2.functional._utils import _KERNEL_REGISTRY

CALLS = collections.Counter()
for _name in _compose._FUNCTIONALS:
    def _counted(*a, _f=getattr(tv, _name), _n=_name, **k):
        CALLS[_n] += 1
        return _f(*a, **k)
    setattr(tv, _name, _counted)
"""


def test_pickle_fresh_process(torch, tmp_path):
    pipeline = cvcuda_tv.Compose(
        [
            v2.RandomResizedCrop(24),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    blob = tmp_path / "compose.pkl"
    blob.write_bytes(pickle.dumps(pipeline))
    code = _CHILD_GUARD + textwrap.dedent(
        f"""
        import pickle
        c = pickle.loads(open({str(blob)!r}, "rb").read())
        def _forbidden(*a, **k):
            raise AssertionError("a torchvision image kernel ran")
        for _fn, _reg in _KERNEL_REGISTRY.items():
            if _fn.__name__.startswith("get_"):
                continue
            for _kind in (torch.Tensor, tv_tensors.Image):
                if _kind in _reg:
                    _reg[_kind] = _forbidden
        o = c(torch.randint(0, 256, (3, 40, 40), dtype=torch.uint8, device="cuda"))
        assert tuple(o.shape) == (3, 24, 24), o.shape
        assert CALLS["resized_crop"] == 1 and CALLS["to_dtype"] == 1, dict(CALLS)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]


_DATALOADER = _CHILD_GUARD + textwrap.dedent(
    """
    from torchvision.transforms import v2


    class Images(torch.utils.data.Dataset):
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def __len__(self):
            return 8

        def __getitem__(self, index):
            torch.manual_seed(index)
            image = torch.randint(10, 240, (3, 40, 48), dtype=torch.uint8, device="cuda")
            CALLS.clear()
            torch.manual_seed(100 + index)
            out = self.pipeline(image)
            assert CALLS["resized_crop"] == 1, dict(CALLS)
            torch.manual_seed(100 + index)
            ref = v2.Compose(self.pipeline.transforms)(image)
            # Resize ties: at most 1 LSB before ToDtype(scale=True).
            assert (out - ref).abs().max().item() <= 1 / 255 + 1e-6
            return out.float().mean().cpu()


    if __name__ == "__main__":
        pipeline = tv.Compose(
            [
                v2.RandomResizedCrop(24),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        loader = torch.utils.data.DataLoader(
            Images(pipeline),
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
            timeout=120,
        )
        values = torch.cat(list(loader))
        assert values.shape == (8,) and torch.isfinite(values).all(), values
    """
)


def test_dataloader_spawn(torch, tmp_path):
    script = tmp_path / "dataloader_spawn.py"
    script.write_text(_DATALOADER)
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


# ----------------------------------------------------------------- all-static path


def _sequential(steps, image):
    """Independent, explicitly specified chain of public functional calls."""
    for name, kwargs in steps:
        image = getattr(cvcuda_tv, name)(image, **kwargs)
    return image


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", ["uint8", "float32"])
@pytest.mark.parametrize("shape", [(3, 32, 37), (1, 1, 32, 37), (2, 3, 32, 37)])
def test_static_matches_functionals(torch, shape, dtype, strided, inplace):
    image = torch.randint(0, 128, shape, device="cuda", dtype=torch.uint8)
    image = image.to(getattr(torch, dtype))
    if strided:
        image = image[..., ::2, :]
    transforms = [
        v2.Resize((19, 23), interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop((17, 21)),
        v2.Pad([1, 2, 3, 4], fill=3),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize([0.25], [0.75], inplace=inplace),
    ]
    actual = cvcuda_tv.Compose(transforms)(image)
    expected = _sequential(
        [
            (
                "resize",
                {
                    "size": (19, 23),
                    "interpolation": v2.InterpolationMode.BICUBIC,
                    "max_size": None,
                    "antialias": True,
                },
            ),
            ("center_crop", {"output_size": (17, 21)}),
            ("pad", {"padding": [1, 2, 3, 4], "fill": 3, "padding_mode": "constant"}),
            ("to_dtype", {"dtype": torch.float32, "scale": False}),
            ("normalize", {"mean": [0.25], "std": [0.75], "inplace": inplace}),
        ],
        image,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert type(actual) is torch.Tensor
    assert actual.stride() == expected.stride()
    assert actual.device == image.device


@pytest.mark.parametrize("inplace", [False, True])
def test_static_aliasing(torch, inplace):
    image = torch.rand((1, 3, 17, 19), device="cuda")
    original = image.clone()
    transforms = [v2.Resize((17, 19)), v2.ToDtype(torch.float32, scale=False)]
    if inplace:
        transforms.append(v2.Normalize([0.5], [0.25], inplace=True))
    actual = cvcuda_tv.Compose(transforms)(image)
    steps = [
        (
            "resize",
            {
                "size": (17, 19),
                "interpolation": v2.InterpolationMode.BILINEAR,
                "max_size": None,
                "antialias": True,
            },
        ),
        ("to_dtype", {"dtype": torch.float32, "scale": False}),
    ]
    if inplace:
        steps.append(("normalize", {"mean": [0.5], "std": [0.25], "inplace": True}))
    expected = _sequential(steps, original)
    assert actual.data_ptr() == image.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("layout", ["CHW", "NCHW", "HWC", "NHWC"])
def test_layout_round_trip(torch, layout, multiple):
    sizes = {"N": 2, "C": 3, "H": 17, "W": 19}
    owner = torch.rand(tuple(sizes[axis] for axis in layout), device="cuda")
    image = cvcuda.as_tensor(owner, layout)
    transforms = [v2.Normalize([0.5], [0.25])]
    if multiple:
        transforms.append(v2.ToDtype(torch.float32))
    actual = cvcuda_tv.Compose(transforms)(image)
    expected = cvcuda_tv.normalize(image, [0.5], [0.25], inplace=False)
    if multiple:
        expected = cvcuda_tv.to_dtype(expected, torch.float32, scale=False)
    assert type(actual) is cvcuda.Tensor
    assert str(actual.layout) == layout
    torch.testing.assert_close(
        as_torch(torch, actual), as_torch(torch, expected), rtol=0, atol=0
    )


@pytest.mark.parametrize("random", [False, True], ids=["static", "proxy"])
@pytest.mark.parametrize("torch_input", [False, True], ids=["cvcuda", "torch"])
def test_converts_only_at_outer_boundary(torch, monkeypatch, torch_input, random):
    from unittest.mock import Mock

    image = torch.ones((1, 3, 17, 19), device="cuda", dtype=torch.uint8)
    if not torch_input:
        image = cvcuda.as_tensor(image, "NCHW")
    transforms = [
        v2.Resize((13, 15)),
        v2.ToDtype(torch.float32),
        v2.Normalize([0.5], [0.25]),
    ]
    if random:
        transforms.insert(1, v2.RandomHorizontalFlip())
    pipeline = cvcuda_tv.Compose(transforms)
    imports = Mock(wraps=tv_torch._as_cvcuda)
    exports = Mock(wraps=tv_torch._to_torch)
    monkeypatch.setattr(tv_torch, "_as_cvcuda", imports)
    monkeypatch.setattr(tv_torch, "_to_torch", exports)
    result = pipeline(image)
    torch.cuda.synchronize()
    assert imports.call_count == int(torch_input)
    assert exports.call_count == int(torch_input)
    assert isinstance(result, torch.Tensor if torch_input else cvcuda.Tensor)


@pytest.mark.parametrize("random", [False, True], ids=["static", "proxy"])
@pytest.mark.parametrize("wrapped_operator", ["resize", "to_dtype"])
@pytest.mark.parametrize("late_wrapper", [False, True])
@pytest.mark.parametrize("torch_input", [False, True], ids=["cvcuda", "torch"])
def test_user_wrapper_honoured(
    torch, monkeypatch, torch_input, wrapped_operator, random, late_wrapper
):
    operator = getattr(cvcuda_tv, wrapped_operator)
    calls = []

    @functools.wraps(operator)
    def wrapped(inpt, *args, **kwargs):
        calls.append(inpt)
        return operator(inpt, *args, **kwargs)

    if not late_wrapper:
        monkeypatch.setattr(cvcuda_tv, wrapped_operator, wrapped)
    transforms = [v2.Resize((3, 4)), v2.ToDtype(torch.float32)]
    if random:
        transforms.insert(1, v2.RandomHorizontalFlip(0.0))
    pipeline = cvcuda_tv.Compose(transforms)
    if late_wrapper:
        monkeypatch.setattr(cvcuda_tv, wrapped_operator, wrapped)
    image = torch.ones((1, 3, 5, 7), device="cuda", dtype=torch.uint8)
    if not torch_input:
        image = cvcuda.as_tensor(image, "NCHW")
    expected = cvcuda_tv.to_dtype(
        cvcuda_tv.resize(image, (3, 4)), torch.float32, scale=False
    )
    calls.clear()
    result = pipeline(image)
    torch.cuda.synchronize()
    assert len(calls) == 1
    assert [type(value) for value in calls] == [cvcuda.Tensor]
    if not random:
        if not torch_input:
            expected = as_torch(torch, expected)
        if not torch_input:
            result = as_torch(torch, result)
        assert torch.equal(result, expected)
    assert tuple(result.shape) == (1, 3, 3, 4)


@pytest.mark.parametrize(
    "case", ["cpu", "rank", "grad", "integer_normalize", "strided_pitch"]
)
@pytest.mark.parametrize("multiple", [False, True])
def test_error_types(torch, case, multiple):
    image = torch.ones((3, 17, 19), device="cuda")
    if case == "cpu":
        image = image.cpu()
    elif case == "rank":
        image = image[0]
    elif case == "grad":
        image.requires_grad_()
    elif case == "integer_normalize":
        image = image.to(torch.uint8)
    elif case == "strided_pitch":
        # Odd-height row slicing leaves less than height * row-stride between
        # channels, which the CV-CUDA tensor contract rejects.
        image = image[..., ::2, :]
    transforms = [v2.Normalize([0.5], [0.25])]
    if multiple:
        transforms.append(v2.ToDtype(torch.float32))
    if case == "cpu":
        with pytest.raises(TypeError, match="needs a CUDA tensor"):
            cvcuda_tv.Compose(transforms)(image)
        return
    with pytest.raises(Exception) as expected:
        expected_image = cvcuda_tv.normalize(image, [0.5], [0.25], inplace=False)
        if multiple:
            cvcuda_tv.to_dtype(expected_image, torch.float32, scale=False)
    with pytest.raises(type(expected.value)):
        cvcuda_tv.Compose(transforms)(image)
