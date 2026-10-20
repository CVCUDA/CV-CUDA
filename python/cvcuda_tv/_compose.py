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

"""Run TorchVision v2 transform pipelines as one CV-CUDA chain.

The image travels between transforms as a storage-less proxy: a Torch wrapper
subclass of ``tv_tensors.Image`` on the real CUDA device that carries the
``cvcuda.Tensor`` in ``_cv``. TorchVision's own parameter sampling
(``make_params``, containers, the AutoAugment family's ``forward``) runs
unchanged on it, so the RNG stream matches ``v2.Compose``. Its ``F.*`` calls
dispatch to the public ``cvcuda.tv`` functionals registered below, and any other
Torch op on the proxy raises.

Loaded lazily by ``cvcuda.tv.__getattr__`` because it imports Torch and
TorchVision, which ``cvcuda.tv`` itself does not need.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Any, NamedTuple, NoReturn, TypeAlias, TypeVar

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2._transform import Transform, _RandomApplyTransform
from torchvision.transforms.v2._utils import _get_fill, check_type

import cvcuda
import cvcuda.tv as tv

from . import _torch


class _State(threading.local):
    """Per-call device, stream and pure-tensor flag of the running pipeline."""

    device: torch.device
    stream: cvcuda.Stream | None
    pure: bool

    @property
    def state(self) -> tuple[torch.device, cvcuda.Stream | None, bool] | None:
        return (
            (self.device, self.stream, self.pure) if hasattr(self, "device") else None
        )


_TLS = _State()


class CvcudaComposeError(RuntimeError):
    """Raised when a transform does something with the image that CV-CUDA cannot run.

    A ``RuntimeError`` because Torch turns a ``TypeError`` raised by a binary
    operator into an unrelated "unsupported operand" message.
    """


_ALIAS: set[torch._ops.OpOverload] = {
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
}


class _ProxyMixin:
    _cv: cvcuda.Tensor | None = None

    @classmethod
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> _Img | _Passive:
        # as_subclass, tv_tensors.wrap(like=) and detach must keep the image.
        if func in _ALIAS:
            return _proxy(args[0]._cv, type(args[0]), args[0].device)
        raise CvcudaComposeError(
            f"cvcuda.tv.Compose: torch op {func} is not supported on a CV-CUDA image; "
            "a transform may only call torchvision.transforms.v2.functional ops that "
            "cvcuda.tv implements"
        )

    def __repr__(self, *, tensor_contents: str | None = None) -> str:
        kind = "Image" if isinstance(self, tv_tensors.Image) else "Tensor"
        return (
            f"{kind}(cvcuda, shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device})"
        )

    def data_ptr(self) -> NoReturn:
        raise CvcudaComposeError(
            "cvcuda.tv.Compose: a CV-CUDA image inside the pipeline has no torch storage"
        )


class _Img(_ProxyMixin, tv_tensors.Image):
    """Transformed element: TorchVision treats it as an Image."""


class _Passive(_ProxyMixin, torch.Tensor):
    """Tuple element that TorchVision's pure-tensor rule passes through untouched."""


_TORCH_DT: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(str(t).split(".")[1]): t
    for t in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.uint16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    )
}
# The proxy always has TorchVision's planar shape; interleaved data is not permuted.
_PLANAR: dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]] = {
    "CHW": lambda s: s,
    "NCHW": lambda s: s,
    "HWC": lambda s: (s[2], s[0], s[1]),
    "NHWC": lambda s: (s[0], s[3], s[1], s[2]),
}


def _proxy(
    cv: cvcuda.Tensor,
    cls: type[_Img] | type[_Passive] = _Img,
    device: torch.device | None = None,
) -> _Img | _Passive:
    p = torch.Tensor._make_wrapper_subclass(
        cls,
        _PLANAR[str(cv.layout)](cv.shape),
        dtype=_TORCH_DT[cv.dtype],
        device=device or _TLS.device,
    )
    p._cv = cv
    return p


def _wrap(
    out: cvcuda.Tensor | tuple[cvcuda.Tensor, ...] | list[cvcuda.Tensor],
    cls: type[_Img] | type[_Passive],
) -> _Img | _Passive | tuple[_Img | _Passive, ...] | list[_Img | _Passive]:
    if isinstance(out, (tuple, list)):
        # TorchVision: after a pure-tensor FiveCrop only crop 0 stays transformable.
        rest = _Passive if _TLS.pure else cls
        return type(out)(_proxy(o, cls if i == 0 else rest) for i, o in enumerate(out))
    return _proxy(out, cls)


_FUNCTIONALS: list[str] = [n for n in tv.__all__ if n != "Compose"]


def _make_kernel(
    name: str,
) -> Callable[
    ..., _Img | _Passive | tuple[_Img | _Passive, ...] | list[_Img | _Passive]
]:
    def kernel(
        inpt: _Img, *args: Any, **kwargs: Any
    ) -> _Img | _Passive | tuple[_Img | _Passive, ...] | list[_Img | _Passive]:
        # The public functional, looked up per call: it honours a user-installed
        # wrapper and converts erase's v / elastic's displacement on our stream.
        out = getattr(tv, name)(inpt._cv, *args, stream=_TLS.stream, **kwargs)
        return _wrap(out, type(inpt))

    return kernel


def _register() -> None:
    # At import, so a Compose unpickled in a spawned worker is ready to run.
    for n in _FUNCTIONALS:
        F.register_kernel(getattr(F, n), _Img)(_make_kernel(n))


_register()


_TYPE_ONLY: tuple[type[Transform], ...] = (v2.Identity, v2.ToPureTensor, v2.ToImage)
_OWN_FORWARD_OK: tuple[type[Transform], ...] = (
    v2.AutoAugment,
    v2.RandAugment,
    v2.TrivialAugmentWide,
)
_BOXES_ONLY = "it only applies to bounding boxes"
_UNSUPPORTED: dict[type[Transform], str] = {
    v2.AugMix: "it mixes images with torch arithmetic (Dirichlet weights, mul/add)",
    v2.Lambda: "it runs arbitrary torch code on the image; call it before or after the pipeline",
    v2.LinearTransformation: "it is a torch matmul with no CV-CUDA kernel; apply it after the pipeline",
    v2.MixUp: "it mixes a batch of samples and labels",
    v2.CutMix: "it mixes a batch of samples and labels",
    v2.RandomIoUCrop: "it requires bounding boxes",
    v2.SanitizeBoundingBoxes: _BOXES_ONLY,
    v2.SanitizeKeyPoints: "it only applies to keypoints",
    v2.ClampBoundingBoxes: _BOXES_ONLY,
    v2.ClampKeyPoints: "it only applies to keypoints",
    v2.ConvertBoundingBoxFormat: _BOXES_ONLY,
    v2.SetClampingMode: "it only applies to bounding boxes and keypoints",
    v2.UniformTemporalSubsample: "it only applies to videos",
    v2.PILToTensor: "it takes PIL images",
    v2.ToPILImage: "it produces PIL images",
    v2.ToTensor: "it takes PIL images or ndarrays (deprecated in torchvision)",
}
_STATIC_TYPES: set[type[Transform]] = {
    v2.Resize,
    v2.CenterCrop,
    v2.Pad,
    v2.Normalize,
    v2.ToDtype,
    v2.Grayscale,
    v2.RGB,
    v2.ConvertImageDtype,
}


class _Step(NamedTuple):
    """One compiled transform: ``kind`` picks the runner in ``_RUNNERS``.

    ``arg`` is kind-specific (a transform, a ValueError, a probability, ...);
    ``name``/``kwargs`` are set for ``static`` and ``children`` for containers.
    """

    kind: str
    arg: Any = None
    name: str = ""
    kwargs: dict[str, Any] | None = None
    children: tuple[_Step, ...] = ()


_Image: TypeAlias = cvcuda.Tensor | _Img | _Passive
_Value: TypeAlias = _Image | tuple[_Image, ...]
_Output = TypeVar("_Output")


def _reject(t: object, why: str) -> NoReturn:
    raise TypeError(f"cvcuda.tv.Compose does not support {type(t).__name__}: {why}")


def _resolve(t: Transform, logical: type[torch.Tensor]) -> Transform | _Step:
    """Copy of ``t`` with its per-type fill/dtype dict resolved for ``logical``."""
    t = copy.copy(t)
    if isinstance(getattr(t, "_fill", None), dict):
        t._fill = {"others": _get_fill(t._fill, logical)}
    if type(t) is v2.ToDtype and isinstance(t.dtype, dict):
        if logical in t.dtype:
            t.dtype = t.dtype[logical]
        elif "others" in t.dtype:
            t.dtype = t.dtype["others"]
        else:
            return _Step(
                "raise", ValueError(f"No dtype was specified for type {logical}.")
            )
    return t


def _has_dict(t: object) -> bool:
    fill = getattr(t, "_fill", None)
    return (isinstance(fill, dict) and set(fill) != {"others"}) or (
        type(t) is v2.ToDtype and isinstance(t.dtype, dict)
    )


def _static(t: Transform) -> _Step:
    """Deterministic transform -> a ``static`` step, or a no-op ``type`` step."""
    match type(t):
        case v2.Resize:
            name, kwargs = "resize", {
                "size": t.size,
                "interpolation": t.interpolation,
                "max_size": t.max_size,
                "antialias": t.antialias,
            }
        case v2.CenterCrop:
            name, kwargs = "center_crop", {"output_size": t.size}
        case v2.Pad:
            name, kwargs = "pad", {
                "padding": t.padding,
                "fill": t._fill["others"],
                "padding_mode": t.padding_mode,
            }
        case v2.Normalize:
            name, kwargs = "normalize", {
                "mean": t.mean,
                "std": t.std,
                "inplace": t.inplace,
            }
        case v2.ToDtype if t.dtype is None:
            return _Step("type", v2.Identity)
        case v2.ToDtype:
            name, kwargs = "to_dtype", {"dtype": t.dtype, "scale": t.scale}
        case v2.Grayscale:
            name, kwargs = "rgb_to_grayscale", {
                "num_output_channels": t.num_output_channels
            }
        case v2.RGB:
            name, kwargs = "grayscale_to_rgb", {}
        case _:  # v2.ConvertImageDtype
            name, kwargs = "to_dtype", {"dtype": t.dtype, "scale": True}
    return _Step("static", name=name, kwargs=kwargs)


def _children(t: Transform) -> tuple[_Step, ...]:
    return tuple(_compile(c) for c in t.transforms)


def _compile_custom(t: Transform) -> _Step:
    """A user ``v2.Transform`` subclass that keeps TorchVision's ``forward``."""
    if type(t).transform is Transform.transform:
        _reject(t, "it does not implement transform()")
    f = type(t).forward
    if f is not Transform.forward and f is not _RandomApplyTransform.forward:
        _reject(t, "it overrides forward(); override make_params()/transform() instead")
    dummy = torch.Tensor._make_wrapper_subclass(
        _Img, (1, 1, 1), dtype=torch.uint8, device="cpu"
    )
    # TorchVision's _needs_transform_list for a lone image.
    needs = check_type(dummy, t._transformed_types)
    kind = "rapply" if f is _RandomApplyTransform.forward else "base"
    return _Step(kind, t, kwargs={"needs": needs})


def _compile(t: Transform | _Step) -> _Step:
    match t:
        case _Step():  # a "raise" step from _resolve
            return t
        case _ if _has_dict(t):
            return _Step(
                "bytype",
                children=(
                    _compile(_resolve(t, torch.Tensor)),
                    _compile(_resolve(t, tv_tensors.Image)),
                ),
            )
        case _ if type(t) in _STATIC_TYPES:
            return _static(t)
        case v2.Compose():
            return _Step("seq", children=_children(t))
        case v2.RandomApply():
            return _Step("apply", t.p, children=_children(t))
        case v2.RandomChoice():
            return _Step("choice", torch.tensor(t.p), children=_children(t))
        case v2.RandomOrder():
            return _Step("order", len(t.transforms), children=_children(t))
        case _ if isinstance(t, tuple(_UNSUPPORTED)):
            _reject(
                t, next(why for cls, why in _UNSUPPORTED.items() if isinstance(t, cls))
            )
        case _ if isinstance(t, _TYPE_ONLY):
            return _Step("type", type(t))
        case _ if isinstance(t, _OWN_FORWARD_OK):
            return _Step("call", t)
        case Transform():
            return _compile_custom(t)
        case _:
            raise TypeError(
                "cvcuda.tv.Compose only accepts torchvision.transforms.v2 transforms, "
                f"got {type(t).__name__}"
            )


def _check(out: _Output, t: Transform) -> _Output:
    # tv_tensors.wrap(fresh, like=proxy) builds a proxy without an image.
    for e in out if isinstance(out, (tuple, list)) else (out,):
        if type(e) is not cvcuda.Tensor and getattr(e, "_cv", None) is None:
            raise CvcudaComposeError(
                f"cvcuda.tv.Compose: {type(t).__name__} returned a {type(e).__name__} "
                "that no CV-CUDA kernel produced; transforms inside the pipeline may "
                "only use torchvision.transforms.v2.functional ops"
            )
    return out


def _run_all(steps: tuple[_Step, ...], x: _Value) -> _Value:
    for s in steps:
        x = _run(s, x)
    return x


def _run_static(step: _Step, x: _Value) -> _Value:
    if isinstance(x, tuple):
        return tuple(e if type(e) is _Passive else _run_static(step, e) for e in x)
    cls = type(x)
    cv = x if cls is cvcuda.Tensor else x._cv
    out = getattr(tv, step.name)(cv, stream=_TLS.stream, **step.kwargs)
    return out if cls is cvcuda.Tensor else _proxy(out, cls)


def _run_bytype(step: _Step, x: _Value) -> _Value:
    return _run(step.children[0] if _TLS.pure else step.children[1], x)


def _run_raise(step: _Step, x: _Value) -> NoReturn:
    raise step.arg


def _run_transform(step: _Step, x: _Value) -> _Value:
    t = step.arg
    if isinstance(x, tuple):  # after FiveCrop/TenCrop: TorchVision's pytree path
        return _check(t(x), t)
    t.check_inputs([x])
    if step.kind == "rapply" and torch.rand(1) >= t.p:
        return x
    if not step.kwargs["needs"]:
        t.make_params([])  # TorchVision still samples with no inputs
        return x
    return _check(t.transform(x, t.make_params([x])), t)


def _run_seq(step: _Step, x: _Value) -> _Value:
    return _run_all(step.children, x)


def _run_apply(step: _Step, x: _Value) -> _Value:
    if torch.rand(1) >= step.arg:
        return x
    return _run_all(step.children, x)


def _run_choice(step: _Step, x: _Value) -> _Value:
    return _run(step.children[int(torch.multinomial(step.arg, 1))], x)


def _run_order(step: _Step, x: _Value) -> _Value:
    for i in torch.randperm(step.arg):
        x = _run(step.children[i], x)
    return x


def _run_call(step: _Step, x: _Value) -> _Value:
    return _check(step.arg(x), step.arg)


def _run_type(step: _Step, x: _Value) -> _Value:
    if step.arg is v2.ToPureTensor:
        _TLS.pure = True
        if isinstance(x, tuple):
            x = tuple(
                _proxy(e._cv, _Img if i == 0 else _Passive) for i, e in enumerate(x)
            )
    elif step.arg is v2.ToImage:
        _TLS.pure = False
        if isinstance(x, tuple):
            x = tuple(_proxy(e._cv) for e in x)
    return x


# Kinds that run on a raw cvcuda.Tensor; every other kind needs a proxy image.
_RAW_RUNNERS: dict[str, Callable[[_Step, _Value], _Value]] = {
    "static": _run_static,
    "bytype": _run_bytype,
    "raise": _run_raise,
}
_RUNNERS: dict[str, Callable[[_Step, _Value], _Value]] = {
    "base": _run_transform,
    "rapply": _run_transform,
    "seq": _run_seq,
    "apply": _run_apply,
    "choice": _run_choice,
    "order": _run_order,
    "call": _run_call,
    "type": _run_type,
}


def _run(step: _Step, x: _Value) -> _Value:
    raw = _RAW_RUNNERS.get(step.kind)
    if raw is not None:
        return raw(step, x)
    if type(x) is cvcuda.Tensor:
        x = _proxy(x)
    return _RUNNERS[step.kind](step, x)


def _cv_of(e: _Image) -> cvcuda.Tensor:
    return e if type(e) is cvcuda.Tensor else e._cv


def _device_of(inpt: object, was_cv: bool) -> torch.device:
    if was_cv:
        return torch.device("cuda", inpt.cuda().__dlpack_device__()[1])
    if not isinstance(inpt, torch.Tensor):
        raise TypeError(
            "cvcuda.tv.Compose takes a cvcuda.Tensor or a CUDA torch.Tensor, "
            f"got {type(inpt).__name__}"
        )
    if not inpt.is_cuda:
        raise TypeError(
            f"cvcuda.tv.Compose needs a CUDA tensor, got one on {inpt.device}"
        )
    return inpt.device


def _export(
    out: cvcuda.Tensor | tuple[cvcuda.Tensor, ...], pure: bool
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    out = _torch._to_torch(out)
    if pure:
        return out
    if isinstance(out, tuple):
        return tuple(tv_tensors.Image(o) for o in out)
    return tv_tensors.Image(out)


class Compose:
    """Run a TorchVision v2 transform pipeline on CV-CUDA.

    Each transform is resolved once at construction. A call imports the image
    once, runs every transform on CV-CUDA in order, and exports the result
    once (one export per crop after ``FiveCrop``/``TenCrop``). Random
    transforms draw their parameters exactly as ``v2.Compose`` does, so
    ``torch.manual_seed`` gives the same choices. Transforms CV-CUDA cannot run
    raise ``TypeError`` at construction.

    Args:
        transforms: A non-empty list or tuple of
            ``torchvision.transforms.v2`` transforms.

    Calling the pipeline takes a ``cvcuda.Tensor`` (``HWC``, ``NHWC``,
    ``CHW`` or ``NCHW``) or a CUDA ``torch.Tensor``/``tv_tensors.Image`` with
    3 or 4 dimensions and non-zero strides, and returns the same kind: a
    ``cvcuda.Tensor`` in the input's layout, a ``torch.Tensor``, or a
    ``tv_tensors.Image``. The work runs on the current Torch stream of the
    input's device.
    """

    transforms: list[Transform] | tuple[Transform, ...]
    _steps: tuple[_Step, ...]
    _static: bool

    def __init__(self, transforms: list[Transform] | tuple[Transform, ...]) -> None:
        if not isinstance(transforms, (list, tuple)):
            raise TypeError("Argument transforms should be a sequence of callables")
        if not transforms:
            raise ValueError("Pass at least one transform")
        self.transforms = transforms
        self._steps = tuple(_compile(t) for t in transforms)
        self._static = all(s.kind == "static" for s in self._steps)

    def __reduce__(
        self,
    ) -> tuple[type[Compose], tuple[list[Transform] | tuple[Transform, ...]]]:
        # Steps hold function objects; unpickling recompiles from the transforms.
        return (type(self), (self.transforms,))

    def __call__(
        self, inpt: cvcuda.Tensor | torch.Tensor
    ) -> cvcuda.Tensor | torch.Tensor | tuple[cvcuda.Tensor | torch.Tensor, ...]:
        was_cv = type(inpt) is cvcuda.Tensor
        dev = _device_of(inpt, was_cv)
        stream = cvcuda.as_stream(torch.cuda.current_stream(dev).cuda_stream)
        if self._static and not isinstance(inpt, tv_tensors.TVTensor):
            cv = inpt if was_cv else _torch._as_cvcuda(inpt)
            for s in self._steps:
                cv = getattr(tv, s.name)(cv, stream=stream, **s.kwargs)
            return cv if was_cv else _torch._to_torch(cv)
        image_in = isinstance(inpt, tv_tensors.Image)
        x = inpt if was_cv else _torch._as_cvcuda(inpt.as_subclass(torch.Tensor))
        saved = _TLS.state
        _TLS.device, _TLS.stream, _TLS.pure = dev, stream, not image_in
        try:
            x = _run_all(self._steps, x)
            pure = _TLS.pure
        finally:
            if saved is not None:
                _TLS.device, _TLS.stream, _TLS.pure = saved
        out = tuple(_cv_of(e) for e in x) if isinstance(x, tuple) else _cv_of(x)
        return out if was_cv else _export(out, pure)


__all__: list[str] = ["Compose"]
