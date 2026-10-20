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

"""Accept ``torch.Tensor`` at the operator boundary.

Without this, substituting ``cvcuda.tv`` for
``torchvision.transforms.v2.functional`` is not a drop-in: the operators take a
layout-tagged ``cvcuda.Tensor``, so a caller holding Torch tensors has to
convert at every call site. Converting here instead is what lets the import be
the only edit.

``torch`` is imported lazily. The module must stay importable without it, since
every operator works on a ``cvcuda.Tensor`` with no Torch involved.
"""

from __future__ import annotations

import functools
import inspect
import sys
from types import SimpleNamespace
from typing import Any

import cvcuda

# CV-CUDA carries the layout on the tensor; Torch carries it in the caller's
# head. Only the two ranks torchvision's image kernels accept are mappable.
_LAYOUTS = {3: "CHW", 4: "NCHW"}


def _wrap(tensor: Any, layout: str | None) -> cvcuda.Tensor:
    import torch

    interface = dict(tensor.__cuda_array_interface__)
    interface.update(
        version=3,
        # The operator runs on this stream. Torch advertises the legacy stream,
        # which CV-CUDA reads as an unknown producer and answers with a
        # device-wide synchronization on every call.
        stream=torch.cuda.current_stream(tensor.get_device()).cuda_stream or -1,
    )
    owner = SimpleNamespace(_tensor=tensor, __cuda_array_interface__=interface)
    return cvcuda.as_tensor(owner, layout)


def _as_cvcuda(tensor: Any) -> cvcuda.Tensor:
    layout = _LAYOUTS.get(tensor.ndim)
    if layout is None:
        raise ValueError(
            f"Argument inpt should be a 3- or 4-dimensional tensor, "
            f"but given {tensor.ndim} dimensions"
        )
    return _wrap(tensor, layout)


# torchvision moves these arguments to the image's device itself (erase through
# indexed assignment, elastic through .to(device)), so a CPU or other-device
# tensor is copied here rather than rejected.
def _erase_value(value: Any, inpt: cvcuda.Tensor, device: Any) -> cvcuda.Tensor:
    import torch

    # torchvision assigns v into the image, so it takes the image's dtype with
    # torch's cast rules; cvcuda.erase's own float32 conversion rounds instead.
    dtype = getattr(torch, inpt.dtype.name)
    value = value.to(device=device, dtype=dtype)
    return _wrap(value.reshape(value.shape or (1,)).contiguous(), None)


def _displacement(value: Any, inpt: cvcuda.Tensor, device: Any) -> cvcuda.Tensor:
    import torch

    value = value.to(device=device, dtype=torch.float32).contiguous()
    return _wrap(value, "NHWC" if value.ndim == 4 else None)


# Tensor-valued arguments other than the image, by operator name.
_TENSOR_ARGUMENTS = {
    "erase": {"v": _erase_value},
    "elastic": {"displacement": _displacement},
}


def _is_torch_tensor(value: Any) -> bool:
    """True only for a real ``torch.Tensor``.

    Deliberately narrower than "not a ``cvcuda.Tensor``": the operators are
    also handed stand-ins whose argument validation has to run and raise on its
    own terms, so anything that is not a Torch tensor is passed through
    untouched. Reads ``sys.modules`` rather than importing, because a caller
    holding a Torch tensor has necessarily already imported Torch.
    """
    if type(value) is cvcuda.Tensor:
        return False
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _to_torch(value: Any) -> Any:
    import torch

    if isinstance(value, cvcuda.Tensor):
        # CAI keeps its Python owner with the Torch storage. The DLPack path
        # currently tears that owner down unsafely during interpreter shutdown.
        buffer = value.cuda()
        tensor = torch.as_tensor(buffer)
        # Some Torch builds (e.g. Jetson nv24.03) copy a CAI object to the CPU.
        if not tensor.is_cuda:
            tensor = tensor.to(torch.device("cuda", buffer.__dlpack_device__()[1]))
        return tensor
    if isinstance(value, (tuple, list)):
        return type(value)(_to_torch(item) for item in value)
    return value


def _prepare_torch_inputs(inpt, args, kwargs, was_torch, tensor_arguments):
    # Non-tensor images fall through so the operator raises its own error.
    extras = [
        (index, name, convert)
        for index, name, convert in tensor_arguments
        if (was_torch or type(inpt) is cvcuda.Tensor)
        and _is_torch_tensor(args[index] if index < len(args) else kwargs.get(name))
    ]
    if not was_torch and not extras:
        return inpt, args, kwargs

    import torch

    device = inpt.get_device() if was_torch else inpt.cuda().__dlpack_device__()[1]
    if kwargs.get("stream") is None:
        # Torch produced the inputs and reads the outputs on its current stream.
        handle = torch.cuda.current_stream(device).cuda_stream
        kwargs["stream"] = cvcuda.as_stream(handle)
    if was_torch:
        inpt = _as_cvcuda(inpt)
    args = list(args)
    for index, name, convert in extras:
        if index < len(args):
            args[index] = convert(args[index], inpt, device)
        else:
            kwargs[name] = convert(kwargs[name], inpt, device)
    return inpt, args, kwargs


def accept_torch(operator):
    """Let ``operator`` take and return ``torch.Tensor`` as well as ``cvcuda.Tensor``.

    The return type follows the input: a caller who passed a ``cvcuda.Tensor``
    keeps getting one back, so this is additive and no existing call changes
    behavior.
    """

    parameters = list(inspect.signature(operator).parameters)
    # Positions are relative to ``args``, which excludes the image.
    tensor_arguments = [
        (parameters.index(name) - 1, name, convert)
        for name, convert in _TENSOR_ARGUMENTS.get(operator.__name__, {}).items()
    ]

    @functools.wraps(operator)
    def wrapper(inpt, *args, **kwargs):
        was_torch = _is_torch_tensor(inpt)
        inpt, args, kwargs = _prepare_torch_inputs(
            inpt, args, kwargs, was_torch, tensor_arguments
        )

        output = operator(inpt, *args, **kwargs)

        if was_torch:
            return _to_torch(output)
        return output

    first_parameter = parameters[0]
    if first_parameter == "inpt":
        return wrapper

    @functools.wraps(operator)
    def named_wrapper(*args, **kwargs):
        if args:
            return wrapper(*args, **kwargs)
        if first_parameter not in kwargs:
            return operator(**kwargs)
        return wrapper(kwargs.pop(first_parameter), **kwargs)

    return named_wrapper


__all__: list[str] = ["accept_torch"]
