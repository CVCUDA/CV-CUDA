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

"""Functional image operations backed by CV-CUDA.

``cvcuda.tv`` provides an alternative functional path for code familiar with
``torchvision.transforms.v2.functional``; it is not a replacement for
TorchVision. Functions accept interleaved ``HWC``/``NHWC`` and planar
``CHW``/``NCHW`` CV-CUDA tensors, plus an optional keyword-only ``stream``
argument. See each function's docstring for supported inputs.

Classes:
    ``Compose``

Functions:
    Augmentation:
        ``erase``, ``jpeg``
    Color:
        ``adjust_brightness``, ``adjust_contrast``, ``adjust_gamma``,
        ``adjust_hue``, ``adjust_saturation``, ``adjust_sharpness``,
        ``autocontrast``, ``equalize``, ``grayscale_to_rgb``, ``invert``,
        ``permute_channels``, ``posterize``, ``rgb_to_grayscale``, ``solarize``
    Geometry:
        ``affine``, ``center_crop``, ``crop``, ``elastic``, ``five_crop``,
        ``horizontal_flip``, ``pad``, ``perspective``, ``resize``,
        ``resized_crop``, ``rotate``, ``ten_crop``, ``vertical_flip``
    Miscellaneous:
        ``gaussian_blur``, ``gaussian_noise``, ``normalize``, ``to_dtype``
"""

from ._augment import erase, jpeg
from ._torch import accept_torch as _accept_torch
from ._color import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    adjust_hue,
    adjust_saturation,
    adjust_sharpness,
    autocontrast,
    equalize,
    grayscale_to_rgb,
    invert,
    permute_channels,
    posterize,
    rgb_to_grayscale,
    solarize,
)
from ._geometry import (
    affine,
    center_crop,
    crop,
    elastic,
    five_crop,
    horizontal_flip,
    pad,
    perspective,
    resize,
    resized_crop,
    rotate,
    ten_crop,
    vertical_flip,
)
from ._misc import gaussian_blur, gaussian_noise, normalize, to_dtype

__all__: list[str] = [
    "Compose",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_gamma",
    "adjust_hue",
    "adjust_saturation",
    "adjust_sharpness",
    "affine",
    "autocontrast",
    "center_crop",
    "crop",
    "elastic",
    "equalize",
    "erase",
    "five_crop",
    "gaussian_blur",
    "gaussian_noise",
    "grayscale_to_rgb",
    "horizontal_flip",
    "invert",
    "jpeg",
    "normalize",
    "pad",
    "permute_channels",
    "perspective",
    "posterize",
    "resize",
    "resized_crop",
    "rgb_to_grayscale",
    "rotate",
    "solarize",
    "ten_crop",
    "to_dtype",
    "vertical_flip",
]

# Wrap operators to accept torch.Tensors; Compose handles Torch input itself.
for _name in __all__:
    if _name.startswith("_") or _name == "Compose":
        continue
    globals()[_name] = _accept_torch(globals()[_name])
del _name


def __getattr__(name):
    # Compose imports torch and torchvision; load it on first use so that
    # importing cvcuda.tv needs neither.
    if name == "Compose":
        from ._compose import Compose

        return Compose
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
