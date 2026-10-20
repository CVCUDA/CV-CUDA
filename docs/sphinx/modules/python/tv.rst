..
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

TV API
======

.. py:class:: cvcuda.tv.Compose(transforms)

   Run a TorchVision v2 transform pipeline on CV-CUDA. Requires PyTorch and
   TorchVision; the functional API below does not require either dependency
   for ``cvcuda.Tensor`` inputs.

   :param transforms: A non-empty list or tuple of TorchVision v2 transforms.
   :raises TypeError: If a transform is unsupported, including at any nesting
      depth.

   Calling the pipeline accepts a ``cvcuda.Tensor`` in ``HWC``, ``NHWC``,
   ``CHW`` or ``NCHW`` layout, or a CUDA ``torch.Tensor``/``tv_tensors.Image``
   with 3 or 4 dimensions and non-zero strides. It returns the same kind and
   preserves the CV-CUDA layout, with one output per crop for ``FiveCrop`` or
   ``TenCrop``. Operations use the current Torch stream of the input's device.

   See :ref:`cvcuda_tv_pipelines` for supported transforms, usage and performance.

.. cvcuda-autofunction:: cvcuda.tv.adjust_brightness

.. cvcuda-autofunction:: cvcuda.tv.adjust_contrast

.. cvcuda-autofunction:: cvcuda.tv.adjust_gamma

.. cvcuda-autofunction:: cvcuda.tv.adjust_hue

.. cvcuda-autofunction:: cvcuda.tv.adjust_saturation

.. cvcuda-autofunction:: cvcuda.tv.adjust_sharpness

.. cvcuda-autofunction:: cvcuda.tv.affine

.. cvcuda-autofunction:: cvcuda.tv.autocontrast

.. cvcuda-autofunction:: cvcuda.tv.center_crop

.. cvcuda-autofunction:: cvcuda.tv.crop

.. cvcuda-autofunction:: cvcuda.tv.elastic

.. cvcuda-autofunction:: cvcuda.tv.equalize

.. cvcuda-autofunction:: cvcuda.tv.erase

.. cvcuda-autofunction:: cvcuda.tv.five_crop

.. cvcuda-autofunction:: cvcuda.tv.gaussian_blur

.. cvcuda-autofunction:: cvcuda.tv.gaussian_noise

.. cvcuda-autofunction:: cvcuda.tv.grayscale_to_rgb

.. cvcuda-autofunction:: cvcuda.tv.horizontal_flip

.. cvcuda-autofunction:: cvcuda.tv.invert

.. cvcuda-autofunction:: cvcuda.tv.jpeg

.. cvcuda-autofunction:: cvcuda.tv.normalize

.. cvcuda-autofunction:: cvcuda.tv.pad

.. cvcuda-autofunction:: cvcuda.tv.permute_channels

.. cvcuda-autofunction:: cvcuda.tv.perspective

.. cvcuda-autofunction:: cvcuda.tv.posterize

.. cvcuda-autofunction:: cvcuda.tv.resize

.. cvcuda-autofunction:: cvcuda.tv.resized_crop

.. cvcuda-autofunction:: cvcuda.tv.rgb_to_grayscale

.. cvcuda-autofunction:: cvcuda.tv.rotate

.. cvcuda-autofunction:: cvcuda.tv.solarize

.. cvcuda-autofunction:: cvcuda.tv.ten_crop

.. cvcuda-autofunction:: cvcuda.tv.to_dtype

.. cvcuda-autofunction:: cvcuda.tv.vertical_flip
