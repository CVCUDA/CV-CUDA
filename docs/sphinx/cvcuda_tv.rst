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

.. _cvcuda_tv:

``cvcuda.tv`` Functional API
============================

``cvcuda.tv`` provides a TorchVision-style functional path for image augmentation
and preprocessing with CV-CUDA tensors. It is an alternative for pipelines that
already use CV-CUDA tensors, not a replacement for TorchVision. Its function names,
signatures, and semantics track ``torchvision.transforms.v2.functional`` where
CV-CUDA provides the required operation. TorchVision is not required to use the
functions; only ``cvcuda.tv.Compose`` needs it.

Callers chain the functions directly (see the table below) or run a TorchVision
v2 transform pipeline with ``cvcuda.tv.Compose`` (see
:ref:`cvcuda_tv_pipelines`).

Using the API
-------------

Access the submodule through the installed ``cvcuda`` package. The equivalent
functional chains have the same composition:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - TorchVision
     - ``cvcuda.tv``
   * - .. code-block:: python

          import torch
          from torchvision.transforms.v2 import functional as F

          output = F.normalize(
              F.to_dtype(F.resize(torch_image, [224, 224]), torch.float32, scale=True),
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
          )
     - .. code-block:: python

          import cvcuda
          import numpy as np

          output = cvcuda.tv.normalize(
              cvcuda.tv.to_dtype(
                  cvcuda.tv.resize(cvcuda_image, [224, 224]), np.float32, scale=True
              ),
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
          )

Inputs are fixed-shape ``cvcuda.Tensor`` values with interleaved
``HWC``/``NHWC`` or planar ``CHW``/``NCHW`` layouts. Supported dtypes and
channel counts vary by operation.

Functions also accept a CUDA ``torch.Tensor`` in ``CHW``/``NCHW`` order and then
return ``torch.Tensor``. The tensor arguments of ``erase`` (``v``) and ``elastic``
(``displacement``) may be ``torch.Tensor`` too; like torchvision, they are moved
to the image's device, so a CPU tensor is copied. ``elastic`` takes torchvision's
relative displacement of shape ``(1, H, W, 2)`` and adds the identity sampling
grid itself.

``resize`` and the resize step of ``resized_crop`` use :func:`cvcuda.tvresize`
for nearest, nearest-exact, bilinear and bicubic interpolation, with or without
antialiasing. Legacy nearest maps to ``cvcuda.Interp.NEAREST_FLOOR``; nearest-exact
maps to ``cvcuda.Interp.NEAREST``. Resizing supports uint8, uint16, int16, float16
and float32, including two-channel tensors. A no-op ``resize`` returns the input
unchanged. PIL-only box, hamming and lanczos modes are rejected, matching
torchvision's tensor API. ``resized_crop`` also retains the crop operator's
input restrictions.

``autocontrast`` on integer images returns ``floor((x - min) * bound / (max - min))``,
which matches Pillow's ``ImageOps.autocontrast``. TorchVision's tensor kernel
divides in float32 and can be one lower where that quotient is a whole number.

Operations are asynchronous. By default they use the current CV-CUDA stream;
pass ``stream=`` to select one explicitly and synchronize only before a consumer
requires completed data:

.. code-block:: python

   stream = cvcuda.Stream()
   output = cvcuda.tv.gaussian_blur(image, 5, stream=stream)
   stream.sync()

.. _cvcuda_tv_pipelines:

TorchVision pipelines
---------------------

``cvcuda.tv.Compose`` runs an existing ``torchvision.transforms.v2`` pipeline on
CV-CUDA. Pass it the same list of transform objects you would pass to
``v2.Compose``:

.. code-block:: python

   import cvcuda
   import torch
   from torchvision.transforms import v2

   pipeline = cvcuda.tv.Compose([
       v2.RandomResizedCrop(224),
       v2.RandomHorizontalFlip(),
       v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
       v2.ToDtype(torch.float32, scale=True),
       v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])
   torch.manual_seed(0)
   output = pipeline(image)  # CUDA torch.Tensor, CHW or NCHW

When to use it
^^^^^^^^^^^^^^

Use ``Compose`` when an augmentation or preprocessing pipeline is already written
with TorchVision v2 transform classes and the images are CUDA tensors. It keeps
the pipeline definition as it is, including random transforms and containers
such as ``RandomApply`` and ``RandomChoice``. Random transforms draw their
parameters as ``v2.Compose`` does, so ``torch.manual_seed`` selects the same
crops, flips and factors.

Performance
^^^^^^^^^^^

Calling the ``cvcuda.tv`` functions one by one on a ``torch.Tensor`` converts the
image to a ``cvcuda.Tensor`` and back on every call. ``Compose`` converts once on
input and once on output, and passes ``cvcuda.Tensor`` intermediates between the
transforms. A pipeline made only of deterministic transforms, such as ``Resize``,
``CenterCrop``, ``ToDtype`` and ``Normalize``, runs as a direct sequence of
CV-CUDA operations.

Inputs and outputs
^^^^^^^^^^^^^^^^^^

The input can be a ``cvcuda.Tensor`` in ``HWC``, ``NHWC``, ``CHW`` or ``NCHW``
layout, or a CUDA ``torch.Tensor`` or ``tv_tensors.Image`` with 3 or 4
dimensions. The output has the same kind: a ``cvcuda.Tensor`` in the input's
layout, a ``torch.Tensor``, or a ``tv_tensors.Image``, with one output per crop
after ``FiveCrop`` or ``TenCrop``. Inputs must have non-zero strides, so call
``.contiguous()`` on an expanded tensor first. The work runs on the current Torch
stream of the input's device.

Supported transforms
^^^^^^^^^^^^^^^^^^^^

Transforms are checked when the pipeline is built. A transform that CV-CUDA
cannot run raises ``TypeError`` at construction, at any nesting depth, rather
than falling back to TorchVision.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Category
     - Transforms
   * - Supported
     - ``AutoAugment``, ``CenterCrop``, ``ColorJitter``, ``Compose``,
       ``ConvertImageDtype``, ``ElasticTransform``, ``FiveCrop``,
       ``GaussianBlur``, ``GaussianNoise``, ``Grayscale``, ``Identity``,
       ``JPEG``, ``Normalize``, ``Pad``, ``RGB``, ``RandAugment``,
       ``RandomAdjustSharpness``, ``RandomAffine``, ``RandomApply``,
       ``RandomAutocontrast``, ``RandomChannelPermutation``, ``RandomChoice``,
       ``RandomCrop``, ``RandomEqualize``, ``RandomErasing``,
       ``RandomGrayscale``, ``RandomHorizontalFlip``, ``RandomInvert``,
       ``RandomOrder``, ``RandomPerspective``, ``RandomPhotometricDistort``,
       ``RandomPosterize``, ``RandomResize``, ``RandomResizedCrop``,
       ``RandomRotation``, ``RandomShortestSize``, ``RandomSolarize``,
       ``RandomVerticalFlip``, ``RandomZoomOut``, ``Resize``, ``ScaleJitter``,
       ``TenCrop``, ``ToDtype``, ``ToImage``, ``ToPureTensor``,
       ``TrivialAugmentWide``
   * - Unsupported: run arbitrary or Torch-arithmetic code
     - ``Lambda``, ``LinearTransformation``, ``AugMix``
   * - Unsupported: mix batches and labels
     - ``CutMix``, ``MixUp``
   * - Unsupported: bounding boxes, keypoints, videos
     - ``ClampBoundingBoxes``, ``ClampKeyPoints``, ``ConvertBoundingBoxFormat``,
       ``RandomIoUCrop``, ``SanitizeBoundingBoxes``, ``SanitizeKeyPoints``,
       ``SetClampingMode``, ``UniformTemporalSubsample``
   * - Unsupported: PIL images or ndarrays
     - ``PILToTensor``, ``ToPILImage``, ``ToTensor``

Per-type ``fill`` and ``dtype`` dictionaries are supported and are selected by
the input's type, including after ``ToImage`` or ``ToPureTensor``.

User subclasses of ``v2.Transform`` that override ``make_params`` or
``transform`` work if ``transform`` only calls
``torchvision.transforms.v2.functional`` operations that ``cvcuda.tv``
implements. Inside the pipeline the image has no Torch storage, so Torch
arithmetic, indexing or a freshly allocated result raises ``RuntimeError``.
A subclass that overrides ``forward`` is rejected at construction.

See the :ref:`cvcuda.tv samples <sample_cvcuda_tv>` for runnable examples and
the :doc:`TV API reference <modules/python/tv>` for the full operator list.
