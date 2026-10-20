..
  # SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Utilities
=========

.. autofunction:: cvcuda.as_image

.. autofunction:: cvcuda.as_images

.. autofunction:: cvcuda.as_stream

.. note::

   Pass an owning ``torch.cuda.Stream`` or ``cupy.cuda.Stream`` object when
   possible. CV-CUDA retains that owner while the wrapped stream or a resource
   carrying its stream dependency is in use. Raw CUDA stream handles and
   ``ExternalStream`` objects remain caller-owned; their underlying stream must
   stay valid for all CV-CUDA uses, including while resources carry dependencies
   on it.

.. autofunction:: cvcuda.as_tensor

.. autofunction:: cvcuda.as_tensors

.. autofunction:: cvcuda.reshape
