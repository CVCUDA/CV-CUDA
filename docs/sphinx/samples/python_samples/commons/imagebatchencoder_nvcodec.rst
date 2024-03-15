..
   # SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _imagebatchencoder_nvcodec:

Image Encoding using nvImageCodec
====================


The image batch encoder is responsible for saving image tensors to the disk as JPG images. The actual encoding is done in batches using the `nvImageCodec <https://github.com/NVIDIA/nvImageCodec>`_ library. The image encoder is generic enough to be across the sample applications. The code associated with this class can be found in the ``samples/common/python/nvcodec_utils.py`` file.

The image batch encoder is a relatively simple class. Here is how its ``__init__`` method is defined.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_init_imagebatchencoder_nvimagecodec
   :end-before: end_init_imagebatchencoder_nvimagecodec
   :dedent:

Once the initialization is complete, we encode the images in the ``__call__`` method. Since the ``Batch`` object is passed, we have information of the data, its batch index and the original file name used to read the data.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_call_imagebatchencoder_nvimagecodec
   :end-before: end_call_imagebatchencoder_nvimagecodec
   :dedent:
