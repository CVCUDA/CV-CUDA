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

.. _imagebatchdecoder_nvcodec:

Image Decoding using nvImageCodec
====================


The image batch decoder is responsible for parsing the input expression, reading and decoding image data. The actual decoding is done in batches using the library `nvImageCodec <https://github.com/NVIDIA/nvImageCodec>`_. Although used in the semantic segmentation sample, this image decoder is generic enough to be used in other applications. The code associated with this class can be found in the ``samples/common/python/nvcodec_utils.py`` file.


Before the data can be read or decoded, we must parse it (i.e figure out what kind of data it is). Depending on the ``input_path``'s value, we either read one image and create a dummy list with the data from the same image to simulate a batch or read a bunch of images from a directory.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_init_imagebatchdecoder_nvimagecodec
   :end-before: end_parse_imagebatchdecoder_nvimagecodec
   :dedent:

Once we have a list of image file names that we can read, we will split them into batches based on the batch size.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_batch_imagebatchdecoder_nvimagecodec
   :end-before: end_init_imagebatchdecoder_nvimagecodec
   :dedent:

That is all we need to do for the initialization. Now as soon as a call to decoder is issued, we would start reading and decoding the data. This begins with reading the data bytes in batches and returning None if there is no data left to be read.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_call_imagebatchdecoder_nvimagecodec
   :end-before: end_read_imagebatchdecoder_nvimagecodec
   :dedent:

Once the data has been read, we use ``nvImageCodec`` to decode it into a list of image tensors. The nvImageCodec instance is allocated either on its first use or whenever there is a change in the batch size (i.e. last batch). Since what we get at this point is a list of images (i.e a python list of 3D tensors), we would need to convert them to a 4D tensor by stacking them up on the first dimension.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_decode_imagebatchdecoder_nvimagecodec
   :end-before: end_decode_imagebatchdecoder_nvimagecodec
   :dedent:

The final step is to pack all of this data into a special CVCUDA samples object called as ``Batch``. The ``Batch`` object helps us keep track of the data associated with the batch, the index of the batch and optionally any filename information one wants to attach (i.e. which files the data came from).

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_return_imagebatchdecoder_nvimagecodec
   :end-before: end_return_imagebatchdecoder_nvimagecodec
   :dedent:
