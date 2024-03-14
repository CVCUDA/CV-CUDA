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

.. _videobatchencoder_pyvideocodec:

Video Encoding using VpyNvVideoCodecPF
====================


The video batch encoder is responsible for writing tensors as an MP4 video. The actual encoding is done in batches using NVIDIA's pyNvVideoCodec. The video encoder is generic enough to be used across the sample applications. The code associated with this class can be found in the ``samples/common/python/nvcodec_utils.py`` file.

There are two classes responsible for the encoding work:

1. ``VideoBatchEncoder`` and
2. ``nvVideoEncoder``

The first class acts as a wrapper on the second class which allows us to:

1. Stay consistent with the API of other encoders used throughout CVCUDA
2. Support batch encoding.
3. Use accelerated ops in CVCUDA to perform the necessary color conversion from RGB to NV12 before encoding the video.


VideoBatchEncoderVPF
------------------

To get started, here is how the class is initialized in its ``__init__`` method. The encoder instance and CVCUDA color conversion tensors both are allocated when needed upon the first use.

**Note**: Due to the nature of NV12, representing it directly as a CVCUDA tensor is a bit challenging. Be sure to read through the explanation in the comments of the code shown below to understand more.


.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_init_videobatchencoder_pyvideocodec
   :end-before: end_init_videobatchencoder_pyvideocodec
   :dedent:


Once things are defined and initialized, we would start the decoding when a call to the ``__call__`` function is made. We need to first allocate the encoder instance if it wasn't done so already.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_call_videobatchencoder_pyvideocodec
   :end-before: end_alloc_videobatchdecoder_pyvideocodec
   :dedent:

Next, we use CVCUDA's ``cvtcolor_into`` function to convert the batch data from RGB format to NV12 format. We allocate tensors once to do the color conversion and avoid allocating same tensors on every batch.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_alloc_cvcuda_videobatchdecoder_pyvideocodec
   :end-before: end_alloc_cvcuda_videobatchdecoder_pyvideocodec
   :dedent:


Once the tensors are allocated, we use CVCUDA ops to perform the color conversion.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_convert_videobatchencoder_pyvideocodec
   :end-before: end_convert_videobatchencoder_pyvideocodec
   :dedent:


Finally, we call the ``nvVideoEncoder`` instance to actually do the encoding.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_encode_videobatchencoder_nvvideoencoder
   :end-before: end_encode_videobatchencoder_nvvideoencoder
   :dedent:


nvVideoEncoder
------------------

This is a class offering hardware accelerated video encoding functionality using pyNvVideoCodec. It encodes tensors and writes as an MP4 file. Please consult the documentation of the pyNvVideoCodec to learn more about its capabilities and APIs.

For use in CVCUDA, this class defines the following ``encode_from_tensor`` functions which encode a Torch tensor.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_imp_nvvideoencoder
   :end-before: end_imp_nvvideoencoder
   :dedent:

Finally, we use the ``av`` library to write packets to an MP4 container. We must properly flush (i.e. write any pending packets) at the end.

.. literalinclude:: ../../../../../samples/common/python/nvcodec_utils.py
   :language: python
   :linenos:
   :start-after: begin_writeframe_nvvideoencoder
   :end-before: end_writeframe_nvvideoencoder
   :dedent:
