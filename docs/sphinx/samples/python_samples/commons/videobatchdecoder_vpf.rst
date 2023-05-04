..
   # SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _videobatchdecoder_vpf:

Video Decoding using VPF
====================


The video batch decoder is responsible for reading an MP4 video as PyTorch tensors. The actual decoding is done per frame using NVIDIA's `Video Processing Framework <https://github.com/NVIDIA/VideoProcessingFramework>`_. The video decoder is generic enough to be used across the sample applications. The code associated with this class can be found in the ``samples/common/python/vpf_utils.py`` file.

There are two classes responsible for the decoding work:

1. ``VideoBatchDecoderVPF`` and
2. ``nvdecoder``

The first class acts as a wrapper on the second class which allows us to:

1. Stay consistent with the API of other decoders used throughout CVCUDA
2. Support batch decoding.
3. Use accelerated ops in CVCUDA to perform the necessary color conversion from NV12 to RGB after decoding the video.


VideoBatchDecoderVPF
------------------

Let's get started by understanding how this class is initialized in its ``__init__`` method. We use VPF's ``PyFFmpegDemuxer`` to read a few properties of the video. The decoder instance and CVCUDA color conversion tensors both are allocated when needed upon the first use.

**Note**: Due to the nature of NV12, representing it directly as a CVCUDA tensor is a bit challenging. Be sure to read through the explanation in the comments of the code shown below to understand more.


.. literalinclude:: ../../../../../samples/common/python/vpf_utils.py
   :language: python
   :linenos:
   :start-after: begin_init_videobatchdecoder_vpf
   :end-before: end_init_videobatchdecoder_vpf
   :dedent:


Once things are defined and initialized, we would start the decoding when a call to the ``__call__`` function is made.

.. literalinclude:: ../../../../../samples/common/python/vpf_utils.py
   :language: python
   :linenos:
   :start-after: begin_call_videobatchdecoder_vpf
   :end-before: end_alloc_videobatchdecoder_vpf
   :dedent:

Next, we call the ``nvdecoder`` instance to actually do the decoding and stack the image tensors up to form a 4D tensor.

.. literalinclude:: ../../../../../samples/common/python/vpf_utils.py
   :language: python
   :linenos:
   :start-after: begin_decode_videobatchdecoder_vpf
   :end-before: end_decode_videobatchdecoder_vpf
   :dedent:

Once the video batch is ready, we use CVCUDA's ``cvtcolor_into`` function to convert its data from NV12 format to RGB format. We will use pre-allocated tensors to do the color conversion to avoid allocating same tensors on every batch.

.. literalinclude:: ../../../../../samples/common/python/vpf_utils.py
   :language: python
   :linenos:
   :start-after: begin_convert_videobatchdecoder_vpf
   :end-before: end_convert_videobatchdecoder_vpf
   :dedent:


The final step is to pack all of this data into a special CVCUDA samples object called as ``Batch``. The ``Batch`` object helps us keep track of the data associated with the batch, the index of the batch and optionally any filename information one wants to attach (i.e. which files did the data come from).

.. literalinclude:: ../../../../../samples/common/python/vpf_utils.py
   :language: python
   :linenos:
   :start-after: begin_batch_videobatchdecoder_vpf
   :end-before: end_batch_videobatchdecoder_vpf
   :dedent:


nvdecoder
------------------

This is a class offering hardware accelerated video decoding functionality using VPF. It reads an MP4 video file, decodes it and returns a 3D PyTorch Tensor per frame. Please consult the documentation of the `Video Processing Framework <https://github.com/NVIDIA/VideoProcessingFramework>`_ to learn more about its capabilities and APIs.

For use in CVCUDA, this class defines the following ``decode_hw`` and ``decode_to_tensor`` functions which decode data to a Torch tensor in a given cuda stream.

.. literalinclude:: ../../../../../samples/common/python/vpf_utils.py
   :language: python
   :linenos:
   :start-after: begin_imp_nvdecoder
   :end-before: end_imp_nvdecoder
   :dedent:
