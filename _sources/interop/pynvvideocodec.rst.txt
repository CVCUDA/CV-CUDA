..
  # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

PyNvVideoCodec
--------------

PyNvVideoCodec provides Python bindings to NVIDIA's hardware-accelerated video codec APIs (NVDEC/NVENC).
It enables efficient video decoding and encoding directly on GPU, perfect for video processing pipelines
with CV-CUDA.

**Key Points:**

* Hardware-accelerated video decoding (H.264, NV12, etc.)
* Decodes directly to GPU memory
* Batch frame decoding for better throughput
* Hardware-accelerated encoding for output
* Minimal CPU involvement in the entire pipeline

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/pynvvideocodec_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**Setup PyNvVideoCodec:**

.. literalinclude:: ../../../samples/interoperability/pynvvideocodec_interop.py
   :language: python
   :start-after: docs_tag: begin_init_pynvvideocodec
   :end-before: docs_tag: end_init_pynvvideocodec
   :dedent: 4

The decoder is configured to output RGB frames directly to device memory. The encoder is set up to
accept NV12 pixel format frames at the target resolution (640×480).

**Read video frames into CV-CUDA tensors and process:**

.. literalinclude:: ../../../samples/interoperability/pynvvideocodec_interop.py
   :language: python
   :start-after: docs_tag: begin_read_and_process_video
   :end-before: docs_tag: end_read_and_process_video
   :dedent: 4

Frames are decoded in batches for efficiency. Each frame is converted to a CV-CUDA tensor with HWC
(Height × Width × Channels) layout.

CV-CUDA operations can be applied to each frame. Here we resize from 1920×1080 to 640×480 and convert
from BGR to NV12 (YUV 4:2:0 planar) to match the encoder's expected format.
Alternatively, you could use :py:func:`cvcuda.stack` to stack the frames into a single tensor and
apply operations on the stacked tensor. For clarity, we process each frame individually here.

**Encode processed frames:**

.. literalinclude:: ../../../samples/interoperability/pynvvideocodec_interop.py
   :language: python
   :start-after: docs_tag: begin_encode_frames
   :end-before: docs_tag: end_encode_frames
   :dedent: 4

The processed frames are encoded back to video. The encoder produces compressed bitstreams that are
written to the output file. Note that ``EndEncode()`` must be called to flush any remaining frames.

**Typical Use Cases:**

* Video analytics preprocessing
* Real-time video transformation
* Video transcoding pipelines
* Video quality enhancement
* Object detection/segmentation on video

**Complete Example:** See ``samples/interoperability/pynvvideocodec_interop.py``
