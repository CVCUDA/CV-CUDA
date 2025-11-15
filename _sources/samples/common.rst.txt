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

.. _sample_common:

Utilities
---------

The ``common.py`` module provides utilities for:

* **Image I/O** - GPU-accelerated reading and writing
* **CUDA Memory** - Host-device memory transfers
* **TensorRT** - Model inference wrapper
* **Model Export** - PyTorch to ONNX to TensorRT

All samples import from this module to avoid code duplication.

Module Location
---------------

File: ``samples/common.py``

.. code-block:: python

   from common import (
        read_image,
        write_image,
        TRT,
        cuda_memcpy_h2d,
        cuda_memcpy_d2h,
        zero_copy_split,
        parse_image_args,
        get_cache_dir,
        engine_from_onnx,
        export_classifier_onnx,
        export_retinanet_onnx,
        export_segmentation_onnx,
   )

Image I/O Functions
-------------------

.. _common_read_image:

read_image()
^^^^^^^^^^^^

.. code-block:: python

   def read_image(path: Path) -> cvcuda.Tensor
       # path: Path to input image file (JPG, PNG, etc.)
       # Returns: CV-CUDA tensor in HWC layout, uint8 data type

Load an image from disk directly into GPU memory using nvImageCodec for GPU-accelerated decoding.

**Example:**

.. code-block:: python

   from common import read_image

   image = read_image(Path("input.jpg"))
   print(image.shape)  # (H, W, 3)
   print(image.dtype)  # uint8

.. _common_write_image:

write_image()
^^^^^^^^^^^^^

.. code-block:: python

   def write_image(tensor: cvcuda.Tensor, path: Path) -> None
       # tensor: CV-CUDA tensor in HWC layout
       # path: Output file path (format determined by extension: .jpg, .png, etc.)

Save a CV-CUDA tensor as an image file using nvImageCodec.

**Example:**

.. code-block:: python

   from common import write_image

   write_image(processed_image, Path("output.jpg"))

CUDA Memory Operations
-----------------------

.. _common_cuda_memcpy_h2d:

cuda_memcpy_h2d()
^^^^^^^^^^^^^^^^^

.. code-block:: python

   def cuda_memcpy_h2d(
       host_array: np.ndarray,              # NumPy array on CPU
       device_array: int | dict | object    # GPU pointer or CV-CUDA tensor
   ) -> None

Copy data from CPU (host) memory to GPU (device) memory.

**Example:**

.. code-block:: python

   # Upload normalization parameters
   mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
   mean_tensor = cvcuda.Tensor((3,), np.float32)
   cuda_memcpy_h2d(mean, mean_tensor.cuda())

.. _common_cuda_memcpy_d2h:

cuda_memcpy_d2h()
^^^^^^^^^^^^^^^^^

.. code-block:: python

   def cuda_memcpy_d2h(
       device_array: int | dict | object,   # GPU pointer or CV-CUDA tensor
       host_array: np.ndarray               # NumPy array on CPU (pre-allocated)
   ) -> None

Copy data from GPU (device) memory to CPU (host) memory.

**Example:**

.. code-block:: python

   # Download inference results
   output = np.zeros((1, 1000), dtype=np.float32)
   cuda_memcpy_d2h(output_tensor.cuda(), output)

   # Now process on CPU
   top_classes = np.argsort(output[0])[::-1][:5]

Tensor Utilities
----------------

.. _common_zero_copy_split:

zero_copy_split()
^^^^^^^^^^^^^^^^^

.. code-block:: python

   def zero_copy_split(batch: cvcuda.Tensor) -> list[cvcuda.Tensor]
       # batch: Batched tensor with shape (N, ...) where N is batch size
       # Returns: List of N tensors, each representing one item from the batch

Split a batched tensor into individual tensors without copying data (creates views into original memory).

**Example:**

.. code-block:: python

   # Stack images
   batch = cvcuda.stack([img1, img2, img3])  # Shape: (3, H, W, C)

   # Process batch
   processed = cvcuda.gaussian(batch, (5, 5), (1.0, 1.0))

   # Split back to individual images
   images = zero_copy_split(processed)  # List of 3 tensors
   for img in images:
       print(img.shape)  # (H, W, C)

Argument Parsing
----------------

.. _common_parse_image_args:

parse_image_args()
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def parse_image_args(default_output: str = "output.jpg") -> argparse.Namespace
       # default_output: Default output filename
       # Returns: Namespace with input, output, width, height attributes

Parse command-line arguments for image processing samples (``--input``, ``--output``, ``--width``, ``--height``).

**Example:**

.. code-block:: python

   args = parse_image_args("processed.jpg")
   input_image = read_image(args.input)
   # ... process ...
   write_image(result, args.output)

TensorRT Integration
--------------------

.. _common_trt:

TRT Class
^^^^^^^^^

.. code-block:: python

   class TRT:
       def __init__(self, engine_path: Path)
           # engine_path: Path to serialized TensorRT engine file (.trtmodel)

       def __call__(self, inputs: list[cvcuda.Tensor]) -> list[cvcuda.Tensor]
           # inputs: List of CV-CUDA tensors matching engine's expected inputs
           # Returns: List of CV-CUDA tensors containing inference results

Wrapper class for TensorRT engine inference with CV-CUDA tensor support via ``__cuda_array_interface__``.

**Example:**

.. code-block:: python

   # Load TensorRT engine
   from common import get_cache_dir
   model = TRT(get_cache_dir() / "resnet50.trtmodel")

   # Run inference
   input_tensors = [preprocessed_image]
   output_tensors = model(input_tensors)

   # Access results
   logits = output_tensors[0]

.. _common_engine_from_onnx:

engine_from_onnx()
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def engine_from_onnx(
       onnx_path: Path,           # Path to ONNX model file
       engine_path: Path,         # Path where TensorRT engine will be saved
       use_fp16: bool = True,     # Enable FP16 precision
       max_batch_size: int = 1    # Maximum batch size to support
   ) -> None

Build a TensorRT engine from an ONNX model with optimizations (FP16, layer fusion, etc.).

**Example:**

.. code-block:: python

   engine_from_onnx(
       Path("model.onnx"),
       Path("model.trtmodel"),
       use_fp16=True
   )

Model Export Functions
----------------------

.. _common_export_classifier_onnx:

export_classifier_onnx()
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def export_classifier_onnx(
       model: torch.nn.Module,           # PyTorch model
       output_path: Path,                # Where to save ONNX file
       input_shape: tuple[int, int, int], # Model input shape (C, H, W)
       verbose: bool = False             # Print export details
   ) -> None

Export a PyTorch classification model to ONNX format.

**Example:**

.. code-block:: python

   import torchvision

   model = torchvision.models.resnet50(weights='DEFAULT')
   export_classifier_onnx(
       model,
       Path("resnet50.onnx"),
       (3, 224, 224)
   )

.. _common_export_retinanet_onnx:

export_retinanet_onnx()
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def export_retinanet_onnx(
       model: torch.nn.Module,           # PyTorch RetinaNet model
       output_path: Path,                # Output ONNX path
       input_shape: tuple[int, int, int], # Input shape (C, H, W)
       score_threshold: float = 0.5,     # Confidence threshold for detections
       iou_threshold: float = 0.5,       # IoU threshold for NMS
       max_detections: int = 100,        # Maximum boxes to return
       verbose: bool = False             # Print export details
   ) -> None

Export RetinaNet detection model with TensorRT EfficientNMS plugin to ONNX (includes GPU-accelerated NMS).

.. _common_export_segmentation_onnx:

export_segmentation_onnx()
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def export_segmentation_onnx(
       model: torch.nn.Module,           # PyTorch segmentation model
       output_path: Path,                # Output ONNX path
       input_shape: tuple[int, int, int], # Input shape (C, H, W)
       verbose: bool = False             # Print export details
   ) -> None

Export segmentation model (FCN, DeepLab, etc.) to ONNX.

**Example:**

.. code-block:: python

   import torchvision

   fcn = torchvision.models.segmentation.fcn_resnet101(weights='DEFAULT')
   export_segmentation_onnx(
       fcn,
       Path("fcn.onnx"),
       (3, 224, 224)
   )

Dependencies
------------

The common module requires:

* **cvcuda** - CV-CUDA
* **numpy** - Array operations
* **tensorrt** - TensorRT inference
* **torch** - PyTorch for model export
* **nvimgcodec** - Image I/O
* **cuda-python** - CUDA runtime bindings

See Also
--------

* :ref:`Hello World Sample <sample_hello_world>` - Uses image I/O functions
* :ref:`Classification Sample <sample_classification>` - Uses TensorRT utilities
* :ref:`Applications <sample_applications>` - End-to-end pipelines
* :ref:`Operators <sample_operators>` - Individual operators
* :ref:`Python API <python_api>` - Core API reference
