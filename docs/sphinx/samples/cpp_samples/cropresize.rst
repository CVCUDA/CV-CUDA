..
  # SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _cpp_cropresize:

Crop And Resize
===============

In this example we will cover some basic concepts to show how to use the CVCUDA C++ API which includes usage of Tensor
,wrapping externally allocated data in CVCUDA Tensor and using Tensors with operators.

Creating a CMake Project
------------------------

Create the cmake project to build the application as follows. The <samples/common> folder provides utilities common across the C++ samples including IO utilities to read and write images using NvJpeg.

.. literalinclude:: ../../../../samples/cropandresize/CMakeLists.txt
   :language: cpp
   :start-after: Build crop and resize sample
   :end-before: Install binaries
   :dedent:

Writing the Sample App
----------------------

The first stage in the sample pipeline is loading the Input image.
A cuda stream is created to enqueue all the tasks

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Create the cuda stream
   :end-before: Allocate input tensor
   :dedent:

Since we need a contiguous buffer for a batch, we will preallocate the Tensor buffer for the input batch.

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Allocate input tensor
   :end-before: Tensor Requirements
   :dedent:

The Tensor Buffer is then wrapped to create a Tensor Object for which we will calculate the requirements of the buffer such as strides and alignment

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Tensor Requirements
   :end-before: Image Loading
   :dedent:

We will use NvJpeg library to decode the images into the required color format and create a buffer on the device.

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Image Loading
   :end-before: The input buffer is now ready to be used
   :dedent:

The CVCUDA Tensor is now ready to be used by the operators.

We will allocate the Tensors required for Resize and Crop using CVCUDA Allocator.

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Allocate Tensors for Crop and Resize
   :end-before: Initialize operators for Crop and Resize
   :dedent:

Initialize the resize and crop operators

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Initialize operators for Crop and Resize
   :end-before: Executes the CustomCrop operation
   :dedent:

We can now enqueue both the operations in the stream

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Executes the CustomCrop operation
   :end-before: Profile section
   :dedent:

To access the output we will synchronize the stream and copy to the CPU Output buffer
We will use the utility below to sync and write the CPU output buffer into a bitmap file

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :start-after: Copy the buffer to CPU
   :end-before: Clean up
   :language: cpp
   :dedent:

Destroy the cuda stream created

.. literalinclude:: ../../../../samples/cropandresize/Main.cpp
   :language: cpp
   :start-after: Clean up
   :end-before: End of Sample
   :dedent:

Build and Run the Sample
------------------------

The sample can now be compiled using cmake.

.. code-block:: bash

   mkdir build
   cd build
   cmake .. && make

To run the sample

.. code-block:: bash

   ./build/nvcv_samples_cropandresize -i <image path> -b <batch size>

Sample Output
-------------

Input Image of size 700x700

.. image:: ../../../../samples/assets/tabby_tiger_cat.jpg
   :width: 350

Output Image cropped with ROI [150, 50, 400, 300] and resized to 320x240

.. image:: ./tabby_cat_crop.bmp
   :width: 160
