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

.. _python_classification:

Image Classification
====================

In this example we will cover use of CVCUDA to accelerate the preprocessing pipeline in DL inference usecase.
The preprocessing pipeline converts the input image to the required format for the input layer of the model.
We will use the Resnet50 model pretrained on Imagenet to implement an image classification pipeline.

The preprocesing operations required for Resnet50 include:

Resize -> Convert Datatype(Float) -> Normalize (std deviation/mean) -> Interleaved to planar

Writing the Sample App
----------------------

The first stage in the sample pipeline is importing the necessary python modules and cvcuda module

.. literalinclude:: ../../../../samples/classification/python/inference.py
   :language: python
   :start-after: Import python module
   :end-before: Classification Sample
   :dedent:

We will then read and load the input images

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :start-after: Image Loading
   :end-before: Validate other inputs
   :dedent:

We will use torchnvjpeg which uses NvJpeg library to decode the images into the desired color format and
create a buffer on the device

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :start-after: NvJpeg Decoder
   :end-before: Wrapping into Tensor
   :dedent:

Once the device buffer is created we will wrap the externally allocated buffer in a CVCUDA Tensor
with the NHWC layout

.. literalinclude:: ../../../../samples/classification/python/inference.py
   :language: python
   :start-after:  Wrapping into Tensor
   :end-before:  Preprocess
   :dedent:

The input buffer is now ready for the preprocessing stage

.. literalinclude:: ../../../../samples/classification/python/inference.py
   :language: python
   :start-after:  Preprocess
   :end-before: Inference
   :dedent:

The preprocessed tensor is used as an input to the resnet model for inference. The cvcuda tensor
can be exported to torch using the .cuda() operator. If the device type of the torch tensor and
cvcuda tensor are same there will be no memory copy

.. literalinclude:: ../../../../samples/classification/python/inference.py
   :language: python
   :start-after: Inference
   :end-before: Postprocess
   :dedent:

The final stage in the pipeline is the post processing to apply softmax to normalize the score and sort the scores to get the TopN scores

.. literalinclude:: ../../../../samples/classification/python/inference.py
   :language: python
   :start-after: Postprocess
   :end-before: Display Top N Results
   :dedent:

Running the Sample
------------------

Run classification sample for single image with batch size 1

.. code-block:: bash

   python3 ./classification/python/inference.py -i ./assets/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 1

Run classification sample for single image with batch size 4, This would copy the same image across the batch

.. code-block:: bash

   python3 ./classification/python/inference.py -i ./assets/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 4

Run classification sample for image directory as input with batch size 2

.. code-block:: bash

   python3 ./classification/python/inference.py -i ./assets -l ./models/imagenet-classes.txt -b 2

Sample Output
-------------

The top 5 classification results for the tabby_cat_tiger.jpg image is as follows:

.. code-block:: bash

   Class :  tiger cat  Score :  0.7251133322715759
   Class :  tabby, tabby cat  Score :  0.15487350523471832
   Class :  Egyptian cat  Score :  0.08538217097520828
   Class :  lynx, catamount  Score :  0.020933201536536217
   Class :  leopard, Panthera pardus  Score :  0.002835722640156746
