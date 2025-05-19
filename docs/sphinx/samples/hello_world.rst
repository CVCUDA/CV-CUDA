..
  # SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _hello_world:

Hello World Tutorial
===================

This tutorial will guide you through creating a simple CV-CUDA application that performs basic image processing operations. This is a Python script that will demonstrate the following:

1. Load a batch of images into CV-CUDA
2. Resize the images
3. Apply a Gaussian blur
4. Save the results
5. Visualize the results

Prerequisites
------------

- NVIDIA GPU with compute capabilities 5.2 or newer.
- Ubuntu 20.04, 22.04 or 24.04
- CUDA 12 runtime with compatible NVIDIA driver.
- Python 3.10
- Python packages from ``samples/hello_world/python/requirements.txt``

To run this tutorial, install the required Python packages, preferrably in a virtual environment. This tutorial was writen for Python 3.10, but newer versions of Python 3 may work.

Install the required pip packages listed in the file ``samples/hello_world/python/requirements.txt`` :

.. code-block:: bash

  pip3 install -r requirements.txt

Writing the Hello World App
--------------------------

Find the complete source code for the tutorial in file ``samples/hello_world/python/hello_world.py``

1. First, let's import the necessary modules:

  .. literalinclude:: ../../../samples/hello_world/python/hello_world.py
     :language: python
     :linenos:
     :start-after: begin_python_imports
     :end-before: end_python_imports
     :dedent:

  Module ``argparse`` is used to implement the command line argument parsing:

  Module ``cvcuda`` imports CV-CUDA API.

  Module ``cupy`` is used to get access to numpy interfaces with support for CUDA backend.

  Module ``nvimagecodec`` is used to load (decode) images from files and decode (store) images to files.

  Module ``pyplot`` is used to display images.

2. The ``main()`` function contains the logic for the app.

  We start by loading all the images from files and stack them as a batch into a CV-CUDA tensor.

  .. literalinclude:: ../../../samples/hello_world/python/hello_world.py
     :language: python
     :linenos:
     :start-after: begin_load_image
     :end-before: end_load_image
     :dedent:

  Here, we use ``nvimgcodec.Decoder.decode()`` to decode an image loaded from a file specified in the list of input images into *RGB uint8 HWC* format, loading it into the default CUDA device.

  We convert the loaded ``nvimgcodec.Image`` into a ``cvcuda.Tensor``, bringing the data into CV-CUDA.

  For each input image, we stack it into a batch in a ``cvcuda.Tensor``, converting it from *HWC* to *NHWC* where ``N`` is the batch size. In this tutorial, we require the images to be all of the same size (width and height) to fit them into a single ``cvcuda.Tensor`` with the *NHWC* layout.

  Note that we can perform the CV-CUDA operations directly on each ``cvcuda.Tensor`` as we obtain it without having to batch them. Batching here is used to illustrate how to operate more efficiently on batches of images.

3. Next we perform the image processing.

  .. literalinclude:: ../../../samples/hello_world/python/hello_world.py
     :language: python
     :linenos:
     :start-after: begin_process_image
     :end-before: end_process_image
     :dedent:

  Once the data is in a ``cvcuda.Tensor``, we perform a resize to ``224 x 224``, followed by a Gaussian blur with a ``3 x 3`` kernel and a sigma of ``1``.

4. Then, we retrieve the results from CV-CUDA and store them to the specified output files.

  .. literalinclude:: ../../../samples/hello_world/python/hello_world.py
     :language: python
     :linenos:
     :start-after: begin_store_image
     :end-before: end_store_image
     :dedent:

  We start by wrapping the ``cvcuda.Tensor`` into a ``cupy.array``. The ``cvcuda.Tensor`` object is opaque for performance purposes. This step grants us the flexibility to access the data contained in each resulting image to store it.

  Then, we save the images to the specified files using the ``nvimgcodec.Encoder.write()`` method.

5. Finally, once we have the resulting images wrapped in a ``cupy.array``, we can use ``pyplot`` to display them. We display the first image in the batch as an example.

  .. literalinclude:: ../../../samples/hello_world/python/hello_world.py
     :language: python
     :linenos:
     :start-after: begin_display_image
     :end-before: end_display_image
     :dedent:

Running the Sample
-----------------

To run the hello world example, make sure the prerequisites are satisfied.

.. code-block:: bash

   python3 hello_world.py -i /path/to/image1.jpg /path/to/image2.jpg -o output1.jpg output2.jpg

This will:

1. Load your input images.
2. Apply image processing (resize and Gaussian blur).
3. Save results to output files (existing files will be overwriten).
4. Display the result of the first image.

Command Line Interface
---------------------

- ``--inputs``, ``-i`` is used to input a list of image files to load into the app. These must all be of the same size (width and height). Only images in these formats are supported: jpeg, jpeg2000, tiff, bmp, png, pnm, or webp.
- ``--outputs``, ``-o`` is used to specify the name of the files where the resulting images will be stored. The number of output files must be the same as the number of input files.

Next Steps
----------

Now that you've completed the hello world tutorial, you can:

1. Try modifying the size values or the Gaussian parameters.
2. Add more image processing operations.
3. Explore other CV-CUDA operators.
4. Check out the more advanced samples in the :ref:`samples` section.
