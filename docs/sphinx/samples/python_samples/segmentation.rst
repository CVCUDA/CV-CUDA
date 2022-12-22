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

.. _segmentation:

Semantic Segmentation
====================

In this example, we use CVCUDA to accelerate the pre and post processing pipelines in the deep learning inference use case involving a semantic segmentation model. The deep learning model can utilize either PyTorch or TensorRT as a backend. The pre-processing pipeline converts the input image into the format required by the input layer of the model whereas the post processing pipeline converts the output produced by the model into a visualization-friendly image. We use the DeepLabV3 model (from torchvision) pre-trained on ImageNet and PASCAL VOC 2012 datasets to generate the predictions in the case of PyTorch and use the FCN_ResNet101 model in the case of TensorRT. Both of these models are available as segmentation models in the torchvision package.

**The exact pre-processing operations are:**

Read Image File -> Decode -> Resize -> Convert Datatype(Float) -> Normalize (to 0-1 range, mean and stddev) -> convert to NCHW

**The exact post-processing operations are:**

Normalize the output using softmax -> Create Binary mask -> Upscale the mask -> Blur the input images -> Overlay the masks onto the input image

This sample can work on a single image or a folder full of images. Images have to be in the JPEG format and preferably all having roughly the same dimensions.

Writing the Sample App
----------------------

The first stage in our pipeline is importing necessary python modules. This includes the modules such as torch and torchvision,
torchnvjpeg and the main package of CVCUDA (i.e. nvcv) among others. torchnvjpeg is used to batch decode JPEG images on the GPU.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_python_imports
   :end-before: end_python_imports
   :dedent:

Then we define the main function which helps us parse various configuration parameters used throughout this sample as command line
arguments. This sample allows configuring following parameters. All of them have their default values already set so that one can execute the sample without supplying any.

1. ``-i``, ``--input_path`` : Either a path to a JPEG image or a directory containing JPEG images to use as input.
2. ``-o``, ``--output_dir`` : The directory where the output segmentation overlay should be stored.
3. ``-c``, ``--class_name`` : The segmentation class to visualize the results for.
4. ``-th``, ``--target_img_height`` : The height to which you want to resize the input_image before running inference.
5. ``-tw``, ``--target_img_width`` : The width to which you want to resize the input_image before running inference.
6. ``-b``, ``--batch_size`` : The batch size used during inference. If only one image is used as input, the same input image will be read and used this many times. Useful for performance benchmarking.
7. ``-d``, ``--device_id``  : The GPU device to use for this sample.
8. ``-bk``, ``--backend``  : The inference backend to use. Currently supports pytorch or tensorrt.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: main_func
   :dedent:

Notice that we have created an instance of the ``SemanticSegmentationSample`` class, giving it all these configuration parameters and
then calling the ``run`` method on it. Let's understand what this class is and how its made. To start off, here is how the class is defined with its ``__init__`` method.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: class_def
   :end-before: begin_class_init
   :dedent:


This ``__init__`` method stores all the configuration parameters as class members and then tests to make sure they contain valid values.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_class_init
   :end-before: end_class_init
   :dedent:

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_input_validation
   :end-before: end_input_validation
   :dedent:


Depending on the ``input_path``'s value, we either read one image and create a dummy list with the data from the same image to simulate a batch or read a bunch of images from a directory.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_data_read
   :end-before: end_data_read
   :dedent:

Next, we define the ``setup_model`` method which, depending on the type of backend, performs basic activities regarding setting up the deep learning model.

In case of PyTorch, we use the ``deeplabv3_resnet101`` model, cross check that the class name user supplied is a valid class name in
the context of that model and then set the model on to the GPU in evaluation mode.


.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: setup_model_def
   :end-before: end_setup_pytorch
   :dedent:


In case of TensorRT, we use the ``fcn_resnet101`` model. Since we would like to run the inference using TensorRT, we would need a TensorRT serialized engine for that. One can generate such an engine file by first converting an existing PyTorch model to ONNX and then converting the ONNX to a TensorRT engine. The serialized TensorRT engine is good to work on the specific GPU with the maximum batch size it was given at the creation time. Since ONNX and TensorRT model generation is a time consuming operation, we avoid doing this every-time by first checking if one of those already exists (most likely due to a previous run of this sample.) If so, we simply use those models rather than generating a new one. The final piece to take care of in case of TensorRT is the I/O bindings. We allocate the output Tensors in advance for TensorRT.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_setup_tensorrt
   :end-before: begin_load_tensorrt
   :dedent:

Once a serialized TensorRT engine file written to the disk, we can simply re-load it and continue to setup the I/O bindings.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_load_tensorrt
   :end-before: end_setup_tensorrt
   :dedent:

Methods such as ``convert_onnx_to_tensorrt`` and ``setup_tensort_bindings`` are defined in the helper script file ``trt_utils.py``

Now that we have implemented our model loading logic, we need to write the logic which runs these models with input Tensors and generates the output. Depending on the type of backend, we would handle these operations differently.


In case of PyTorch, we simply execute the model without any gradients computation.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_execute_inference
   :end-before: end_infer_pytorch
   :dedent:

In case of TensorRT, we first start by unpacking three pieces of information from the ``model_info`` object.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_infer_tensorrt
   :end-before: end_tensorrt_unpack
   :dedent:


Next, we need to check if we are dealing with the last batch which may not have a full ``batch_size`` number of samples in it. If so, we would give padded inputs with all zeros so that TensorRT can still run inference as if a full batch was supplied and would simply discard the output results from these padded inputs later.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_check_last_batch
   :end-before: end_check_last_batch
   :dedent:

Next, we prepare the I/O bindings and run the inference.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_tensorrt_run
   :end-before: end_tensorrt_run
   :dedent:

Finally, we would check if we need to discard results corresponding to the padded input samples. We use the ``torch.split`` function here.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_discard_last_batch
   :end-before: end_discard_last_batch
   :dedent:

Once we have the model setup logic and the run inference logic figured out, the only remaining part is to put these things to use in a complete deep learning semantic segmentation pipeline that uses CVCUDA to accelerate the pre and post processing operations. We do so by defining the run method.

Then first couple of things the run method has to do is to actually call the ``setup_model`` method followed by splitting the data into batches with the batch size. Depending on the total items present in the data and the batch size, the last batch may be of size less than the batch size.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_run
   :end-before: end_run_basics
   :dedent:

Then we repeat the pre-processing, inference and post-processing steps for all the batches.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_batch_loop
   :end-before: begin_decode
   :dedent:

We start off the processing by using the torchnvjpeg library to decode the images in a batch into the desired color format and create a tensor list on the device.

Since the steps after this works on torch.Tensor instead of a list of torch.Tensor, we would also convert the list of torch.Tensor
to a higher dimensional torch.Tensor by stacking all the tensors up.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_decode
   :end-before: end_decode
   :dedent:

Once the torch.Tensor is created, we can convert it to a CVCUDA Tensor before starting any CVCUDA based pre-processing.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_torch_to_cvcuda
   :end-before: end_torch_to_cvcuda
   :dedent:

Now we are ready for the pre-processing stages. Here we resize the images, convert to float, normalize them and reformat them in NCHW layout.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_preproc
   :end-before: end_preproc
   :dedent:

The pre-processed tensor is used as an input to the model for inference. We call the ``execute_inference`` method to run the inference.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_run_infer
   :end-before: end_run_infer
   :dedent:

After the inference we deal with all the post-processing operations. We start off by applying softmax to normalize the segmentation probabilities.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_normalize
   :end-before: end_normalize
   :dedent:

Then we filter out the class of our interest by using the argmax function and convert it to a regular uint8 image.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_argmax
   :end-before: end_argmax
   :dedent:

After that we upscale the mask using CVCUDA to the size of the original input images so that we can later overlay the mask onto the image.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_mask_upscale
   :end-before: end_mask_upscale
   :dedent:

Then we generate the blurred version of the input images for later usage in generating the overlays.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_input_blur
   :end-before: end_input_blur
   :dedent:

Now its time to create the overlays. We do this by selectively blurring out pixels in the input image where the class mask prediction was absent (i.e. False)
We already have all the things required for this: The input images, the blurred version of the input images and the upscale version of the mask

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_overlay
   :end-before: end_overlay
   :dedent:

The final stage in the pipeline simply loops over all the images in the batch, generates the overlays, converts it to a PIL image and saves it on the disk.

.. literalinclude:: ../../../../samples/segmentation/python/inference.py
   :language: python
   :linenos:
   :start-after: begin_visualization_loop
   :end-before: end_visualization_loop
   :dedent:


Running the Sample
------------------
This sample can be invoked without any command-line arguments like the following. In that case it will use the default values. It uses the Weimaraner.jpg as the input image, writes the output overlay for the dog class to the /tmp directory with batch size of 1.

.. code-block:: bash

   python3 sample/segmentation/python/inference.py


To run it on a single image with batch size 5 for the background class writing the output to a specific directory:

.. code-block:: bash

   python3 segmentation/python/inference.py -i assets/tabby_tiger_cat.jpg -o /tmp -b 5 -c __background__


To run it on a folder worth of images with batch size 5 for the background class writing the output to a specific directory:

.. code-block:: bash

   python3 segmentation/python/inference.py -i assets/ -o /tmp -b 5 -c __background__


To run on a single image with custom resized input given to the model for the dog class with batch size of 1:

.. code-block:: bash

   python3 segmentation/python/inference.py -i assets/Weimaraner.jpg -o /tmp -b 1 -c dog -th 224 -tw 224


To run on a single image with custom resized input given to the model for the dog class with batch size of 1 using the TensorRT backend instead of PyTorch:

.. code-block:: bash

   python3 segmentation/python/inference.py -i assets/Weimaraner.jpg -o /tmp -b 1 -c dog -th 224 -tw 224 -bk tensorrt



Understanding the Output
------------------------

This sample takes as input the one or more images and generates the semantic segmentation mask overlay on the input image corresponding to a class of your choice and saves it in a directory. Since this sample works on batches, sometimes the batch size and the number of images read may not be a perfect multiple. In such cases, the last batch may have a smaller batch size. If the batch size to anything above 1 for one image input case, the same image is fed in the entire batch and identical image masks are generated and saved for all of them.

.. code-block:: bash

   user@machine:~/cvcuda/samples$ python3 segmentation/python/inference.py -b 5 -c __background__ -o /tmp -i assets/
   Read a total of 2 JPEG images.
   Processing batch 1 of 1
      Saving the overlay result for __background__ class for to: /tmp/out_tabby_tiger_cat.jpg
      Saving the overlay result for __background__ class for to: /tmp/out_Weimaraner.jpg
   user@machine:~/cvcuda/samples$ python3 segmentation/python/inference.py -i assets/Weimaraner.jpg -o /tmp -b 5 -c dog -th 224 -tw 224
   Processing batch 1 of 1
      Saving the overlay result for dog class for to: /tmp/out_Weimaraner.jpg
      Saving the overlay result for dog class for to: /tmp/out_Weimaraner.jpg
      Saving the overlay result for dog class for to: /tmp/out_Weimaraner.jpg
      Saving the overlay result for dog class for to: /tmp/out_Weimaraner.jpg
      Saving the overlay result for dog class for to: /tmp/out_Weimaraner.jpg


.. image:: out_Weimaraner.jpg
   :width: 350
