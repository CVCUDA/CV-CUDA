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

.. _detection:

Object Detection
====================

In this example, we use CVCUDA to accelerate the pre processing, post processing and rendering pipelines in the deep learning inference use case involving an object detection model. The deep learning model can utilize either Tensorflow or TensorRT to run the inference. The pre-processing pipeline converts the input into the format required by the input layer of the model whereas the post processing pipeline extracts and filters the bounding boxes and renders them on the frame. We use the `Peoplenet Model <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet>`_ from NVIDIA NGC to detect people, face and bags in the frame. This sample can work on a single image or a folder full of images or on a single video. All images have to be in the JPEG format and with the same dimensions unless run under the batch size of one. Video has to be in mp4 format with a fixed frame rate. We use the torchnvjpeg library to read the images and NVIDIA's Video Processing Framework (VPF) to read/write videos.

**The exact pre-processing operations are:** ::

   Decode Data -> Resize -> Convert Datatype(Float) -> Normalize (to 0-1 range) -> Convert to NCHW

**The exact post-processing operations are:** ::

   Bounding box and score detections from the network -> Interpolate bounding boxes to the image size -> Filter the bounding boxes using NMS -> Render the bounding boxes -> Blur the ROI's


Writing the Sample App
----------------------

The object detection app has been designed to be modular in all aspects. It imports and uses various modules such as data decoders, encoders, pipeline pre and post processors and the model inference. Some of these modules are defined in the same folder as the sample whereas the rest are defined in the common scripts folder for a wider re-use.

1. Modules used by this sample app that are defined in the common folder (i.e. not specific just to this sample) are the ``ImageBatchDecoderPyTorch`` and ``ImageBatchEncoderPyTorch`` for PyTorch based image decoding and encoding and ``VideoBatchDecoderVPF`` and ``VideoBatchEncoderVPF`` for VPF based video decoding and encoding.

2. Modules specific to this sample (i.e. defined in the object_detection sample folder) are ``PreprocessorCvcuda`` and ``PostprocessorCvcuda`` for CVCUDA based pre and post processing pipelines and ``ObjectDetectionTensorRT`` and ``ObjectDetectionTensorflow`` for the model inference.

The first stage in our pipeline is importing all necessary python modules. Apart from the modules described above, this also includes modules such as torch and torchvision, torchnvjpeg, vpf and the main package of CVCUDA (i.e. nvcv) among others. Be sure to import ``pycuda.driver`` before importing any other GPU packages like torch or cvcuda to ensure a proper initialization.

.. literalinclude:: ../../../../samples/object_detection/python/main.py
   :language: python
   :linenos:
   :start-after: begin_python_imports
   :end-before: end_python_imports
   :dedent:

Then we define the main function which helps us parse various configuration parameters used throughout this sample as command line
arguments. This sample allows configuring following parameters. All of them have their default values already set so that one can execute the sample without supplying any. Some of these arguments are shared across many other CVCUDA samples and hence come from the ``perf_utils.py`` class's ``get_default_arg_parser()`` method.

1. ``-i``, ``--input_path`` : Either a path to a JPEG image/MP4 video or a directory containing JPG images to use as input. When pointing to a directory, only JPG images will be read.
2. ``-o``, ``--output_dir`` : The directory where the output object_detection overlay should be stored.
3. ``-th``, ``--target_img_height`` : The height to which you want to resize the input_image before running inference.
4. ``-tw``, ``--target_img_width`` : The width to which you want to resize the input_image before running inference.
5. ``-b``, ``--batch_size`` : The batch size used during inference. If only one image is used as input, the same input image will be read and used this many times. Useful for performance bench-marking.
6. ``-d``, ``--device_id``  : The GPU device to use for this sample.
7. ``-c``, ``--confidence_threshold``  : The confidence threshold for filtering out the detected bounding boxes.
8. ``-iou``, ``--iou_threshold``  : The Intersection over Union threshold for NMS.
9. ``-bk``, ``--backend``  : The inference backend to use. Currently supports Tensorflow or TensorRT.

Once we are done parsing all the command line arguments, we would setup the ``CvCudaPerf`` object for any performance benchmarking needs and simply call the function ``run_sample`` with all those arguments.

.. literalinclude:: ../../../../samples/object_detection/python/main.py
   :language: python
   :linenos:
   :start-after: start_call_run_sample
   :end-before: end_call_run_sample
   :dedent:

The ``run_sample`` function is the primary function that runs this sample. It sets up the requested CUDA device, CUDA context and CUDA stream. CUDA streams help us execute CUDA operations on a non-default stream and enhances the overall performance. Additionally, NVTX markers are used throughout this sample (via ``CvCudaPerf``) to facilitate performance bench-marking using `NVIDIA NSIGHT systems <https://developer.nvidia.com/nsight-systems>`_ and ``benchmark.py``. In order to keep things simple, we are only creating one CUDA stream to run all the stages of this sample. The same stream is available in CVCUDA, PyTorch and TensorRT.

.. literalinclude:: ../../../../samples/object_detection/python/main.py
   :language: python
   :linenos:
   :start-after: begin_setup_sample
   :end-before: end_setup_sample
   :dedent:

Once the streams have been defined and initialized, all the operations in the rest of this sample will be executed inside the stream.

.. literalinclude:: ../../../../samples/object_detection/python/main.py
   :language: python
   :linenos:
   :start-after: begin_setup_gpu
   :end-before: end_setup_gpu
   :dedent:

Next, we instantiate various classes to help us run the sample. These classes are:

1. ``PreprocessorCvcuda`` : A CVCUDA based pre-processing pipeline for object detection.
2. ``ImageBatchDecoderPyTorch`` : A PyTorch based image decoder to read the images.
3. ``ImageBatchEncoderPyTorch`` : A PyTorch based image encoder to write the images.
4. ``VideoBatchDecoderVPF`` : A VPF based video decoder to read the video.
5. ``VideoBatchEncoderVPF`` : A VPF based video encoder to write the video.
6. ``PostProcessorCvcuda`` : A CVCUDA based post-processing pipeline for object detection.
7. ``ObjectDetectionTensorflow`` : A TensorFlow based object detection model to execute inference.
8. ``ObjectDetectionTensorRT`` : A TensorRT based object detection model to execute inference.

These classes are defined in modular fashion and exposes a uniform interface which allows easy plug-and-play in appropriate places. For example, one can use the same API to decode/encode images using PyTorch as that of decode/encode videos using VPF. Similarly, one can invoke the inference in the exact same way with TensorFlow as with TensorRT.

Additionally, the encoder and decoder interfaces also exposes start and join methods, making it easy to upgrade them to a multi-threading environment (if needed.) Such multi-threading capabilities are slated for a future release.

With all of these components initialized, the overall data flow per a data batch looks like the following:

Decode batch -> Preprocess Batch -> Run Inference -> Post Process Batch -> Encode batch

.. literalinclude:: ../../../../samples/object_detection/python/main.py
   :language: python
   :linenos:
   :start-after: begin_pipeline
   :end-before: end_pipeline
   :dedent:


That's it for the object detection sample. To understand more about how each stage in the pipeline works, please explore the following sections:

.. toctree::
    :maxdepth: 1

    PreprocessorCvcuda <object_detection/preprocessor_cvcuda>
    PostprocessorCvcuda <object_detection/postprocessor_cvcuda>
    ImageBatchDecoderPyTorch <commons/imagebatchdecoder_pytorch>
    ImageBatchEncoderPyTorch <commons/imagebatchencoder_pytorch>
    VideoBatchDecoderVPF <commons/videobatchdecoder_vpf>
    VideoBatchEncoderVPF <commons/videobatchencoder_vpf>
    ObjectDetectionTensorFlow <object_detection/objectdetection_tensorflow>
    ObjectDetectionTensorRT <object_detection/objectdetection_tensorrt>


Running the Sample
------------------

The sample can be invoked without any command-line arguments like the following. In that case it will use the default values. It uses peoplenet.jpg as the input image, renders the bounding boxes and writes the image to /tmp directory with batch size of 1. The default confidence threshold is 0.9 and iou threshold is 0.1.

.. code-block:: bash

   python3 object_detection/python/main.py


To run it on a specific image

.. code-block:: bash

   python3 object_detection/python/main.py -i assets/images/tabby_tiger_cat.jpg


To run it a folder worth of images with with batch size 2

.. code-block:: bash

   python3 object_detection/python/main.py -i assets/images -b 2


To run it a folder worth of images with with batch size 2 with the TensorFlow backend

.. code-block:: bash

   python3 object_detection/python/main.py -i assets/images -b 2 -bk tensorflow


To run it on a video with batch size 4

.. code-block:: bash

   python3 object_detection/python/main.py -i assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 -b 4


Understanding the Output
------------------------

This sample takes as input one or more images or one video and generates the object detection boxes with the regions inside the bounding box blurred. The IOU threshold for NMS and confidence threshold for the bounding boxes can be configured as a runtime parameter from the command line. Since this sample works on batches, sometimes the batch size and the number of images read may not be a perfect multiple. In such cases, the last batch may have a smaller batch size.

.. code-block:: bash

        user@machine:~/cvcuda/samples$ python3 object_detection/python/main.py
        [perf_utils:85] 2023-07-27 23:15:34 WARNING perf_utils is used without benchmark.py. Benchmarking mode is turned off.
        [perf_utils:89] 2023-07-27 23:15:34 INFO   Using CV-CUDA version: 0.5.0-beta
        [pipelines:30] 2023-07-27 23:15:36 INFO   Using CVCUDA as preprocessor.
        [torch_utils:77] 2023-07-27 23:15:36 INFO   Using torchnvjpeg as decoder.
        [torch_utils:151] 2023-07-27 23:15:36 INFO   Using PyTorch/PIL as encoder.
        [pipelines:137] 2023-07-27 23:15:36 INFO   Using CVCUDA as post-processor.
        [model_inference:210] 2023-07-27 23:15:37 INFO   Using TensorRT as the inference engine.
        [object_detection:166] 2023-07-27 23:15:37 INFO   Processing batch 0
        [torch_utils:165] 2023-07-27 23:15:37 INFO   Saving the overlay result to: /tmp/out_peoplenet.jpg
        [torch_utils:165] 2023-07-27 23:15:37 INFO   Saving the overlay result to: /tmp/out_peoplenet.jpg
        [torch_utils:165] 2023-07-27 23:15:37 INFO   Saving the overlay result to: /tmp/out_peoplenet.jpg
        [torch_utils:165] 2023-07-27 23:15:37 INFO   Saving the overlay result to: /tmp/out_peoplenet.jpg


Input Image

.. image:: peoplenet.jpg
   :width: 350

Output Image

.. image:: out_peoplenet.jpg
   :width: 350
