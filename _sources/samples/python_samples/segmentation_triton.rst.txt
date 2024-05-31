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

.. _segmentation_triton:

Semantic Segmentation deployed using Triton
===========================================

NVIDIA's Triton Inference Server enables teams to deploy, run, and scale trained AI models from any framework on any GPU- or CPU-based infrastructure.
It offers backend support for most machine learning ML frameworks, as well as custom C++ and python backend
In this tutorial, we will go over an example of taking a CVCUDA accelerated inference workload and deploy it using Triton's Custom Python backend.
Note that for video modality, our Triton implementation supports a non-streamed mode and a streamed processing mode. Non-streamed mode will send decoded/uncompressed frames over the Triton network, where video-to-frame decoding/encoding are processed on the client side. In streamed mode, raw compressed frames are sent over the network and offloads the entire decoding-preprocessing-inference-postprocessing-encoding pipeline to the server side. Performance benchmark indicates the streamed mode has great advantages over the non-streamed for video workload, thus it is highly recommended to turn on with --stream_video (-sv) flag.
Refer to the Segmentation sample documentation to understand the details of the pipeline.

Terminologies
-------------
* Triton Server
Manages and deploys model at scale. Refer the Triton documentation to review all the features Triton has to offer.

* Triton model repository
Triton model represents a inference workload that needs to be deployed. The triton server loads the model repository when started.

* Triton Client
Triton client libraries facilitates communication with Triton using Python or C++ API.
In this example we will demonstrate how to to the Python API to communicate with Triton using GRPC requests.

Tutorial
---------

* Download the Triton server and client dockers. To download the dockers from NGC, the following is required

  * nvidia-docker v2.11.0
  * Working NVIDIA NGC account (visit https://ngc.nvidia.com/setup to get started using NGC) and follow through the NGC documentation here https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#ngc-image-prerequisites
  * docker CLI logged into nvcr.io (NGC's docker registry) to be able to pull docker images.

  .. code-block:: bash

     docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
     docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk

  where xx.yy refers to the Triton version

* Create the model repository

  Triton loads the model repository using the following command:

  .. code-block:: bash

     tritonserver --model-repository=<model-repository-path>

  The model repository paths needs to conform to a layout specified below:

  <model-repository-path>/
      <model-name>/
        <version>/
          <model-definition-file>
        config.pbtxt

  For the segmentation sample, we will create a model.py which creates a TritonPythonModel
  that runs the preprocess, inference and post process workloads.
  We will copy the necessary files and modules from the segmentation sample for preprocess,
  inference and postprocess stages and create the following folder structure:

  triton_models/
     fcn_resnet101/
        1/
           model.py
        config.pbtxt

  Each model in the model repository must include a model configuration that provides the required
  and optional information about the model. Typically, this configuration is provided in a config.pbtxt


  The segmentation config is shown below for non-streamed mode

  .. literalinclude:: ../../../../samples/segmentation/python/triton_models/fcn_resnet101/config.pbtxt
     :language: cpp

  And the following for streamed mode

  .. literalinclude:: ../../../../samples/segmentation/python/triton_models/fcn_resnet101_streaming/config.pbtxt
     :language: cpp

* Triton client (non-streamed mode)

  Triton receives as input the frames (in batches) and returns the segmentation output
  These are represented as input and output layers of the Triton model.
  Additional parameters for initialization of the model can be specified as well.

  We will use the Triton Python API using GRPC protocol to communicate with triton.
  Below is an example on how to create a python Triton client for the segmentation sample

  * Create the Triton GRPC client and set the input and output layer names of the model

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_setup_triton_client
     :end-before: end_setup_triton_client
     :dedent:

  * The client takes as input a set of images or video and decodes the input image or video into a batched tensor.
    We will first initialize the data loader based on the data modality

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_init_dataloader
     :end-before: end_init_dataloader
     :dedent:

  * We are now finished with the initialization steps and will iterate over the video or images and run the pipeline.
    The decoder will return a batch of frames

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_data_decode
     :end-before: end_data_decode
     :dedent:

  * Create a Triton Inference Request by setting the layer name, data, data shape and data type of the input.
    We will also create an InferResponse to receive the output from the server

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_create_triton_input
     :end-before: end_create_triton_input
     :dedent:

  * Create an Async Infer Request to the server

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_async_infer
     :end-before: end_async_infer
     :dedent:

  * Wait for the response from the server. Verify no exception is returned from the server.
    Parse the output data from the InferResponse returned by the server

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_sync_output
     :end-before: end_sync_output
     :dedent:

  * Encode the output based on the data modality

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_encode_output
     :end-before: end_encode_output
     :dedent:

* Triton client (streamed mode)

  For streamed mode of video modality, the workflow is further simplified as the GPU
  workloads are all offloaded to the server side.
  Triton receives raw video packets with metadata instead decompressed frame data,
  and sends output frames as compressed data as well.

  * Demux and stream the input data to server for decoding

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_streamed_decode
     :end-before: end_streamed_decode
     :dedent:

  * Asynchronously receive output data from server

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_async_receive
     :end-before: end_async_receive
     :dedent:

  * Stream output data from server for muxing

  .. literalinclude:: ../../../../samples/segmentation/python/triton_client.py
     :language: python
     :linenos:
     :start-after: begin_streamed_encode
     :end-before: end_streamed_encode
     :dedent:

Running the Sample
------------------

Follow the instructions in the README.md for the setup and instructions to run the sample

.. literalinclude:: ../../../../samples/segmentation/python/README.md
   :language: cpp
