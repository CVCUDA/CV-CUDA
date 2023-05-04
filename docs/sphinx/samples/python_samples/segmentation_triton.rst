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

1. Download the Triton server and client dockers. To download the dockers from NGC, the following is required

a. nvidia-docker v2.11.0
b. Working NVIDIA NGC account (visit https://ngc.nvidia.com/setup to get started using NGC) and follow through the NGC documentation here https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#ngc-image-prerequisites
c. docker CLI logged into nvcr.io (NGC's docker registry) to be able to pull docker images.

.. code-block:: bash

   docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
   docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk

where xx.yy refers to the Triton version

2. Create the model repository

Triton loads the model repository using the following command:

.. code-block:: bash

   tritonserver --model-repository=<model-repository-path>

The model repository paths needs to conform to a layout specified below:

<model-repository-path>/
    <model-name>/
      <config.pbtxt>
      <version>/
        <model-definition-file>


For the segmentation sample, we will create a model.py which creates a TritonPythonModel
that runs the preprocess, inference and post process workloads.
We will copy the necessary files and modules from the segmentation sample for preprocess,
inference and postprocess stages and create the following folder structure:

models/
   fcn_resnet101/
      1/
         model.py
         model_inference.py
         pipelines.py
      config.pbtxt

Each model in the model repository must include a model configuration that provides the required
and optional information about the model. Typically, this configuration is provided in a config.pbtxt

The segmentation config is shown below

.. literalinclude:: ../../../../samples/segmentation_triton/python/models/fcn_resnet101/config.pbtxt
   :language: cpp

Triton receives as input the frames (in batches) and returns the segmentation output
These are represented as input and output layers of the Triton model.
Additional parameters for initialization of the model can be specified as well

3. Triton client
We will use the Triton Python API using GRPC protocol to communicate with triton.
Below is an example on how to create a python Triton client for the segmentation sample

a. Create the Triton GRPC client and set the input and output layer names of the model

.. literalinclude:: ../../../../samples/segmentation_triton/python/triton_client.py
   :language: python
   :linenos:
   :start-after: begin_setup_triton_client
   :end-before: end_setup_triton_client
   :dedent:

b. The client takes as input a set of images or video and decodes the input image or video into a batched tensor.
We will first initialize the data loader based on the data modality

.. literalinclude:: ../../../../samples/segmentation_triton/python/triton_client.py
   :language: python
   :linenos:
   :start-after: begin_init_dataloader
   :end-before: end_init_dataloader
   :dedent:

c. We are now finished with the initialization steps and will iterate over the video or images and run the pipeline.
The decoder will return a batch of frames

.. literalinclude:: ../../../../samples/segmentation_triton/python/triton_client.py
   :language: python
   :linenos:
   :start-after: begin_data_decode
   :end-before: end_data_decode
   :dedent:

d. Create a Triton Inference Request by setting the layer name, data, data shape and data type of the input.
We will also create an InferResponse to receive the output from the server

.. literalinclude:: ../../../../samples/segmentation_triton/python/triton_client.py
   :language: python
   :linenos:
   :start-after: begin_create_triton_input
   :end-before: end_create_triton_input
   :dedent:

e. Create an Async Infer Request to the server

.. literalinclude:: ../../../../samples/segmentation_triton/python/triton_client.py
   :language: python
   :linenos:
   :start-after: begin_async_infer
   :end-before: end_async_infer
   :dedent:

f. Wait for the response from the server. Verify no exception is returned from the server.
Parse the output data from the InferResponse returned by the server

.. literalinclude:: ../../../../samples/segmentation_triton/python/triton_client.py
   :language: python
   :linenos:
   :start-after: begin_sync_output
   :end-before: end_sync_output
   :dedent:

g. Encode the output based on the data modality

.. literalinclude:: ../../../../samples/segmentation_triton/python/triton_client.py
   :language: python
   :linenos:
   :start-after: begin_encode_output
   :end-before: end_encode_output
   :dedent:

Running the Sample
------------------

Follow the instructions in te README.md for the setup and instructions to run the sample

.. literalinclude:: ../../../../samples/segmentation_triton/README.md
   :language: cpp
