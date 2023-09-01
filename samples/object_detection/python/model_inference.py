# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Bring the commons folder from the samples directory into our path so that
# we can import modules from it.
import os
import sys
import logging
import urllib.request
import cvcuda
import nvcv
import torch
import tensorrt as trt
import tensorflow as tf
import numpy as np


common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
assets_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
)
sys.path.insert(0, common_dir)

from trt_utils import setup_tensort_bindings  # noqa: E402


# docs_tag: begin_init_objectdetectiontensorflow
class ObjectDetectionTensorflow:
    def __init__(
        self,
        output_dir,
        batch_size,
        image_size,
        device_id,
        cvcuda_perf,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf

        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[self.device_id], True)

        hdf5_model_path = os.path.join(output_dir, "resnet34_peoplenet.hdf5")

        if not os.path.isfile(hdf5_model_path):
            # We need to download the HDF5 model first from NGC.
            model_url = (
                "https://api.ngc.nvidia.com/v2/models/"
                "org/nvidia/team/tao/peoplenet/trainable_unencrypted_v2.6/"
                "files?redirect=true&path=model.hdf5"
            )
            self.logger.info("Downloading the PeopleNet model from NGC: %s" % model_url)
            urllib.request.urlretrieve(model_url, hdf5_model_path)
            self.logger.info("Download complete. Saved to: %s" % hdf5_model_path)

        with tf.device("/GPU:%d" % self.device_id):
            self.model = tf.keras.models.load_model(hdf5_model_path)
            self.logger.info("TensorFlow PeopleNet model is loaded.")

        self.logger.info("Using TensorFlow as the inference engine.")
        # docs_tag: end_init_objectdetectiontensorflow

    # docs_tag: begin_call_objectdetectiontensorflow
    def __call__(self, frame_nchw):
        self.cvcuda_perf.push_range("inference.tensorflow")

        if isinstance(frame_nchw, torch.Tensor):
            # We convert torch.Tensor to tf.Tensor by:
            # torch.Tensor -> Pytorch Flat Tensor -> DlPack -> tf.Tensor -> Un-flatten
            frame_nchw_shape = frame_nchw.shape
            frame_nchw = frame_nchw.flatten()
            frame_nchw_tf = tf.experimental.dlpack.from_dlpack(frame_nchw.__dlpack__())
            frame_nchw_tf = tf.reshape(frame_nchw_tf, frame_nchw_shape)

        elif isinstance(frame_nchw, nvcv.Tensor):
            # We convert nvcv.Tensor to tf.Tensor by:
            # nvcv.Tensor -> PyTorch Tensor -> Pytorch Flat Tensor -> DlPack -> tf.Tensor -> Un-flatten
            frame_nchw_pyt = torch.as_tensor(
                frame_nchw.cuda(), device="cuda:%d" % self.device_id
            )
            frame_nchw_pyt = frame_nchw_pyt.flatten()
            frame_nchw_tf = tf.experimental.dlpack.from_dlpack(
                frame_nchw_pyt.__dlpack__()
            )
            frame_nchw_tf = tf.reshape(frame_nchw_tf, frame_nchw.shape)

        elif isinstance(frame_nchw, np.ndarray):
            frame_nchw_tf = tf.convert_to_tensor(frame_nchw)

        else:
            raise ValueError(
                "Invalid type of input tensor for tensorflow inference: %s"
                % str(type(frame_nchw))
            )

        with tf.device("/GPU:%d" % self.device_id):
            output_tensors = self.model(frame_nchw_tf)  # returns a tuple.

        # Convert the output to PyTorch Tensors
        boxes = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(output_tensors[0]))
        score = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(output_tensors[1]))

        self.cvcuda_perf.pop_range()  # inference.tensorflow
        return boxes, score
        # docs_tag: end_call_objectdetectiontensorflow


# docs_tag: begin_init_objectdetectiontensorrt
class ObjectDetectionTensorRT:
    def __init__(
        self,
        output_dir,
        batch_size,
        image_size,
        device_id,
        cvcuda_perf,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf

        # Download and prepare the models for the first use.
        etlt_model_path = os.path.join(self.output_dir, "resnet34_peoplenet_int8.etlt")
        trt_engine_file_path = os.path.join(
            self.output_dir,
            "resnet34_peoplenet_int8.%d.%d.%d.trtmodel"
            % (
                batch_size,
                image_size[1],
                image_size[0],
            ),
        )

        # Check if we have a previously generated model.
        if not os.path.isfile(trt_engine_file_path):
            if not os.path.isfile(etlt_model_path):
                # We need to download the ETLE model first from NGC.
                model_url = (
                    "https://api.ngc.nvidia.com/v2/models/"
                    "nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/"
                    "files/resnet34_peoplenet_int8.etlt"
                )
                self.logger.info(
                    "Downloading the PeopleNet model from NGC: %s" % model_url
                )
                urllib.request.urlretrieve(model_url, etlt_model_path)
                self.logger.info("Download complete. Saved to: %s" % etlt_model_path)

            # Convert ETLE to TensorRT model using the TAO-Converter.
            self.logger.info("Converting the PeopleNet model to TensorRT...")
            if os.system(
                "tao-converter -e %s -k tlt_encode -d 3,%d,%d -m %d -i nchw %s"
                % (
                    trt_engine_file_path,
                    image_size[1],
                    image_size[0],
                    batch_size,
                    etlt_model_path,
                )
            ):
                raise Exception("Conversion failed.")
            else:
                self.logger.info(
                    "Conversion complete. Saved to: %s" % trt_engine_file_path
                )

        # Once the TensorRT engine generation is all done, we load it.
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(trt_engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            # Keeping this as a class variable because we want to be able to
            # allocate the output tensors either on its first use or when the
            # batch size changes
            self.trt_model = runtime.deserialize_cuda_engine(f.read())

        # Create execution context.
        self.model = self.trt_model.create_execution_context()

        # We will allocate the output tensors and its bindings either when we
        # use it for the first time or when the batch size changes.
        self.output_tensors, self.output_idx = None, None

        self.logger.info("Using TensorRT as the inference engine.")
        # docs_tag: end_init_objectdetectiontensorrt

    # docs_tag: begin_call_objectdetectiontensorrt
    def __call__(self, tensor):
        self.cvcuda_perf.push_range("inference.tensorrt")

        # Grab the data directly from the pre-allocated tensor.
        input_bindings = [tensor.cuda().__cuda_array_interface__["data"][0]]
        output_bindings = []

        actual_batch_size = tensor.shape[0]

        # Need to allocate the output tensors
        if not self.output_tensors or actual_batch_size != self.batch_size:
            self.output_tensors, self.output_idx = setup_tensort_bindings(
                self.trt_model,
                actual_batch_size,
                self.device_id,
                self.logger,
            )

        for t in self.output_tensors:
            output_bindings.append(t.data_ptr())
        io_bindings = input_bindings + output_bindings

        # Call inference for implicit batch
        self.model.execute_async(
            actual_batch_size,
            bindings=io_bindings,
            stream_handle=cvcuda.Stream.current.handle,
        )

        boxes = self.output_tensors[0]
        score = self.output_tensors[1]

        self.cvcuda_perf.pop_range()  # inference.tensorrt
        return boxes, score
        # docs_tag: end_call_objectdetectiontensorrt
