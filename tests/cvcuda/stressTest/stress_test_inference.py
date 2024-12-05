# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import numpy as np
import cvcuda
import torch
import random
import nvcv

import os
import sys
import urllib.request
import time

import tensorrt as trt
import tensorflow as tf

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

time_of_test_in_min = 15
max_batch_size = 10

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def preprocess(input, out_size):
    frame_nhwc = cvcuda.as_tensor(
        torch.as_tensor(input).to(device="cuda:0", non_blocking=True),
        "NHWC",
    )
    resized = cvcuda.resize(
        frame_nhwc,
        (
            frame_nhwc.shape[0],
            out_size[1],
            out_size[0],
            frame_nhwc.shape[3],
        ),
        cvcuda.Interp.LINEAR,
    )
    # Convert to floating point range 0-1.
    normalized = cvcuda.convertto(resized, np.float32, scale=1 / 255)
    # Convert it to NCHW layout and return it.
    normalized = cvcuda.reformat(normalized, "NCHW")
    return normalized


def preprocess_into(input, out_size):
    torch.cuda.synchronize()
    cvcuda_RGBtensor = cvcuda.as_tensor(input.cuda(), "NHWC")
    torch.cuda.synchronize()
    torch_RGBtensor_resized = torch.empty(
        (
            cvcuda_RGBtensor.shape[0],
            out_size[1],
            out_size[0],
            cvcuda_RGBtensor.shape[3],
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )
    cvcuda_RGBtensor_resized = cvcuda.as_tensor(
        torch_RGBtensor_resized.cuda(),
        "NHWC",
    )
    cvcuda.resize_into(
        cvcuda_RGBtensor_resized,
        cvcuda_RGBtensor,
        cvcuda.Interp.LINEAR,
    )

    torch_nchw = torch.empty(
        (input.shape[0], 3, out_size[1], out_size[0]),
        dtype=torch.uint8,
        device="cuda:0",
    )
    cvcuda_nchw = cvcuda.as_tensor(torch_nchw.cuda(0), "NCHW")
    # normalized = cvcuda.convertto(cvcuda_nchw, np.float32, scale=1 / 255)
    cvcuda.reformat_into(cvcuda_nchw, cvcuda_RGBtensor_resized)
    return torch_nchw


def generate_images(N, width=None, height=None, random_size=False):
    if random_size:
        w = random.randint(100, 500)
        h = random.randint(100, 500)
    else:
        w = width
        h = height
    return torch.as_tensor(torch.rand(N, h, w, 3), dtype=torch.uint8)


class ObjectDetectionTensorflow:
    def __init__(
        self,
        output_dir,
        batch_size,
        image_size,
        device_id,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.device_id = device_id

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

    def __call__(self, frame_nchw):

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
        score = torch.from_dlpack(
            tf.experimental.dlpack.to_dlpack(output_tensors[1])
        )  # inference.tensorflow
        return boxes, score


class ObjectDetectionTensorRT:
    def __init__(
        self,
        output_dir,
        batch_size,
        image_size,
        device_id,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.device_id = device_id

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

    def __call__(self, tensor):

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
        score = self.output_tensors[1]  # inference.tensorrt
        return boxes, score


def test_random_image_size():
    target_img_width = 960
    target_img_height = 544
    image_size = (target_img_width, target_img_height)
    batch_size = 1
    device_id = 0
    backend = "tensorflow"
    output_dir = ""
    if backend == "tensorflow":
        inference = ObjectDetectionTensorflow(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )

    elif backend == "tensorrt":
        inference = ObjectDetectionTensorRT(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )
    else:
        raise ValueError("Unknown backend: %s" % backend)

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess(gradient_img_batch, image_size)
        bboxes, probabilities = inference(result)
    print("Random Image Size Test Complete")


def test_increasing_batch_size():
    target_img_width = 960
    target_img_height = 544
    image_size = (target_img_width, target_img_height)
    batch_size = 1
    device_id = 0
    backend = "tensorflow"
    output_dir = ""
    if backend == "tensorflow":
        inference = ObjectDetectionTensorflow(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )

    elif backend == "tensorrt":
        inference = ObjectDetectionTensorRT(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )
    else:
        raise ValueError("Unknown backend: %s" % backend)
    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration and batch_size < max_batch_size:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess(gradient_img_batch, image_size)
        bboxes, probabilities = inference(result)
        batch_size += 1
    print("Random Image Size Test Complete")


def test_random_batch_size():
    target_img_width = 960
    target_img_height = 544
    image_size = (target_img_width, target_img_height)
    batch_size = 1
    device_id = 0
    backend = "tensorflow"
    output_dir = ""
    if backend == "tensorflow":
        inference = ObjectDetectionTensorflow(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )

    elif backend == "tensorrt":
        inference = ObjectDetectionTensorRT(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )
    else:
        raise ValueError("Unknown backend: %s" % backend)

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(width=1080, height=1920, N=batch_size)
        image_size = (target_img_width, target_img_height)
        result = preprocess(gradient_img_batch, image_size)
        bboxes, probabilities = inference(result)
        batch_size = random.randint(1, 80)
    print("Random Batch Size Test Complete")


def test_random_image_size_into():
    target_img_width = 960
    target_img_height = 544
    image_size = (target_img_width, target_img_height)
    batch_size = 1
    device_id = 0
    backend = "tensorflow"
    output_dir = ""
    if backend == "tensorflow":
        inference = ObjectDetectionTensorflow(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )

    elif backend == "tensorrt":
        inference = ObjectDetectionTensorRT(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )
    else:
        raise ValueError("Unknown backend: %s" % backend)
    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        # while True:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        # print(gradient_img_batch.size())
        image_size = (target_img_width, target_img_height)
        result = preprocess_into(gradient_img_batch, image_size)
        bboxes, probabilities = inference(result)
        # print(f"bboxes :{bboxes}")
        # print(f"probabilities :{probabilities}")
    print("Into operator Random Image Size Test Complete")


def test_increasing_batch_size_into():
    target_img_width = 960
    target_img_height = 544
    image_size = (target_img_width, target_img_height)
    batch_size = 1
    device_id = 0
    backend = "tensorflow"
    output_dir = ""
    if backend == "tensorflow":
        inference = ObjectDetectionTensorflow(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )

    elif backend == "tensorrt":
        inference = ObjectDetectionTensorRT(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )
    else:
        raise ValueError("Unknown backend: %s" % backend)
    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration and batch_size < max_batch_size:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess_into(gradient_img_batch, image_size)
        bboxes, probabilities = inference(result)
        batch_size += 1
    print("Into operator Random Image Size Test Complete")


def test_random_batch_size_into():
    target_img_width = 960
    target_img_height = 544
    image_size = (target_img_width, target_img_height)
    batch_size = 1
    device_id = 0
    backend = "tensorflow"
    output_dir = ""
    if backend == "tensorflow":
        inference = ObjectDetectionTensorflow(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )

    elif backend == "tensorrt":
        inference = ObjectDetectionTensorRT(
            output_dir,
            batch_size,
            image_size,
            device_id,
        )
    else:
        raise ValueError("Unknown backend: %s" % backend)
    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess_into(gradient_img_batch, image_size)
        bboxes, probabilities = inference(result)
        batch_size = random.randint(1, 80)
    print("Into Operator Random Batch Size Test Complete")


def main():
    print(torch.cuda.get_device_properties(0))
    test_random_image_size()
    test_random_batch_size()
    test_random_image_size_into()
    test_random_batch_size_into()

    # test_increasing_batch_size()
    # test_increasing_batch_size_into()


if __name__ == "__main__":
    main()
