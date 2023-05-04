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
import tensorrt as trt
import nvtx
import cvcuda

common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)

from trt_utils import setup_tensort_bindings  # noqa: E402


# docs_tag: begin_init_objectdetectiontensorrt
class ObjectDetectionTensorRT:
    def __init__(self, engine_file_path, batch_size, device_id):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.batch_size = batch_size

        # Once the TensorRT engine generation is all done, we load it.
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
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
        nvtx.push_range("inference.tensorrt")

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

        nvtx.pop_range()  # inference.tensorrt
        return boxes, score
        # docs_tag: end_call_objectdetectiontensorrt
