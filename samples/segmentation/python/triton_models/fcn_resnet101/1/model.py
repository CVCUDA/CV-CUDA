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

# docs_tag: begin_python_imports
# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda
import json
from types import SimpleNamespace
import torch
import cvcuda
import os
import sys
from pathlib import Path

# Import Triton modules
import triton_python_backend_utils as pb_utils


# Bring module folders from the samples directory into our path so that
# we can import modules from it.
samples_dir = Path(os.path.abspath(__file__)).parents[5]  # samples/
segmentation_dir = Path(os.path.abspath(__file__)).parents[
    3
]  # samples/segmentation/python
sys.path.insert(0, os.path.join(samples_dir, ""))
sys.path.insert(0, os.path.join(segmentation_dir, ""))

from model_inference import SegmentationPyTorch, SegmentationTensorRT  # noqa: E402
from pipelines import PreprocessorCvcuda, PostprocessorCvcuda  # noqa: E402

from common.python.perf_utils import CvCudaPerf  # noqa: E402

# docs_tag: end_python_imports


# Triton Python Model
class TritonPythonModel:
    def initialize(self, args):
        # docs_tag: begin_init_model
        self.model_config = json.loads(args["model_config"])
        params = self.model_config["parameters"]
        self.max_batch_size = self.model_config["max_batch_size"]
        self.device_id = int(params["device_id"]["string_value"])
        self.network_width = int(params["network_width"]["string_value"])
        self.network_height = int(params["network_height"]["string_value"])
        self.visualization_class_name = params["visualization_class_name"][
            "string_value"
        ]
        self.inference_backend = params["inference_backend"]["string_value"]
        cuda_device = cuda.Device(self.device_id)
        self.cuda_ctx = cuda_device.retain_primary_context()
        self.cuda_ctx.push()
        self.cvcuda_stream = cvcuda.Stream()
        self.torch_stream = torch.cuda.ExternalStream(self.cvcuda_stream.handle)

        # Use CvCudaPerf class to record performance of various portions of code
        # It reports the data back to nvtx internally.
        # Since it requires a minimal object with certain properties passed in it
        # we will create it here. SimpleNamespace is used to create an object
        # with arbitrary attributes.
        args = SimpleNamespace()
        args.output_dir = "/tmp"
        args.device_id = self.device_id
        self.cvcuda_perf = CvCudaPerf("segmentation_triton_server", default_args=args)

        if self.inference_backend == "tensorrt":
            self.inference = SegmentationTensorRT(
                output_dir="/tmp",
                seg_class_name=self.visualization_class_name,
                batch_size=self.max_batch_size,
                image_size=(self.network_width, self.network_height),
                device_id=self.device_id,
                cvcuda_perf=self.cvcuda_perf,
            )
        else:
            self.inference = SegmentationPyTorch(
                output_dir="/tmp",
                seg_class_name=self.visualization_class_name,
                batch_size=1,
                image_size=(self.network_width, self.network_height),
                device_id=self.device_id,
                cvcuda_perf=self.cvcuda_perf,
            )

        self.input_tensor_name = "inputrgb"
        self.output_tensor_name = "outputrgb"

        self.preprocess = PreprocessorCvcuda(self.device_id, self.cvcuda_perf)
        self.postprocess = PostprocessorCvcuda(
            "NCHW",
            gpu_output=True,
            device_id=self.device_id,
            cvcuda_perf=self.cvcuda_perf,
        )

        self.logger = pb_utils.Logger

        # docs_tag: end_init_model

    # docs_tag: begin_execute_model
    def execute(self, requests):
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        try:
            with self.cvcuda_stream, torch.cuda.stream(self.torch_stream):
                for idx, request in enumerate(requests):
                    self.cvcuda_perf.push_range("request", batch_idx=idx)

                    self.cvcuda_perf.push_range("preprocess")
                    in_0 = pb_utils.get_input_tensor_by_name(
                        request, self.input_tensor_name
                    )
                    in_0_numpy = in_0.as_numpy()
                    image_tensors = torch.from_numpy(in_0_numpy)

                    orig_tensor, resized_tensor, normalized_tensor = self.preprocess(
                        image_tensors.cuda(),
                        out_size=(self.network_width, self.network_height),
                    )
                    self.cvcuda_perf.pop_range()

                    self.cvcuda_perf.push_range("inference")
                    probabilities = self.inference(normalized_tensor)
                    self.cvcuda_perf.pop_range()

                    self.cvcuda_perf.push_range("postprocess")
                    blurred_frame = self.postprocess(
                        probabilities,
                        orig_tensor,
                        resized_tensor,
                        self.inference.class_index,
                    )
                    # Get Triton output tensor
                    out_tensor_0 = pb_utils.Tensor(
                        "outputrgb", blurred_frame.cpu().numpy()
                    )
                    # Create inference response
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[out_tensor_0]
                    )
                    responses.append(inference_response)
                    self.cvcuda_perf.pop_range()

                    self.cvcuda_perf.pop_range(total_items=in_0_numpy.shape[0])

            # You should return a list of pb_utils.InferenceResponse. Length
            # of this list must match the length of `requests` list.

            return responses
        except Exception as e:
            print(e)
        # docs_tag: end_execute_model

    # docs_tag: begin_finalize_model
    def finalize(self):
        self.cvcuda_perf.finalize()
        self.cuda_ctx.pop()

    # docs_tag: end_finalize_model
