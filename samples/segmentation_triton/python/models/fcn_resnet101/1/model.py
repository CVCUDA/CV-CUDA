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
import torch
import cvcuda
from model_inference import SegmentationPyTorch
from pipelines import PreprocessorCvcuda, PostprocessorCvcuda

# Import Triton modules
import triton_python_backend_utils as pb_utils

# docs_tag: end_python_imports


# Triton Python Model
class TritonPythonModel:
    def initialize(self, args):
        # docs_tag: begin_init_model
        self.model_config = model_config = json.loads(args["model_config"])
        params = model_config["parameters"]
        self.device_id = int(params["device_id"]["string_value"])
        self.network_width = int(params["network_width"]["string_value"])
        self.network_height = int(params["network_height"]["string_value"])
        self.visualization_class_name = params["visualization_class_name"][
            "string_value"
        ]
        cuda_device = cuda.Device(self.device_id)
        self.cuda_ctx = cuda_device.retain_primary_context()
        self.cuda_ctx.push()
        self.cvcuda_stream = cvcuda.Stream()
        self.torch_stream = torch.cuda.ExternalStream(self.cvcuda_stream.handle)

        self.inference = SegmentationPyTorch(
            output_dir="/tmp",
            seg_class_name=self.visualization_class_name,
            batch_size=1,
            image_size=(self.network_width, self.network_height),
            device_id=self.device_id,
        )

        self.input_tensor_name = "inputrgb"
        self.output_tensor_name = "outputrgb"

        self.preprocess = PreprocessorCvcuda(self.device_id)
        self.postprocess = PostprocessorCvcuda(
            "NCHW",
            gpu_output=True,
            device_id=self.device_id,
        )
        # docs_tag: end_init_model

    # docs_tag: begin_execute_model
    def execute(self, requests):
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        try:
            with self.cvcuda_stream, torch.cuda.stream(self.torch_stream):
                for request in requests:
                    in_0 = pb_utils.get_input_tensor_by_name(
                        request, self.input_tensor_name
                    )
                    in_0_numpy = in_0.as_numpy()
                    image_tensors = torch.from_numpy(in_0_numpy)

                    orig_tensor, resized_tensor, normalized_tensor = self.preprocess(
                        image_tensors.cuda(),
                        out_size=(self.network_width, self.network_height),
                    )

                    probabilities = self.inference(normalized_tensor)

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

            # You should return a list of pb_utils.InferenceResponse. Length
            # of this list must match the length of `requests` list.

            return responses
        except Exception as e:
            print(e)
        # docs_tag: end_execute_model

    # docs_tag: begin_finalize_model
    def finalize(self):
        self.cuda_ctx.pop()

    # docs_tag: end_finalize_model
