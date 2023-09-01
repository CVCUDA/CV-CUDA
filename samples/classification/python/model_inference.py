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
import cvcuda
import torch
from torchvision import models as torchvision_models
import tensorrt as trt

common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)

from trt_utils import convert_onnx_to_tensorrt, setup_tensort_bindings  # noqa: E402

# docs_tag: begin_init_classificationpytorch
class ClassificationPyTorch:  # noqa: E302
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
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf
        # The underlying PyTorch model that we use for inference is the ResNet50 model
        # from torchvision.
        torch_model = torchvision_models.resnet50
        weights = torchvision_models.ResNet50_Weights.DEFAULT
        self.labels = weights.meta["categories"]
        # Save the list of labels so that the C++ sample can read it.
        with open(os.path.join(output_dir, "labels.txt"), "w") as f:
            for line in self.labels:
                f.write("%s\n" % line)

        # Inference uses PyTorch to run a classification model on the pre-processed
        # input and outputs the classification scores.
        class Resnet50_Softmax(torch.nn.Module):
            def __init__(self, resnet50):
                super(Resnet50_Softmax, self).__init__()
                self.resnet50 = resnet50

            def forward(self, x):
                infer_output = self.resnet50(x)
                return torch.nn.functional.softmax(infer_output, dim=1)

        resnet_base = torch_model(weights=weights)
        resnet_base.eval()
        self.model = Resnet50_Softmax(resnet_base).cuda(self.device_id)
        self.model.eval()

        self.logger.info("Using PyTorch as the inference engine.")
        # docs_tag: end_init_classificationpytorch

    # docs_tag: begin_call_classificationpytorch
    def __call__(self, tensor):
        self.cvcuda_perf.push_range("inference.torch")

        with torch.no_grad():

            if isinstance(tensor, torch.Tensor):
                if not tensor.is_cuda:
                    tensor = tensor.to("cuda:%d" % self.device_id)
            else:
                # Convert CVCUDA tensor to Torch tensor.
                tensor = torch.as_tensor(
                    tensor.cuda(), device="cuda:%d" % self.device_id
                )

            classification_scores = self.model(tensor)

        self.cvcuda_perf.pop_range()
        return classification_scores

    # docs_tag: end_call_classificationpytorch


# docs_tag: begin_init_classificationtensorrt
class ClassificationTensorRT:
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
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf
        # For TensorRT, the process is the following:
        # We check if there already exists a TensorRT engine generated
        # previously. If not, we check if there exists an ONNX model generated
        # previously. If not, we will generate both of the one by one
        # and then use those.
        # The underlying PyTorch model that we use in case of TensorRT
        # inference is the ResNet50 model from torchvision. It is only used during
        # the conversion process and not during the inference.
        onnx_file_path = os.path.join(
            self.output_dir,
            "model.%d.%d.%d.onnx"
            % (
                batch_size,
                image_size[1],
                image_size[0],
            ),
        )
        trt_engine_file_path = os.path.join(
            self.output_dir,
            "model.%d.%d.%d.trtmodel"
            % (
                batch_size,
                image_size[1],
                image_size[0],
            ),
        )

        with torch.cuda.stream(torch.cuda.ExternalStream(cvcuda.Stream.current.handle)):

            torch_model = torchvision_models.resnet50
            weights = torchvision_models.ResNet50_Weights.DEFAULT
            self.labels = weights.meta["categories"]
            # Save the list of labels so that the C++ sample can read it.
            with open(os.path.join(output_dir, "labels.txt"), "w") as f:
                for line in self.labels:
                    f.write("%s\n" % line)

            # Check if we have a previously generated model.
            if not os.path.isfile(trt_engine_file_path):
                if not os.path.isfile(onnx_file_path):
                    # First we use PyTorch to create a classification model.
                    with torch.no_grad():

                        class Resnet50_Softmax(torch.nn.Module):
                            def __init__(self, resnet50):
                                super(Resnet50_Softmax, self).__init__()
                                self.resnet50 = resnet50

                            def forward(self, x):
                                infer_output = self.resnet50(x)
                                return torch.nn.functional.softmax(infer_output, dim=1)

                        resnet_base = torch_model(weights=weights)
                        resnet_base.eval()
                        pyt_model = Resnet50_Softmax(resnet_base)
                        pyt_model.cuda(self.device_id)
                        pyt_model.eval()

                        # Allocate a dummy input to help generate an ONNX model.
                        dummy_x_in = torch.randn(
                            batch_size,
                            3,
                            image_size[1],
                            image_size[0],
                            requires_grad=False,
                        ).cuda(self.device_id)

                        # Generate an ONNX model using the PyTorch's onnx export.
                        torch.onnx.export(
                            pyt_model,
                            args=dummy_x_in,
                            f=onnx_file_path,
                            export_params=True,
                            opset_version=15,
                            do_constant_folding=True,
                            input_names=["input"],
                            output_names=["output"],
                            dynamic_axes={
                                "input": {0: "batch_size"},
                                "output": {0: "batch_size"},
                            },
                        )

                        # Remove the tensors and model after this.
                        del pyt_model
                        del dummy_x_in
                        torch.cuda.empty_cache()

                # Now that we have an ONNX model, we will continue generating a
                # serialized TensorRT engine from it.
                convert_onnx_to_tensorrt(
                    onnx_file_path,
                    trt_engine_file_path,
                    max_batch_size=batch_size,
                    max_workspace_size=1,
                )

            # Once the TensorRT engine generation is all done, we load it.
            trt_logger = trt.Logger(trt.Logger.ERROR)
            with open(trt_engine_file_path, "rb") as f, trt.Runtime(
                trt_logger
            ) as runtime:
                trt_model = runtime.deserialize_cuda_engine(f.read())

            # Create execution context.
            self.model = trt_model.create_execution_context()

            # Allocate the output bindings.
            self.output_tensors, self.output_idx = setup_tensort_bindings(
                trt_model,
                batch_size,
                self.device_id,
                self.logger,
            )

            self.logger.info("Using TensorRT as the inference engine.")
        # docs_tag: end_init_classificationtensorrt

    # docs_tag: begin_call_classificationtensorrt
    def __call__(self, tensor):
        self.cvcuda_perf.push_range("inference.tensorrt")

        # Grab the data directly from the pre-allocated tensor.
        input_bindings = [tensor.cuda().__cuda_array_interface__["data"][0]]
        output_bindings = []
        for t in self.output_tensors:
            output_bindings.append(t.data_ptr())
        io_bindings = input_bindings + output_bindings

        # Must call this before inference
        binding_i = self.model.engine.get_binding_index("input")
        assert self.model.set_binding_shape(binding_i, tensor.shape)

        self.model.execute_async_v2(
            bindings=io_bindings, stream_handle=cvcuda.Stream.current.handle
        )

        # Since this model produces only 1 output, we can grab it now.
        classification_scores = self.output_tensors[0]

        self.cvcuda_perf.pop_range()
        return classification_scores

    # docs_tag: end_call_classificationtensorrt
