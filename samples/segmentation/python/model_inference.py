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
import nvtx
from torchvision.models import segmentation as segmentation_models
import tensorrt as trt

common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)

from trt_utils import convert_onnx_to_tensorrt, setup_tensort_bindings  # noqa: E402

# docs_tag: begin_init_segmentationpytorch
class SegmentationPyTorch:  # noqa: E302
    def __init__(
        self,
        output_dir,
        seg_class_name,
        batch_size,
        image_size,
        device_id,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.device_id = device_id
        # Fetch the segmentation index to class name information from the weights
        # meta properties.
        # The underlying pytorch model that we use for inference is the FCN model
        # from torchvision.
        torch_model = segmentation_models.fcn_resnet101
        weights = segmentation_models.FCN_ResNet101_Weights.DEFAULT

        try:
            self.class_index = weights.meta["categories"].index(seg_class_name)
        except ValueError:
            raise ValueError(
                "Requested segmentation class '%s' is not supported by the "
                "fcn_resnet101 model. All supported class names are: %s"
                % (seg_class_name, ", ".join(weights.meta["categories"]))
            )

        # Inference uses PyTorch to run a segmentation model on the pre-processed
        # input and outputs the segmentation masks.
        class FCN_Softmax(torch.nn.Module):
            def __init__(self, fcn):
                super(FCN_Softmax, self).__init__()
                self.fcn = fcn

            def forward(self, x):
                infer_output = self.fcn(x)["out"]
                return torch.nn.functional.softmax(infer_output, dim=1)

        fcn_base = torch_model(weights=weights)
        fcn_base.eval()
        self.model = FCN_Softmax(fcn_base).cuda(self.device_id)
        self.model.eval()

        self.logger.info("Using PyTorch as the inference engine.")
        # docs_tag: end_init_segmentationpytorch

    # docs_tag: begin_call_segmentationpytorch
    def __call__(self, tensor):
        nvtx.push_range("inference.torch")

        with torch.no_grad():

            if isinstance(tensor, torch.Tensor):
                # We are all good here. Nothing needs to be done.
                pass
            else:
                # Convert CVCUDA tensor to Torch tensor.
                tensor = torch.as_tensor(
                    tensor.cuda(), device="cuda:%d" % self.device_id
                )

            segmented = self.model(tensor)

        nvtx.pop_range()
        return segmented

    # docs_tag: end_call_segmentationpytorch


# docs_tag: begin_init_segmentationtensorrt
class SegmentationTensorRT:
    def __init__(
        self,
        output_dir,
        seg_class_name,
        batch_size,
        image_size,
        device_id,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.device_id = device_id
        # For TensorRT, the process is the following:
        # We check if there already exists a TensorRT engine generated
        # previously. If not, we check if there exists an ONNX model generated
        # previously. If not, we will generate both of the one by one
        # and then use those.
        # The underlying pytorch model that we use in case of TensorRT
        # inference is the FCN model from torchvision. It is only used during
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

            torch_model = segmentation_models.fcn_resnet101
            weights = segmentation_models.FCN_ResNet101_Weights.DEFAULT

            try:
                self.class_index = weights.meta["categories"].index(seg_class_name)
            except ValueError:
                raise ValueError(
                    "Requested segmentation class '%s' is not supported by the "
                    "fcn_resnet101 model. All supported class names are: %s"
                    % (seg_class_name, ", ".join(weights.meta["categories"]))
                )

            # Check if we have a previously generated model.
            if not os.path.isfile(trt_engine_file_path):
                if not os.path.isfile(onnx_file_path):
                    # First we use PyTorch to create a segmentation model.
                    with torch.no_grad():
                        fcn_base = torch_model(weights=weights)

                        class FCN_Softmax(torch.nn.Module):
                            def __init__(self, fcn):
                                super(FCN_Softmax, self).__init__()
                                self.fcn = fcn

                            def forward(self, x):
                                infer_output = self.fcn(x)["out"]
                                return torch.nn.functional.softmax(infer_output, dim=1)

                        fcn_base.eval()
                        pyt_model = FCN_Softmax(fcn_base)
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
        # docs_tag: end_init_segmentationtensorrt

    # docs_tag: begin_call_segmentationtensorrt
    def __call__(self, tensor):
        nvtx.push_range("inference.tensorrt")

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

        segmented = self.output_tensors[self.output_idx]

        nvtx.pop_range()
        return segmented

    # docs_tag: end_call_segmentationtensorrt
