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
import logging
import torch
import nvtx
from torchvision.models import segmentation as segmentation_models


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
