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

# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401

from bench_utils import AbstractOpBase
import cvcuda
import torch
from torchvision.io import read_image
import os


class OpBoundingBox(AbstractOpBase):
    def setup(self, input):
        self.border_color = (0, 255, 0, 255)
        self.fill_color = (0, 0, 255, 0)
        self.thickness = 5

        data = read_image(os.path.join(self.assets_dir, "brooklyn.jpg"))
        data = data.moveaxis(0, -1).contiguous()  # From CHW to HWC
        data = data.cuda(self.device_id)
        data = [data.clone() for _ in range(input.shape[0])]
        data = torch.stack(data)
        self.input = cvcuda.as_tensor(data, "NHWC")

        bboxes = torch.load(
            os.path.join(self.assets_dir, "brooklyn_bboxes.pt"),
            map_location="cuda:%d" % self.device_id,
        )
        bboxes = [bboxes[0].clone() for _ in range(input.shape[0])]
        self.bboxes_pyt = torch.stack(bboxes)
        bboxes = cvcuda.as_tensor(self.bboxes_pyt)

        scores = torch.load(
            os.path.join(self.assets_dir, "brooklyn_scores.pt"),
            map_location="cuda:%d" % self.device_id,
        )
        scores = [scores[0].clone() for _ in range(input.shape[0])]
        scores = torch.stack(scores)
        scores = cvcuda.as_tensor(scores)

        self.nms_masks_pyt = torch.load(
            os.path.join(self.assets_dir, "brooklyn_nms_masks.pt"),
            map_location="cuda:%d" % self.device_id,
        )

    def run(self, input):
        bounding_boxes = []
        # Create an array of bounding boxes with render settings.
        for current_boxes, current_masks in zip(self.bboxes_pyt, self.nms_masks_pyt):
            filtered_boxes = current_boxes[current_masks]
            BndBoxI_list = []

            for box in filtered_boxes:
                BndBoxI_list.append(
                    cvcuda.BndBoxI(
                        box=tuple(box),
                        thickness=self.thickness,
                        borderColor=self.border_color,
                        fillColor=self.fill_color,
                    )
                )

            bounding_boxes.append(BndBoxI_list)

        batch_bounding_boxes = cvcuda.BndBoxesI(boxes=bounding_boxes)

        cvcuda.bndbox_into(self.input, self.input, batch_bounding_boxes)

        return self.input
