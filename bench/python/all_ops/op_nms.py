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
import os


class OpNMS(AbstractOpBase):
    def setup(self, input):
        bboxes = torch.load(
            os.path.join(self.assets_dir, "brooklyn_bboxes.pt"),
            map_location="cuda:%d" % self.device_id,
        )
        bboxes = [bboxes[0].clone() for _ in range(input.shape[0])]
        bboxes = torch.stack(bboxes)
        self.bboxes = cvcuda.as_tensor(bboxes)

        scores = torch.load(
            os.path.join(self.assets_dir, "brooklyn_scores.pt"),
            map_location="cuda:%d" % self.device_id,
        )
        scores = [scores[0].clone() for _ in range(input.shape[0])]
        scores = torch.stack(scores)
        self.scores = cvcuda.as_tensor(scores)
        self.confidence_threshold = 0.9
        self.iou_threshold = 0.2

    def run(self, input):
        return cvcuda.nms(
            self.bboxes, self.scores, self.confidence_threshold, self.iou_threshold
        )

    def visualize(self):
        pass
