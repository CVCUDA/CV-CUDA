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


class OpJointBilateral(AbstractOpBase):
    def setup(self, input):
        self.diameter = 5
        self.sigma_color = 50
        self.sigma_space = 1

        data = read_image(os.path.join(self.assets_dir, "brooklyn.jpg"))
        data = data.moveaxis(0, -1).contiguous()  # From CHW to HWC
        data = [data.clone() for _ in range(input.shape[0])]
        data = torch.stack(data)
        data = data.cuda(self.device_id)
        data = cvcuda.as_tensor(data, "NHWC")
        self.grayscale_input = cvcuda.cvtcolor(data, cvcuda.ColorConversion.RGB2GRAY)

        mask = read_image(os.path.join(self.assets_dir, "brooklyn_mask.jpg"))
        mask = mask.moveaxis(0, -1).contiguous()  # From CHW to HWC
        mask = [mask.clone() for _ in range(input.shape[0])]
        mask = torch.stack(mask)
        mask = mask.cuda(self.device_id)
        self.class_masks = cvcuda.as_tensor(mask, "NHWC")

    def run(self, input):
        return cvcuda.joint_bilateral_filter(
            self.class_masks,
            self.grayscale_input,
            diameter=self.diameter,
            sigma_color=self.sigma_color,
            sigma_space=self.sigma_space,
        )
