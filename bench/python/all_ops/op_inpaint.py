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

# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401

from bench_utils import AbstractOpBase
import cvcuda
import torch
from torchvision.io import read_image
import os


class OpInpaint(AbstractOpBase):
    def setup(self, input):
        data = read_image(os.path.join(self.assets_dir, "brooklyn.jpg"))
        mask = read_image(os.path.join(self.assets_dir, "countour_lines.jpg"))
        # Binarize the mask
        mask[mask <= 50] = 0
        mask[mask > 50] = 255

        # Add scratch marks on the top of the input data and convert it to tensor
        mask3 = mask.repeat(3, 1, 1)
        data[mask3 > 0] = mask3[mask3 > 0]
        data = data.moveaxis(0, -1).contiguous()  # From CHW to HWC
        data = [data.clone() for _ in range(input.shape[0])]
        data = torch.stack(data)
        data = data.cuda(self.device_id)
        self.data = cvcuda.as_tensor(data, "NHWC")

        mask = torch.unsqueeze(mask[0], -1)  # 3 channel chw to 1 channel hwc mask
        mask = [mask.clone() for _ in range(input.shape[0])]
        mask = torch.stack(mask)
        mask = mask.cuda(self.device_id)
        self.masks = cvcuda.as_tensor(mask, "NHWC")
        self.inpaint_radius = 3

    def run(self, input):
        return cvcuda.inpaint(
            self.data,
            self.masks,
            self.inpaint_radius,
        )
