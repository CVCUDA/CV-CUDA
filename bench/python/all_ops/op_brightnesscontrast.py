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


class OpBrightnessContrast(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        brightness = torch.tensor([1.2]).cuda(self.device_id)
        self.brightness = cvcuda.as_tensor(brightness, "N")

        contrast = torch.tensor([0.7]).cuda(self.device_id)
        self.contrast = cvcuda.as_tensor(contrast, "N")

        brightness_shift = torch.tensor([130.0]).cuda(self.device_id)
        self.brightness_shift = cvcuda.as_tensor(brightness_shift, "N")

        contrast_center = torch.tensor([0.5]).cuda(self.device_id)
        self.contrast_center = cvcuda.as_tensor(contrast_center, "N")

    def run(self, input):
        return cvcuda.brightness_contrast(
            input,
            brightness=self.brightness,
            contrast=self.contrast,
            brightness_shift=self.brightness_shift,
            contrast_center=self.contrast_center,
        )
