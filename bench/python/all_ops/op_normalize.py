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


class OpNormalize(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        mean_tensor = (
            torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3).cuda(self.device_id)
        )
        self.mean_tensor = cvcuda.as_tensor(mean_tensor, "NHWC")
        stddev_tensor = (
            torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3).cuda(self.device_id)
        )
        self.stddev_tensor = cvcuda.as_tensor(stddev_tensor, "NHWC")

    def run(self, input):
        return cvcuda.normalize(
            input,
            base=self.mean_tensor,
            scale=self.stddev_tensor,
            flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        )

    def visualize(self):
        pass
