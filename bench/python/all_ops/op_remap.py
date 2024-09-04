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
import numpy as np
import torch


class OpRemap(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        batch_size, width, height = input.shape[0], input.shape[2], input.shape[1]
        batch_map = np.stack([self.flipH(w=width, h=height) for _ in range(batch_size)])
        batch_map = torch.as_tensor(batch_map, device="cuda")
        self.batch_map = cvcuda.as_tensor(batch_map, "NHWC")
        self.src_interp = cvcuda.Interp.LINEAR
        self.map_interp = cvcuda.Interp.LINEAR
        self.map_type = cvcuda.Remap.ABSOLUTE
        self.align_corners = True
        self.border_type = cvcuda.Border.CONSTANT
        self.border_value = np.array([], dtype=np.float32)

    def flipH(self, w, h):
        mesh = np.meshgrid(np.arange(w)[::-1], np.arange(h))
        return np.stack(mesh, axis=2).astype(np.float32)

    def run(self, input):
        return cvcuda.remap(
            input,
            self.batch_map,
            self.src_interp,
            self.map_interp,
            self.map_type,
            align_corners=self.align_corners,
            border=self.border_type,
            border_value=self.border_value,
        )
