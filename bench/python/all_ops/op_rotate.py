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


class OpRotate(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        self.angle_deg = 40
        self.shift = [input.shape[2] // 4, input.shape[1] // 4]
        self.interpolation_type = cvcuda.Interp.LINEAR

    def run(self, input):
        return cvcuda.rotate(input, self.angle_deg, self.shift, self.interpolation_type)
