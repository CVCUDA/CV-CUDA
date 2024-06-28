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
import numpy as np


class OpWarpAffine(AbstractOpBase):
    def setup(self, input):
        self.xform = np.array(
            [[1.26666667, 0.6, -83.33333333], [-0.33333333, 1.0, 66.66666667]]
        )
        self.flags = cvcuda.Interp.LINEAR
        self.border_mode = cvcuda.Border.CONSTANT
        self.border_value = []

    def run(self, input):
        return cvcuda.warp_affine(
            input,
            self.xform,
            flags=self.flags,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )


class OpWarpAffineInverse(AbstractOpBase):
    def setup(self, input):
        self.xform = np.array(
            [[1.26666667, 0.6, -83.33333333], [-0.33333333, 1.0, 66.66666667]]
        )
        self.flags = cvcuda.Interp.LINEAR
        self.border_mode = cvcuda.Border.CONSTANT
        self.border_value = []

    def run(self, input):
        return cvcuda.warp_affine(
            input,
            self.xform,
            flags=self.flags,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )