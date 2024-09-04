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


class OpWarpPerspective(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        self.xform = np.array(
            [
                [3.46153846e-01, 3.33031674e-01, 1.28000000e02],
                [0.00000000e00, 6.92307692e-01, 0.00000000e00],
                [-4.50721154e-04, 5.65610860e-04, 1.00000000e00],
            ],
            np.float32,
        )
        self.flags = cvcuda.Interp.LINEAR
        self.border_mode = cvcuda.Border.CONSTANT
        self.border_value = []

    def run(self, input):
        return cvcuda.warp_perspective(
            input,
            self.xform,
            flags=self.flags,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )


class OpWarpPerspectiveInverse(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        self.xform = np.array(
            [
                [3.46153846e-01, 3.33031674e-01, 1.28000000e02],
                [0.00000000e00, 6.92307692e-01, 0.00000000e00],
                [-4.50721154e-04, 5.65610860e-04, 1.00000000e00],
            ],
            np.float32,
        )
        self.flags = cvcuda.Interp.LINEAR | cvcuda.Interp.WARP_INVERSE_MAP
        self.border_mode = cvcuda.Border.CONSTANT
        self.border_value = []

    def run(self, input):
        return cvcuda.warp_perspective(
            input,
            self.xform,
            flags=self.flags,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )
