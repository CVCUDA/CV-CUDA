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


class OpAdaptiveThreshold(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        self.maxval = 255.0
        self.adaptive_method = cvcuda.AdaptiveThresholdType.GAUSSIAN_C
        self.threshold_type = cvcuda.ThresholdType.BINARY
        self.block_size = 11
        self.c = 2
        self.grayscale_input = cvcuda.cvtcolor(input, cvcuda.ColorConversion.RGB2GRAY)

    def run(self, input):
        return cvcuda.adaptivethreshold(
            self.grayscale_input,
            max_value=self.maxval,
            adaptive_method=self.adaptive_method,
            threshold_type=self.threshold_type,
            block_size=self.block_size,
            c=self.c,
        )
