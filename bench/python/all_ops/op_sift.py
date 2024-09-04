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


class OpSIFT(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        self.max_features = 100
        self.num_octave_layers = 3
        self.contrast_threshold = 0.04
        self.edge_threshold = 10.0
        self.init_sigma = 1.6
        self.grayscale_input = cvcuda.cvtcolor(
            self.input, cvcuda.ColorConversion.RGB2GRAY
        )

    def run(self, input):
        return cvcuda.sift(
            self.grayscale_input,
            self.max_features,
            self.num_octave_layers,
            self.contrast_threshold,
            self.edge_threshold,
            self.init_sigma,
            flags=cvcuda.SIFT.USE_EXPANDED_INPUT,
        )

    def visualize(self):
        pass
