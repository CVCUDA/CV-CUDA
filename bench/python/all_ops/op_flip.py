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


class OpFlipX(AbstractOpBase):
    def setup(self, input):
        self.flip_code = 0  # means flipping around x axis.

    def run(self, input):
        return cvcuda.flip(input, flipCode=self.flip_code)


class OpFlipY(AbstractOpBase):
    def setup(self, input):
        self.flip_code = 1  # means flipping around y axis.

    def run(self, input):
        return cvcuda.flip(input, flipCode=self.flip_code)


class OpFlipXY(AbstractOpBase):
    def setup(self, input):
        self.flip_code = -1  # means flipping around x and y axis.

    def run(self, input):
        return cvcuda.flip(input, flipCode=self.flip_code)
