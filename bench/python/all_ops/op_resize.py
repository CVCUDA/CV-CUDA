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


class OpResizeDown(AbstractOpBase):
    def setup(self, input):
        self.resize_width = 640
        self.resize_height = 420

    def run(self, input):
        return cvcuda.resize(
            input,
            (
                input.shape[0],
                self.resize_height,
                self.resize_width,
                input.shape[3],
            ),
            cvcuda.Interp.AREA,
        )


class OpResizeUp(AbstractOpBase):
    def setup(self, input):
        self.resize_width = 1920
        self.resize_height = 1280

    def run(self, input):
        return cvcuda.resize(
            input,
            (
                input.shape[0],
                self.resize_height,
                self.resize_width,
                input.shape[3],
            ),
            cvcuda.Interp.LINEAR,
        )
