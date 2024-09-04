# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvcv

# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401
from bench_utils import AbstractOpBase


class OpAsImageFromNVCVImage(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        # dummy run that does not use cache
        img = nvcv.Image((128, 128), nvcv.Format.RGBA8)

        self.imglist = []
        for _ in range(10):
            img = nvcv.Image((128, 128), nvcv.Format.RGBA8)
            self.imglist.append(img.cuda())
        self.cycle = 0

    def run(self, input):
        nvcv.as_image(self.imglist[self.cycle % len(self.imglist)])
        self.cycle += 1
        return
