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
import nvcv
import cvcuda


class OpResizeCropConvertReformat(AbstractOpBase):
    def setup(self, input):
        super().setup(input)
        resize = 256
        crop = 224
        delta_shape = resize - crop
        start = delta_shape // 2
        self.resize_dim = (resize, resize)
        self.resize_interpolation = cvcuda.Interp.LINEAR
        self.crop_rect = cvcuda.RectI(start, start, crop, crop)

    def run(self, input):
        return cvcuda.resize_crop_convert_reformat(
            input,
            self.resize_dim,
            self.resize_interpolation,
            self.crop_rect,
            layout="NHWC",
            data_type=nvcv.Type.U8,
            manip=cvcuda.ChannelManip.REVERSE,
        )
