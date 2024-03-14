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
import torch


class MorphologyBase:
    def __init__(self, device_id, input, morphology_type):
        self.device_id = device_id
        self.mask_size = [5, 5]
        self.anchor = [-1, -1]
        self.num_iterations = 3
        self.border_type = cvcuda.Border.CONSTANT
        self.morphology_type = morphology_type

        # Morphology requires binary input, with mostly white foreground
        threshold_value = torch.tensor([150.0] * input.shape[0])
        threshold_value = threshold_value.type(torch.float64)
        threshold_value = threshold_value.cuda(self.device_id)
        threshold_value = cvcuda.as_tensor(threshold_value, "N")

        maxval = torch.tensor([255.0] * input.shape[0])
        maxval = maxval.type(torch.float64)
        maxval = maxval.cuda(self.device_id)
        maxval = cvcuda.as_tensor(maxval, "N")
        self.binary_input = cvcuda.threshold(
            input, threshold_value, maxval, type=cvcuda.ThresholdType.BINARY
        )

        if self.num_iterations > 1:
            self.workspace = cvcuda.Tensor(input.shape, input.dtype, "NHWC")
        else:
            self.workspace = None

    def __call__(self):
        return cvcuda.morphology(
            self.binary_input,
            self.morphology_type,
            maskSize=self.mask_size,
            anchor=self.anchor,
            workspace=self.workspace,
            iteration=self.num_iterations,
            border=self.border_type,
        )


class OpMorphologyOpen(AbstractOpBase):
    def setup(self, input):
        self.MorphologyBase = MorphologyBase(
            self.device_id, input, cvcuda.MorphologyType.OPEN
        )

    def run(self, input):
        return self.MorphologyBase()


class OpMorphologyClose(AbstractOpBase):
    def setup(self, input):
        self.MorphologyBase = MorphologyBase(
            self.device_id, input, cvcuda.MorphologyType.CLOSE
        )

    def run(self, input):
        return self.MorphologyBase()


class OpMorphologyDilate(AbstractOpBase):
    def setup(self, input):
        self.MorphologyBase = MorphologyBase(
            self.device_id, input, cvcuda.MorphologyType.DILATE
        )

    def run(self, input):
        return self.MorphologyBase()


class OpMorphologyErode(AbstractOpBase):
    def setup(self, input):
        self.MorphologyBase = MorphologyBase(
            self.device_id, input, cvcuda.MorphologyType.ERODE
        )

    def run(self, input):
        return self.MorphologyBase()
