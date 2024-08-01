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

import cvcuda
import nvcv
import torch

# TODO: These tests technically belong to nvcv, but since it doesn't expose any
# operator (or anything that we could submit to a stream), we need to add this
# to cvcuda instead. Ideally nvcv should allows us to write proper stream tests
# using only the facilities it provides. Maybe a "noop" operator, or some mocks,
# maybe container copy, etc.


def test_stream_gcbag_vs_streamsync_race_condition():
    inputImage = torch.randint(0, 256, (100, 1500, 1500, 3), dtype=torch.uint8).cuda()
    nvcvInputTensor = nvcv.as_tensor(inputImage, "NHWC")
    inputmap = torch.randint(0, 256, (100, 1500, 1500, 2), dtype=torch.float).cuda()
    nvcvInputMap = nvcv.as_tensor(inputmap, "NHWC")

    cvcuda_stream = cvcuda.Stream()
    with cvcuda_stream:
        nvcvResizeTensor = cvcuda.remap(nvcvInputTensor, nvcvInputMap)
    del nvcvResizeTensor
