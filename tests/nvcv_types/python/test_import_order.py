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

# Import order is important,
# torch must be loaded correctly even if nvcv was imported first
import nvcv
import torch
import numpy as np


def test_import_nvcv_first_works():
    torch.as_tensor(np.ndarray((4, 6), dtype=np.uint8), device="cuda")
    nvcv.Tensor((4, 6), dtype=np.uint8)
