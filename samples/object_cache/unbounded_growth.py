# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

# docs-start: main
import cvcuda
import numpy as np


def create_tensor(h: int, w: int) -> None:
    _ = cvcuda.Tensor((h, w, 3), np.float32, cvcuda.TensorLayout.HWC)


def main(iters: int = 100) -> None:
    for i in range(iters):
        create_tensor(1000 + i, 1000)


# docs-end: main


if __name__ == "__main__":
    main(iters=100)
