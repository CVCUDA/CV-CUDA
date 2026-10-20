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


def create_tensor1():
    tensor1 = cvcuda.Tensor(  # noqa: F841
        (16, 32, 4), np.float32, cvcuda.TensorLayout.HWC
    )


def create_tensor2():
    # The wider, shorter tensor fits in the first allocation.
    tensor2 = cvcuda.Tensor(  # noqa: F841
        (4, 64, 4), np.float32, cvcuda.TensorLayout.HWC
    )


def main() -> None:
    create_tensor1()
    # tensor1 runs out of scope after leaving create_tensor1()
    create_tensor2()


# docs-end: main


if __name__ == "__main__":
    main()
