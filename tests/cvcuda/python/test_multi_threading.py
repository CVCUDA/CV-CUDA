# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import pytest as t
import cvcuda_util as util


def test_parallel_resource_submit():
    """Check that Resource::submitSync is thread-safe"""

    def submit(thread_no: int, resource: cvcuda.Resource) -> None:
        for _ in range(10000):
            resource.submitStreamSync(cvcuda.Stream())

    # Easiest way to get a Resource from Python
    resource = cvcuda.Tensor((720, 1280), np.uint8)
    util.run_parallel(submit, resource)


# data copied from test_obnbox.py
@t.mark.parametrize(
    "inputp, bndboxes",
    [
        (
            (((3, 224, 224, 4), np.uint8, "NHWC")),
            cvcuda.BndBoxesI(
                boxes=[
                    [
                        cvcuda.BndBoxI(
                            box=(10, 10, 5, 5),
                            thickness=2,
                            borderColor=(255, 255, 0),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.BndBoxI(
                            box=(20, 10, 5, 5),
                            thickness=3,
                            borderColor=(0, 255, 255),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.BndBoxI(
                            box=(30, 10, 5, 5),
                            thickness=3,
                            borderColor=(0, 255, 255),
                            fillColor=(0, 128, 255, 128),
                        ),
                    ],
                    [
                        cvcuda.BndBoxI(
                            box=(10, 20, 5, 5),
                            thickness=2,
                            borderColor=(255, 255, 0),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.BndBoxI(
                            box=(20, 20, 5, 5),
                            thickness=3,
                            borderColor=(0, 255, 255),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.BndBoxI(
                            box=(30, 20, 5, 5),
                            thickness=3,
                            borderColor=(0, 255, 255),
                            fillColor=(0, 128, 255, 128),
                        ),
                    ],
                    [
                        cvcuda.BndBoxI(
                            box=(10, 20, 5, 5),
                            thickness=2,
                            borderColor=(255, 255, 0),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.BndBoxI(
                            box=(20, 20, 5, 5),
                            thickness=3,
                            borderColor=(0, 255, 255),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.BndBoxI(
                            box=(30, 20, 5, 5),
                            thickness=3,
                            borderColor=(0, 255, 255),
                            fillColor=(0, 128, 255, 128),
                        ),
                    ],
                ],
            ),
        ),
    ],
)
def test_parallel_bndbox(inputp, bndboxes):
    """
    With a global cache, the bndbox operator crashes when used
    in parallel with the GIL disabled.
    """

    def bndbox(thread_no: int, input: cvcuda.Tensor):
        out = cvcuda.bndbox(input, bndboxes)
        outputs.append(out)

    input = cvcuda.Tensor(*inputp)
    outputs = []
    util.run_parallel(bndbox, input)

    for out in outputs:
        assert out.layout == input.layout
        assert out.shape == input.shape
        assert out.dtype == input.dtype
