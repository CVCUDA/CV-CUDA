# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Apply cvcuda.tv miscellaneous operators to an image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.insert(0, str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Prepare an image for a model and create a noisy, blurred preview."""
    # docs_tag: begin_example
    args: argparse.Namespace = parse_image_args(output_name="cat_cvcuda_tv_misc.jpg")
    input_image: cvcuda.Tensor = read_image(args.input)
    stream = cvcuda.Stream()

    float_image = cvcuda.tv.to_dtype(input_image, np.float32, scale=True, stream=stream)
    preprocessed = cvcuda.tv.normalize(
        float_image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        stream=stream,
    )

    blurred = cvcuda.tv.gaussian_blur(input_image, kernel_size=[5, 5], stream=stream)
    output_image = cvcuda.tv.gaussian_noise(blurred, sigma=0.03, seed=42, stream=stream)
    stream.sync()

    assert (preprocessed.shape, preprocessed.layout, preprocessed.dtype) == (
        input_image.shape,
        input_image.layout,
        cvcuda.Type.F32,
    )
    assert (output_image.shape, output_image.layout, output_image.dtype) == (
        input_image.shape,
        input_image.layout,
        input_image.dtype,
    )
    write_image(output_image, args.output)
    assert args.output.is_file()
    # docs_tag: end_example


if __name__ == "__main__":
    main()
