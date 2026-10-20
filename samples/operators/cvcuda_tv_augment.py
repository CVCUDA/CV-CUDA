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
"""Apply cvcuda.tv augmentation operators to an image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.insert(0, str(common_dir))

from common import (  # noqa: E402
    parse_image_args,
    read_image,
    upload_tensor,
    write_image,
)


def main() -> None:
    """Erase part of an image and add JPEG artifacts."""
    # docs_tag: begin_example
    args: argparse.Namespace = parse_image_args(output_name="cat_cvcuda_tv_augment.jpg")
    input_image: cvcuda.Tensor = read_image(args.input)
    height, width, _ = input_image.shape

    fill_host = np.array([[[32]], [[96]], [[224]]], dtype=np.uint8)
    fill = cvcuda.Tensor(fill_host.shape, dtype=np.uint8, layout="CHW")
    upload_tensor(fill_host, fill)

    erase_height = max(1, height // 4)
    erase_width = max(1, width // 4)
    erased = cvcuda.tv.erase(
        input_image,
        (height - erase_height) // 2,
        (width - erase_width) // 2,
        erase_height,
        erase_width,
        fill,
    )
    output_image = cvcuda.tv.jpeg(erased, quality=65)

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
