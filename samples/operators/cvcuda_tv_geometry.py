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
"""Apply cvcuda.tv geometry operators to an image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.insert(0, str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Resize and rotate an image."""
    # docs_tag: begin_example
    args: argparse.Namespace = parse_image_args(
        output_name="cat_cvcuda_tv_geometry.jpg"
    )
    input_image: cvcuda.Tensor = read_image(args.input)

    resized = cvcuda.tv.resize(input_image, [args.height, args.width])
    output_image = cvcuda.tv.rotate(resized, angle=15.0)

    assert (output_image.shape, output_image.layout, output_image.dtype) == (
        (args.height, args.width, input_image.shape[-1]),
        input_image.layout,
        input_image.dtype,
    )
    write_image(output_image, args.output)
    assert args.output.is_file()
    # docs_tag: end_example


if __name__ == "__main__":
    main()
