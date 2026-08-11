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
"""Simple HQ Resize example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """High-quality resize an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_hq_resize.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_hq_resize
    # HQ Resize uses separate interpolation filters for downscaling (min) and
    # upscaling (mag), which produces sharper results than standard resize.
    # LANCZOS for minification avoids moiré patterns; LINEAR for magnification
    # is fast and smooth. antialias=True applies a low-pass filter before
    # downscaling to further suppress aliasing.
    output_image: cvcuda.Tensor = cvcuda.hq_resize(
        input_image,
        (args.height, args.width),
        min_interpolation=cvcuda.Interp.LANCZOS,
        mag_interpolation=cvcuda.Interp.LINEAR,
        antialias=True,
    )
    write_image(output_image, args.output)
    # docs_tag: end_hq_resize
    # docs_tag: end_main


if __name__ == "__main__":
    main()
