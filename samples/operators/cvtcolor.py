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
"""Simple Color Conversion example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Convert image color space with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_cvtcolor.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_cvtcolor
    # cvtcolor requires a batched (NHWC) tensor, so wrap the HWC image in a batch
    # dimension using cvcuda.stack before passing it to the operator.
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])

    # Swap the R and B channels (RGB2BGR) — the output is a visually distinct
    # but still fully viewable 3-channel uint8 image, making the conversion easy
    # to verify by eye (warm tones shift to cool and vice versa).
    converted: cvcuda.Tensor = cvcuda.cvtcolor(
        nhwc_image, code=cvcuda.ColorConversion.RGB2BGR
    )

    # Drop the batch dimension back to HWC so write_image can encode the result.
    output_image: cvcuda.Tensor = converted.reshape(converted.shape[1:], "HWC")
    write_image(output_image, args.output)
    # docs_tag: end_cvtcolor
    # docs_tag: end_main


if __name__ == "__main__":
    main()
