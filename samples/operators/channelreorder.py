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
"""Simple Channel Reorder example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import (  # noqa: E402
    parse_image_args,
    read_image,
    write_image,
    cuda_memcpy_h2d,
)


def main() -> None:
    """Reorder channels of an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_channelreorder.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_channelreorder_setup
    # channelreorder operates on ImageBatchVarShape rather than plain tensors,
    # so we wrap the HWC tensor as a cvcuda.Image and add it to a batch.
    # cvcuda.as_image ties the buffer lifetime to the Image object.
    cv_image: cvcuda.Image = cvcuda.as_image(
        input_image.cuda(), format=cvcuda.Format.RGB8
    )

    batch = cvcuda.ImageBatchVarShape(1)
    batch.pushback(cv_image)

    # Build the orders tensor with layout "NC" (num_images × num_channels).
    # Each row lists, for every output channel, which input channel index to read.
    # [2, 1, 0] maps R→B, G→G, B→R — a standard RGB-to-BGR channel swap.
    order_data = np.array([[2, 1, 0]], dtype=np.int32)
    orders: cvcuda.Tensor = cvcuda.Tensor((1, 3), dtype=np.int32, layout="NC")
    cuda_memcpy_h2d(order_data, orders.cuda())
    # docs_tag: end_channelreorder_setup

    # docs_tag: begin_channelreorder
    # Run the channel-reorder operator.  The output is an ImageBatchVarShape
    # of the same format (RGB8) — channels are permuted in-place on the GPU.
    out_batch: cvcuda.ImageBatchVarShape = cvcuda.channelreorder(batch, orders)

    # Extract the single result image from the batch and convert back to a
    # cvcuda.Tensor in HWC layout so write_image can encode it.
    out_image: cvcuda.Image = list(out_batch)[0]
    out_tensor: cvcuda.Tensor = cvcuda.as_tensor(out_image.cuda(), "HWC")
    # docs_tag: end_channelreorder

    write_image(out_tensor, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
