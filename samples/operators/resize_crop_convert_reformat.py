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
"""Simple Resize Crop Convert Reformat example with CVCUDA."""

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
    download_tensor,
    upload_tensor,
)


def nchw_f32_to_hwc_u8(tensor: cvcuda.Tensor) -> cvcuda.Tensor:
    """Convert an NCHW float32 tensor back to HWC uint8 for saving.

    The batch dimension is removed (first image only) and the channel-last
    memory layout expected by nvimgcodec is restored.

    Args:
        tensor: NCHW or CHW float32 tensor from resize_crop_convert_reformat.

    Returns:
        HWC uint8 cvcuda.Tensor suitable for write_image.
    """
    # Download to host so we can reorder axes and clip.  download_tensor honours
    # the NCHW row pitch (the W rows are padded to an alignment boundary).
    host = download_tensor(tensor)

    # Drop the batch dimension when present (NCHW -> CHW)
    if host.ndim == 4:
        host = host[0]  # take first image in the batch

    # CHW -> HWC and scale back to [0, 255]
    hwc = np.transpose(host, (1, 2, 0))
    hwc = np.clip(hwc, 0.0, 255.0).astype(np.uint8)
    hwc = np.ascontiguousarray(hwc)

    h, w, c = hwc.shape
    out_tensor = cvcuda.Tensor((h, w, c), cvcuda.Type.U8, "HWC")
    upload_tensor(hwc, out_tensor)
    return out_tensor


def main() -> None:
    """Resize, crop, convert, and reformat an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_resize_crop_convert_reformat.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_resize_crop_convert_reformat_setup
    # Wrap the single HWC image in a batch dimension so we can use the NHWC
    # path, which also demonstrates the typical DL pipeline usage pattern.
    nhwc_image: cvcuda.Tensor = input_image.reshape(
        (1, *input_image.shape), layout="NHWC"
    )

    # Target resize and crop dimensions.  We resize to the requested
    # height x width (224x224 by default), then crop a region of the same size
    # from the top-left corner — a typical pre-processing pipeline.
    resize_dim = (args.height, args.width)  # (H, W) after resize
    crop_w = min(args.width, resize_dim[1])
    crop_h = min(args.height, resize_dim[0])
    crop_rect = cvcuda.RectI(0, 0, crop_w, crop_h)
    # docs_tag: end_resize_crop_convert_reformat_setup

    # docs_tag: begin_resize_crop_convert_reformat
    # Fused pipeline: resize → crop → convert to float32 → reformat to NCHW.
    # ChannelManip.REVERSE swaps BGR↔RGB in one pass, which is common when
    # feeding models trained with a different channel ordering than the codec.
    output: cvcuda.Tensor = cvcuda.resize_crop_convert_reformat(
        nhwc_image,
        resize_dim,
        cvcuda.Interp.LINEAR,
        crop_rect,
        layout="NCHW",
        data_type=cvcuda.Type.F32,
        manip=cvcuda.ChannelManip.REVERSE,
        scale=1.0,
        offset=0.0,
    )
    # docs_tag: end_resize_crop_convert_reformat

    # The output is NCHW float32; convert back to HWC uint8 so that the
    # image encoder can save it as a standard JPEG.
    viewable: cvcuda.Tensor = nchw_f32_to_hwc_u8(output)
    write_image(viewable, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
