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
"""Simple Crop Flip Normalize Reformat example with CVCUDA."""

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


def main() -> None:
    """Crop, flip, normalize, and reformat an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_crop_flip_normalize_reformat.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_crop_flip_normalize_reformat_setup
    # crop_flip_normalize_reformat operates on an ImageBatchVarShape, so
    # wrap the single HWC tensor into a one-image batch.
    height, width, channels = input_image.shape
    cvcuda_image = cvcuda.as_image(input_image.cuda(), cvcuda.Format.RGB8)
    batch = cvcuda.ImageBatchVarShape(1)
    batch.pushback([cvcuda_image])

    # Define a crop rectangle [crop_x, crop_y, crop_width, crop_height] per image.
    # We crop the central 80% of the image so the result still looks good.
    crop_x = int(width * 0.1)
    crop_y = int(height * 0.1)
    crop_w = int(width * 0.8)
    crop_h = int(height * 0.8)
    crop_data = np.array([[[[crop_x, crop_y, crop_w, crop_h]]]], dtype=np.int32)
    crop_rect_host = np.ascontiguousarray(crop_data)

    # Allocate crop rect tensor on GPU (shape: [N, 1, 1, 4], layout NHWC)
    crop_tensor = cvcuda.Tensor((1, 1, 1, 4), np.int32, "NHWC")
    upload_tensor(crop_rect_host, crop_tensor)

    # flip_code per image: 1 = flip around y-axis (horizontal flip)
    flip_data = np.array([[1]], dtype=np.int32)
    flip_host = np.ascontiguousarray(flip_data)
    flip_tensor = cvcuda.Tensor((1, 1), np.int32, "NC")
    upload_tensor(flip_host, flip_tensor)

    # Normalization parameters: base (mean) and scale (std-dev) per channel.
    # Using ImageNet-style mean and std for demonstration.
    base_data = np.array([[[[0.485, 0.456, 0.406]]]], dtype=np.float32)
    scale_data = np.array([[[[0.229, 0.224, 0.225]]]], dtype=np.float32)
    base_host = np.ascontiguousarray(base_data)
    scale_host = np.ascontiguousarray(scale_data)

    base_tensor = cvcuda.Tensor((1, 1, 1, 3), np.float32, "NHWC")
    scale_tensor = cvcuda.Tensor((1, 1, 1, 3), np.float32, "NHWC")
    upload_tensor(base_host, base_tensor)
    upload_tensor(scale_host, scale_tensor)

    # The output will be in NCHW layout (planar), float32, at the cropped size.
    out_shape = (1, channels, crop_h, crop_w)
    # docs_tag: end_crop_flip_normalize_reformat_setup

    # docs_tag: begin_crop_flip_normalize_reformat
    # Run the combined crop + horizontal-flip + normalize + reformat pipeline in one
    # GPU kernel.  SCALE_IS_STDDEV tells the operator to treat 'scale' as std-dev
    # so it divides (pixel/255 - mean) / std to produce a zero-centred float tensor.
    normalized_nchw: cvcuda.Tensor = cvcuda.crop_flip_normalize_reformat(
        batch,
        out_shape=out_shape,
        out_dtype=np.float32,
        out_layout="NCHW",
        rect=crop_tensor,
        flip_code=flip_tensor,
        base=base_tensor,
        scale=scale_tensor,
        globalscale=1.0 / 255.0,
        globalshift=0.0,
        epsilon=1e-8,
        flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        border=cvcuda.Border.REPLICATE,
        bvalue=0.0,
    )
    # docs_tag: end_crop_flip_normalize_reformat

    # Reverse the normalization on the host so the result is a viewable uint8 image.
    # normalized pixel = (raw/255 - mean) / std  =>  raw = clip((normalized*std + mean)*255, 0, 255)
    # download_tensor honours the NCHW row pitch (W rows are padded).
    host_nchw = download_tensor(normalized_nchw)

    # Reorder NCHW -> HWC for display (squeeze the batch dim)
    host_chw = host_nchw[0]  # (C, H, W)
    host_hwc = np.transpose(host_chw, (1, 2, 0))  # (H, W, C)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    host_uint8 = np.clip((host_hwc * std + mean) * 255.0, 0, 255).astype(np.uint8)

    # Upload the recovered uint8 image back to GPU for write_image.
    # upload_tensor honours the tensor's row pitch, so no contiguous-reshape
    # trick is needed.
    hwc_shape = host_uint8.shape  # (H, W, C)
    out_tensor = cvcuda.Tensor(hwc_shape, dtype=np.uint8, layout="HWC")
    upload_tensor(np.ascontiguousarray(host_uint8), out_tensor)

    write_image(out_tensor, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
