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
"""Simple connected components labeling example with CVCUDA."""

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


def color_labels_nhwc(labels):
    """Convert a label map to an RGB image

    Args:
        labels : Output of cvcuda.label operator

    Returns:
        cvcuda.Tensor: RGB image, with each label having a unique color
    """
    # Download labels from GPU
    host_labels = download_tensor(labels)

    # Create RGB output on CPU (contiguous)
    rgb_shape = (host_labels.shape[0], host_labels.shape[1], host_labels.shape[2], 3)
    a_rgb = np.zeros(rgb_shape, dtype=np.uint8)

    # Colorize labels
    for n in range(host_labels.shape[0]):
        a_labels = np.unique(host_labels[n, :, :, :])
        for label in a_labels:
            np.random.seed(label)
            rgb_label_color = np.random.randint(0, 256, 3, dtype=np.uint8)
            mask = host_labels[n] == label
            a_rgb[n][mask[:, :, 0]] = rgb_label_color

    # Allocate an NHWC tensor and upload the colorized labels.  upload_tensor
    # honours the tensor's row pitch, so no contiguous-reshape trick is needed.
    nhwc_tensor = cvcuda.Tensor(rgb_shape, dtype=np.uint8, layout="NHWC")
    upload_tensor(a_rgb, nhwc_tensor)

    # Return the NHWC tensor
    return nhwc_tensor


def main() -> None:
    """Connected components labeling an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_labeled.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_preprocessing
    # 1. Grayscale and histogram equalize the image
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])
    gray_image: cvcuda.Tensor = cvcuda.cvtcolor(
        nhwc_image, cvcuda.ColorConversion.RGB2GRAY
    )
    histogram_image: cvcuda.Tensor = cvcuda.histogrameq(gray_image, cvcuda.Type.U8)

    # 2. Compute threshold
    tp_host = np.array([128], dtype=np.float64)
    tp = cvcuda.Tensor((1,), dtype=np.float64, layout="N")
    upload_tensor(tp_host, tp)

    mp_host = np.array([255], dtype=np.float64)
    mp = cvcuda.Tensor((1,), dtype=np.float64, layout="N")
    upload_tensor(mp_host, mp)

    threshold_image: cvcuda.Tensor = cvcuda.threshold(
        histogram_image, tp, mp, cvcuda.ThresholdType.BINARY
    )
    # docs_tag: end_preprocessing

    # docs_tag: begin_labeling
    # 3. Connected components labeling
    cc_labels, _, _ = cvcuda.label(threshold_image)
    # docs_tag: end_labeling

    # docs_tag: begin_visualization
    # 4. Generate and save the visualization image
    argb_image: cvcuda.Tensor = color_labels_nhwc(cc_labels)
    argb_image = argb_image.reshape((*argb_image.shape[1:],), "HWC")
    write_image(argb_image, args.output)
    # docs_tag: end_visualization
    # docs_tag: end_main


if __name__ == "__main__":
    main()
