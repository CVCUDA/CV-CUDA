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
"""Simple Threshold example with CVCUDA."""

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
    """Apply binary threshold to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_threshold.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_threshold_setup
    # threshold() requires a batch dimension (NHWC), so wrap the HWC image.
    # One thresh/maxval scalar is needed per image in the batch.
    nhwc_image: cvcuda.Tensor = input_image.reshape((1, *input_image.shape), "NHWC")
    batch_size = nhwc_image.shape[0]

    # Allocate per-image threshold and maxval tensors on the GPU (dtype F64, layout "N").
    thresh_host = np.array([128.0] * batch_size, dtype=np.float64)
    thresh_tensor = cvcuda.Tensor((batch_size,), dtype=np.float64, layout="N")
    cuda_memcpy_h2d(thresh_host, thresh_tensor.cuda())

    maxval_host = np.array([255.0] * batch_size, dtype=np.float64)
    maxval_tensor = cvcuda.Tensor((batch_size,), dtype=np.float64, layout="N")
    cuda_memcpy_h2d(maxval_host, maxval_tensor.cuda())
    # docs_tag: end_threshold_setup

    # docs_tag: begin_threshold
    # Apply BINARY threshold: pixels > thresh become maxval, others become 0.
    # The operator returns an NHWC tensor of the same shape and dtype as the input.
    thresholded: cvcuda.Tensor = cvcuda.threshold(
        src=nhwc_image,
        thresh=thresh_tensor,
        maxval=maxval_tensor,
        type=cvcuda.ThresholdType.BINARY,
    )
    # docs_tag: end_threshold

    # Strip the batch dimension back to HWC so write_image accepts it.
    out_hwc: cvcuda.Tensor = thresholded.reshape(thresholded.shape[1:], "HWC")
    write_image(out_hwc, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
