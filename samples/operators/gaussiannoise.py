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
"""Simple Gaussian Noise example with CVCUDA."""

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
    """Add Gaussian noise to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_gaussiannoise.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_gaussiannoise_setup
    # Wrap the HWC single image in a batch dimension (N=1) so the
    # gaussiannoise operator can receive per-image mu/sigma tensors.
    nhwc_image: cvcuda.Tensor = input_image.reshape((1, *input_image.shape), "NHWC")

    # mu and sigma are per-image scalars with layout "N".
    # mu=0 means zero-mean (no brightness shift), sigma controls noise strength.
    sigma_value = 25.0  # visible but not destructive for an 8-bit image
    mu_host = np.array([0.0], dtype=np.float32)
    sigma_host = np.array([sigma_value], dtype=np.float32)

    mu_tensor = cvcuda.Tensor((1,), cvcuda.Type.F32, "N")
    sigma_tensor = cvcuda.Tensor((1,), cvcuda.Type.F32, "N")
    cuda_memcpy_h2d(mu_host, mu_tensor.cuda())
    cuda_memcpy_h2d(sigma_host, sigma_tensor.cuda())
    # docs_tag: end_gaussiannoise_setup

    # docs_tag: begin_gaussiannoise
    # Apply Gaussian noise.  per_channel=False applies the same noise sample
    # to every channel; seed fixes the random state for reproducibility.
    noisy_batch: cvcuda.Tensor = cvcuda.gaussiannoise(
        src=nhwc_image,
        mu=mu_tensor,
        sigma=sigma_tensor,
        per_channel=False,
        seed=42,
    )
    # docs_tag: end_gaussiannoise

    # The output is NHWC uint8.  Reshape to HWC by dropping the batch dimension
    # and write directly — no unnecessary host round-trip needed.
    write_image(noisy_batch.reshape(input_image.shape, "HWC"), args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
