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

"""Run a TorchVision v2 training pipeline on CV-CUDA with cvcuda.tv.Compose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import torch
from torchvision.transforms import v2

common_dir = Path(__file__).parent.parent
sys.path.insert(0, str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Augment an image with a seeded torchvision pipeline and save a preview."""
    # docs_tag: begin_example
    args: argparse.Namespace = parse_image_args(output_name="cat_cvcuda_tv_compose.jpg")
    input_image: cvcuda.Tensor = read_image(args.input)
    # Convert HWC to CHW to demonstrate Torch output; Compose also accepts HWC.
    image = torch.as_tensor(input_image.cuda()).permute(2, 0, 1).contiguous()

    pipeline = cvcuda.tv.Compose(
        [
            v2.RandomResizedCrop(224, scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    torch.manual_seed(0)
    augmented = pipeline(image)
    torch.manual_seed(0)
    repeated = pipeline(image)
    assert (augmented.shape, augmented.dtype) == ((3, 224, 224), torch.float32)
    assert torch.equal(augmented, repeated)

    preview = cvcuda.tv.Compose(
        [v2.RandomResizedCrop(224, scale=(0.5, 1.0)), v2.RandomHorizontalFlip()]
    )
    torch.manual_seed(0)
    output_image = preview(image).permute(1, 2, 0).contiguous()
    write_image(cvcuda.as_tensor(output_image, "HWC"), args.output)
    assert args.output.is_file()
    # docs_tag: end_example


if __name__ == "__main__":
    main()
