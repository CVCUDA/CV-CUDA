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
"""
Simple application showing how to build an application with CV-CUDA.

This application demonstrates how to do the following:

1. Load an image(s) into CV-CUDA from disk.
2. Resize the image(s).
3. If multiple images are loaded, batch them into a single CV-CUDA tensor.
4. Apply a Gaussian blur on the batch.
5. Save each image back to disk.

All without leaving the GPU.

Example Usage:
-------------
Default args (uses the tabby_tiger_cat.jpg image in the assets directory and writes to cvcuda/.cache):
    python3 hello_world.py

Single image:
    python3 hello_world.py -i input.jpg -o output.jpg

Multiple images:
    python3 hello_world.py -i img1.jpg img2.jpg img3.jpg -o out1.jpg out2.jpg out3.jpg

With custom parameters:
    python3 hello_world.py -i input.jpg -o output.jpg --width 512 --height 512 -k 7 -s 2.0

Using absolute paths:
    python3 hello_world.py -i /path/to/input.jpg -o /path/to/output.jpg

With all options:
    python3 hello_world.py \\
        --inputs image1.png image2.png \\
        --outputs result1.png result2.png \\
        --width 224 \\
        --height 224 \\
        --kernel 5 \\
        --sigma 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import cvcuda

from nvidia import nvimgcodec

if TYPE_CHECKING:
    from typing_extensions import Generator

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import get_cache_dir, zero_copy_split  # noqa: E402


@contextmanager
def timer(tag: str) -> Generator[None, None, None]:
    """
    Context manager to time a block of code.

    Args:
        tag: Tag to print.
    """
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f"{tag}: ")  # noqa: E231
    print(f"  Time: {(t1 - t0) * 1000:.2f} ms")  # noqa: E231
    print("--------------------------------")


def main() -> None:
    """Hello world CV-CUDA application."""
    with timer("Parse Arguments"):
        # docs_tag: begin_main
        # docs_tag: begin_parse_args
        parser = argparse.ArgumentParser(description="Hello world CV-CUDA application.")
        parser.add_argument(
            "--inputs",
            "-i",
            type=Path,
            nargs="+",
            default=[
                Path(__file__).parent.parent
                / "assets"
                / "images"
                / "tabby_tiger_cat.jpg"
            ],
            help="Input image files.",
        )
        parser.add_argument(
            "--outputs",
            "-o",
            type=Path,
            nargs="+",
            default=[get_cache_dir() / "cat_hw.jpg"],
            help="Output image files.",
        )
        parser.add_argument(
            "--width", type=int, default=224, help="Width of the image."
        )
        parser.add_argument(
            "--height", type=int, default=224, help="Height of the image."
        )
        parser.add_argument(
            "--kernel",
            "-k",
            type=int,
            default=5,
            help="Kernel size of the Gaussian blur.",
        )
        parser.add_argument(
            "--sigma", "-s", type=float, default=1.0, help="Sigma of the Gaussian blur."
        )
        args = parser.parse_args()

        # Parse and validate input paths
        input_paths: list[Path] = args.inputs
        output_paths: list[Path] = args.outputs

        # Validate input paths
        supported_formats = {
            ".jpg",
            ".jpeg",
            ".png",
        }
        for input_path in input_paths:
            if not input_path.exists():
                parser.error(f"Input file does not exist: {input_path}")
            if not input_path.is_file():
                parser.error(f"Input path is not a file: {input_path}")
            if input_path.suffix.lower() not in supported_formats:
                parser.error(
                    f"Unsupported image format: {input_path.suffix}. "
                    f"Supported formats: {', '.join(sorted(supported_formats))}"
                )

        # Validate output paths
        for output_path in output_paths:
            if output_path.parent and not output_path.parent.exists():
                parser.error(f"Output directory does not exist: {output_path.parent}")
            if output_path.suffix.lower() not in supported_formats:
                parser.error(
                    f"Unsupported output format: {output_path.suffix}. "
                    f"Supported formats: {', '.join(sorted(supported_formats))}"
                )

        # Validate that number of inputs matches number of outputs
        if len(input_paths) != len(output_paths):
            parser.error(
                f"Number of input files ({len(input_paths)}) must match "
                f"number of output files ({len(output_paths)})"
            )
        # docs_tag: end_parse_args

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("--------------------------------")

    with timer("Load images"):
        # docs_tag: begin_load_images
        # 1. Load the images into CV-CUDA
        decoder = nvimgcodec.Decoder()
        images: list[nvimgcodec.Image] = [
            decoder.read(str(i_path)) for i_path in input_paths
        ]
        tensors: list[cvcuda.Tensor] = [
            cvcuda.as_tensor(image, "HWC") for image in images
        ]
        # docs_tag: end_load_images

    with timer("Resize images"):
        # docs_tag: begin_resize
        # 2. Resize the images
        resized_tensors: list[cvcuda.Tensor] = [
            cvcuda.resize(
                tensor,
                (args.height, args.width, 3),
                interp=cvcuda.Interp.LINEAR,
            )
            for tensor in tensors
        ]
        # docs_tag: end_resize

    with timer("Batch images"):
        # docs_tag: begin_batch
        # 3. Batch all the images into a single CV-CUDA tensor
        batch_tensor: cvcuda.Tensor = cvcuda.stack(resized_tensors)
        # docs_tag: end_batch

    with timer("Apply Gaussian blur"):
        # docs_tag: begin_gaussian_blur
        # 4. Apply a Gaussian blur on the batch
        blurred_tensor_batch: cvcuda.Tensor = cvcuda.gaussian(
            batch_tensor,
            (args.kernel, args.kernel),
            (args.sigma, args.sigma),
            cvcuda.Border.CONSTANT,
        )
        # docs_tag: end_gaussian_blur

    with timer("Split images"):
        # docs_tag: begin_split_and_save
        # 5. Save the images back to disk
        # 5.1 Split the batch into individual images
        tensors: list[cvcuda.Tensor] = zero_copy_split(blurred_tensor_batch)

    with timer("Write images to disk"):
        # 5.2 Encode the images back to disk
        encoder = nvimgcodec.Encoder()
        for tensor, output_path in zip(tensors, output_paths):
            nvc_img = nvimgcodec.as_image(tensor.cuda())
            encoder.write(str(output_path), nvc_img)

    # 6. Verify output files exist
    for output_path in output_paths:
        assert output_path.exists()
        print(f"Wrote image to {output_path}")
    # docs_tag: end_split_and_save
    # docs_tag: end_main


if __name__ == "__main__":
    main()
