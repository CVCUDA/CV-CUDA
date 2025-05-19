#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# docs_tag: begin_python_imports

import argparse
import cvcuda
import cupy as cp
from nvidia import nvimgcodec
from matplotlib import pyplot as plt

# docs_tag: end_python_imports

app_title = "CV-CUDA Hello World Example"


def main(inputs: list[str], outputs: list[str]):
    assert len(inputs) == len(outputs)

    # docs_tag: begin_main

    print(app_title)

    # docs_tag: begin_load_image

    # Create the nvimgcodec decoder to load images.
    decoder = nvimgcodec.Decoder()

    print("Loading images...")

    cv_tensors: cvcuda.Tensor = None
    img_shape = None
    for input_filename in inputs:

        # Open the input file and decode it into an image.
        # nvimgcodec supports jpeg, jpeg2000, tiff, bmp, png, pnm, webp image file formats.
        print(f"Loading image from {input_filename}")
        with open(input_filename, "rb") as in_file:
            data = in_file.read()
        # Decode the loaded image and store in the default CUDA device.
        # nvimgcodec decodes images into RGB uint8 HWC format.
        nv_gpu_img: nvimgcodec.Image = decoder.decode(data).cuda()

        # Wrap an existing CUDA buffer in a CVCUDA tensor.
        # CVCUDA supports (N)HWC image layout only.
        cv_tensor = cvcuda.as_tensor(nv_gpu_img, "HWC")

        # Add loaded image to batch:

        # Check that image sizes are the same.
        if img_shape:
            if img_shape != cv_tensor.shape:
                raise RuntimeError(
                    f"All images in input must be of the same size: {img_shape} != {cv_tensor.shape}"
                )
        else:
            img_shape = cv_tensor.shape
        # Pack the loaded tensor into a batch (NHWC).
        cv_tensor = cv_tensor.reshape((1, *cv_tensor.shape), "NHWC")
        cv_tensors = (
            cvcuda.stack([cv_tensors, cv_tensor])
            if cv_tensors
            else cvcuda.stack([cv_tensor])
        )
    # docs_tag: end_load_image

    # docs_tag: begin_process_image

    # The resulting cv_tensors has the NHWC layout with N = len(inputs).
    assert cv_tensors.shape[0] == len(inputs)
    print(cv_tensors.shape)

    # Manipulate the tensor data in CVCUDA.

    # Resize the tensors.
    cv_tensors_result = cvcuda.resize(
        cv_tensors,
        (cv_tensors.shape[0], 224, 224, cv_tensors.shape[-1]),  # N, H, W, C
        interp=cvcuda.Interp.LINEAR,
    )

    # Apply a gaussian blur.
    kernel_size = (3, 3)
    gaussian_sigma = (1, 1)
    cv_tensors_result = cvcuda.gaussian(
        cv_tensors_result, kernel_size, gaussian_sigma, cvcuda.Border.CONSTANT
    )
    # docs_tag: end_process_image

    # docs_tag: begin_store_image

    print("Storing images...")

    # Create the nvimgcodec encoder to store images.
    encoder = nvimgcodec.Encoder()

    # Use cupy to separate the tensor batch.
    # cvcuda.Tensor.cuda() returns the buffer with __cuda_array_interface__.
    cp_array_result = cp.asarray(cv_tensors_result.cuda())
    # Write each image to storage.
    encoder.write(outputs, [cp_arr for cp_arr in cp_array_result])
    # docs_tag: end_store_image

    # docs_tag: begin_display_image

    # Use pyplot to display the first result.
    print("Displaying the first result...")
    plt.imshow(cp_array_result[0].get())
    plt.show()
    # docs_tag: end_display_image

    print("Completed")

    # docs_tag: end_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=app_title)
    parser.add_argument(
        "-i",
        "--inputs",
        required=True,
        nargs="+",
        help="List of paths to input image. All images must be of the same size. "
        "Files must be in one of these formats: jpeg, jpeg2000, tiff, bmp, png, pnm, or webp.",
    )
    parser.add_argument(
        "-o",
        "--outputs",
        required=True,
        nargs="+",
        help="List of paths to save output image. "
        "Must match the number of input files.",
    )
    args = parser.parse_args()

    if len(args.inputs) != len(args.outputs):
        parser.error("Number of outputs must match number of inputs.")

    main(args.inputs, args.outputs)
