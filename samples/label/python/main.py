# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pycuda.driver as cuda
import os
import sys
import logging
import cvcuda
import torch
import numpy as np

# Bring the commons folder from the samples directory into our path so that
# we can import modules from it.
common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)

from perf_utils import (  # noqa: E402
    CvCudaPerf,
    get_default_arg_parser,
    parse_validate_default_args,
)

from torch_utils import ImageBatchDecoderPyTorch, ImageBatchEncoderPyTorch  # noqa: E402
from interop_utils import to_cpu_numpy_buffer, to_cuda_buffer  # noqa: E402

# docs_tag: end_python_imports


def save_batch(images, label, encoder, batch):
    """Save a batch of images to disk

    Args:
        images (nvcv Tensor): Batch of images to save

        label : label value for output file name,
                  used to differentiate between outputs
                  appended to the original file name.

        encoder : Encoder object to save the images

        batch : Batch object to save the images

    Returns:
        nvcv Tensor: RGB color, random for each label
    """
    # Function to modify filenames in the batch
    def modify_filenames(suffix):
        modified_filenames = []
        for filename in batch.fileinfo:
            name, extension = filename.rsplit(".", 1)
            modified_filename = f"{name}_{suffix}.{extension}"
            modified_filenames.append(modified_filename)
        return modified_filenames

    # convert to NCHW
    imagesNCHW = cvcuda.reformat(images, "NCHW")

    # Modify filenames with "_labels" suffix
    oldFileNames = batch.fileinfo
    batch.fileinfo = modify_filenames(label)
    batch.data = torch.as_tensor(imagesNCHW.cuda())
    encoder(batch)
    batch.fileinfo = oldFileNames


def simple_cmap(label):
    """Convert a label map to an random RGB color

    Args:
        labels : label value

    Returns:
        nvcv Tensor: RGB color, random for each label
    """
    np.random.seed(label)  # Ensure consistent color for each label
    return np.random.randint(0, 256, 3)  # Random RGB color


def color_labels_nhwc(labels):
    """Convert a label map to an RGB image

    Args:
        labels : Output of cvcuda.label operator

    Returns:
        nvcv Tensor: RGB image, with each label having a unique color
    """
    npLabels = to_cpu_numpy_buffer(labels.cuda())
    # Initialize the output array with the same batch size, height, width, and RGB channels
    a_rgb = np.zeros(
        [npLabels.shape[0], npLabels.shape[1], npLabels.shape[2], 3], dtype=np.uint8
    )

    # Iterate over each image in the batch
    for n in range(npLabels.shape[0]):
        # Extract unique labels for the current image
        a_labels = np.unique(npLabels[n, :, :, :])

        # Process each label in the current image
        for label in a_labels:
            rgb_label_color = simple_cmap(label)
            # Create a mask for the current label
            mask = npLabels[n] == label
            # Use the mask to assign color to the corresponding pixels
            a_rgb[n][mask[:, :, 0]] = rgb_label_color.astype(np.uint8)

    return cvcuda.as_tensor(to_cuda_buffer(a_rgb), "NHWC")


def run_sample(
    input_path,
    output_dir,
    batch_size,
    target_img_height,
    target_img_width,
    device_id,
    cvcuda_perf,
):
    logger = logging.getLogger("Distance_Label_Sample")

    logger.debug("Using batch size of %d" % batch_size)
    logger.debug("Using CUDA device: %d" % device_id)

    # docs_tag: begin_setup_gpu
    cvcuda_perf.push_range("run_sample")

    # Define the objects that handle the pipeline stages ---
    image_size = (target_img_width, target_img_height)
    logger.debug("Image size: %d %d" % image_size)

    # Define the cuda device, context and streams.
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    cvcuda_stream = cvcuda.Stream()
    torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    # docs_tag: end_setup_gpu

    # docs_tag: encoder_decoder setup
    # Now define the object that will handle pre-processing
    if os.path.splitext(input_path)[1] == ".jpg" or os.path.isdir(input_path):
        # Treat this as data modality of images
        decoder = ImageBatchDecoderPyTorch(
            input_path, batch_size, device_id, cuda_ctx, cvcuda_perf
        )
        encoder = ImageBatchEncoderPyTorch(
            output_dir,
            fps=0,
            device_id=device_id,
            cuda_ctx=cuda_ctx,
            cvcuda_perf=cvcuda_perf,
        )
    else:
        raise ValueError("Unknown data modality: %s." % input_path)
    # docs_tag: encoder_decoder setup

    # docs_tag: begin_pipeline
    # Define and execute the processing pipeline
    cvcuda_perf.push_range("pipeline")

    # Fire up encoder/decoder
    decoder.start()
    encoder.start()

    # Loop through all input frames
    batch_idx = 0
    while True:
        cvcuda_perf.push_range("batch", batch_idx=batch_idx)

        # Execute everything inside the streams.
        with cvcuda_stream, torch.cuda.stream(torch_stream):
            # Stage 1: decode
            batch = decoder()
            if batch is None:
                cvcuda_perf.pop_range(total_items=0)  # for batch
                break  # No more frames to decode
            assert batch_idx == batch.batch_idx

            logger.info("Processing batch %d" % batch_idx)

            # docs_tag: process_batch
            # Stage 2: process

            # docs_tag: begin_tensor_conversion
            # Need to check what type of input we have received:
            # 1) CVCUDA tensor --> Nothing needs to be done.
            # 2) Numpy Array --> Convert to torch tensor first and then CVCUDA tensor
            # 3) Torch Tensor --> Convert to CVCUDA tensor
            if isinstance(batch.data, torch.Tensor):
                cvcudaTensorNHWC = cvcuda.as_tensor(batch.data, "NHWC")
            elif isinstance(batch.data, np.ndarray):
                cvcudaTensorNHWC = cvcuda.as_tensor(
                    torch.as_tensor(batch.data).to(
                        device="cuda:%d" % device_id, non_blocking=True
                    ),
                    "NHWC",
                )
            # docs_tag: end_tensor_conversion

            # Convert to grayscale
            out = cvcuda.cvtcolor(cvcudaTensorNHWC, cvcuda.ColorConversion.RGB2GRAY)

            save_batch(out, "grayscale", encoder, batch)

            # Histogram eq the image
            out = cvcuda.histogrameq(out, cvcuda.Type.U8)

            save_batch(out, "histogrameq", encoder, batch)

            # Threshold the image
            # Use torch tensor for this to take advantage of easy data manipulation
            thParam = torch.zeros(out.shape[0], dtype=torch.float64).cuda()
            maxParam = torch.zeros(out.shape[0], dtype=torch.float64).cuda()

            # The parameters below can be set per image. For now, we are setting them to a constant value.
            # Proper threshold values must be determined by the input images and requirement.
            thParam.fill_(
                128
            )  # Configure the threshold value for each image anything below this will be 0 in the output.
            maxParam.fill_(255)  # Value to set the areas meeting the threshold.

            thParam = cvcuda.as_tensor(thParam, "N")
            maxParam = cvcuda.as_tensor(maxParam, "N")
            out = cvcuda.threshold(out, thParam, maxParam, cvcuda.ThresholdType.BINARY)

            save_batch(out, "threshold", encoder, batch)

            # Create label map
            ccLabels, _, _ = cvcuda.label(out)

            # Create and ARGB image from the label map, this is for visualization purposes only.
            argbImage = color_labels_nhwc(ccLabels)

            save_batch(argbImage, "label", encoder, batch)

            batch_idx += 1
            # docs_tag: end_process

        cvcuda_perf.pop_range(total_items=batch.data.shape[0])  # for batch

    # Make sure encoder finishes any outstanding work
    encoder.join()

    cvcuda_perf.pop_range()  # for pipeline

    cuda_ctx.pop()
    # docs_tag: end_pipeline

    cvcuda_perf.pop_range()  # for this example.

    # Once everything is over, we need to finalize the perf-numbers.
    cvcuda_perf.finalize()


# docs_tag: begin_main_func
def main():
    # docs_tag: begin_parse_args
    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets",
    )
    parser = get_default_arg_parser(
        "Label sample using CV-CUDA. This sample will execute the label operator on a "
        "single or a batch of images (must be same size). Each step of the pipeline will "
        "produce an *_stage.jpg output showing the processing done at that stage.",
        input_path=os.path.join(assets_dir, "images", "peoplenet.jpg"),
        target_img_height=544,
        target_img_width=960,
    )

    args = parse_validate_default_args(parser)

    logging.basicConfig(
        format="[%(name)s:%(lineno)d] %(asctime)s %(levelname)-6s %(message)s",
        level=getattr(logging, args.log_level.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # docs_tag: end_parse_args

    # Run the sample.
    # docs_tag: start_call_run_sample
    cvcuda_perf = CvCudaPerf("Distance_Label_sample", default_args=args)
    run_sample(
        args.input_path,
        args.output_dir,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
        cvcuda_perf,
    )
    # docs_tag: end_call_run_sample


# docs_tag: end_main_func

if __name__ == "__main__":
    main()
