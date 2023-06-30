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
# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda
import os
import sys
import logging
import argparse
import cvcuda
import torch
import nvtx

# Bring the commons folder from the samples directory into our path so that
# we can import modules from it.
common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)

from torch_utils import ImageBatchDecoderPyTorch, ImageBatchEncoderPyTorch  # noqa: E402

from vpf_utils import (  # noqa: E402
    VideoBatchDecoderVPF,
    VideoBatchEncoderVPF,
)

from pipelines import (  # noqa: E402
    PreprocessorCvcuda,
    PostprocessorCvcuda,
)

from model_inference import (  # noqa: E402
    ObjectDetectionTensorRT,
)

# docs_tag: end_python_imports


def run_sample(
    input_path,
    output_dir,
    engine_path,
    batch_size,
    target_img_height,
    target_img_width,
    device_id,
    conf_threshold,
    iou_threshold,
):
    logger = logging.getLogger(__name__)

    logger.debug("Using batch size of %d" % batch_size)
    logger.debug("Using CUDA device: %d" % device_id)

    # docs_tag: begin_setup_sample
    logger.debug(
        "Using NVTX markers to log the performance. Use NVIDIA NSIGHT system to benchmark."
    )

    nvtx.push_range("run_sample")

    # Define the objects that handle the pipeline stages ---
    network_size = (target_img_width, target_img_height)

    # Define the cuda device, context and streams.
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    cvcuda_stream = cvcuda.Stream()
    torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    # docs_tag: end_setup_sample

    # docs_tag: begin_setup_data_loader
    if os.path.splitext(input_path)[1] == ".jpg" or os.path.isdir(input_path):
        # Treat this as data modality of images
        decoder = ImageBatchDecoderPyTorch(input_path, batch_size, device_id, cuda_ctx)

        encoder = ImageBatchEncoderPyTorch(
            output_dir, fps=0, device_id=device_id, cuda_ctx=cuda_ctx
        )
    else:
        # Treat this as data modality of videos
        decoder = VideoBatchDecoderVPF(input_path, batch_size, device_id, cuda_ctx)

        encoder = VideoBatchEncoderVPF(output_dir, decoder.fps, device_id, cuda_ctx)

    # Fire up encoder/decoder
    decoder.start()
    encoder.start()

    # docs_tag: end_setup_data_loader

    # docs_tag: begin_setup_obj_detection
    # Now define the object that will handle pre-processing
    preprocess = PreprocessorCvcuda(device_id)

    # Define the post-processor
    postprocess = PostprocessorCvcuda(
        conf_threshold,
        iou_threshold,
        device_id,
        encoder.input_layout,
        encoder.gpu_input,
        batch_size,
    )

    # Define the inferencer
    inference = ObjectDetectionTensorRT(engine_path, batch_size, device_id)

    # docs_tag: end_setup_obj_detection

    # docs_tag: begin_pipeline
    # Define and execute the processing pipeline ------------
    nvtx.push_range("pipeline")

    # Loop through all input frames
    batch_idx = 0
    while True:
        nvtx.push_range("batch_%d" % batch_idx)
        # Execute everything inside the streams.
        with cvcuda_stream, torch.cuda.stream(torch_stream):
            # Stage 1: decode
            batch = decoder()
            if batch is None:
                nvtx.pop_range()  # for batch
                break  # No more frames to decode
            logger.info("Processing batch %d" % batch_idx)
            assert batch_idx == batch.batch_idx

            # docs_tag: start_preproc
            # Stage 2: pre-processing
            orig_tensor, resized_tensor, normalized_tensor = preprocess(
                batch.data, network_size
            )

            # docs_tag: start_run_infer
            # Stage 3: inference using tensorrt
            bboxes, probabilities = inference(normalized_tensor)

            # docs_tag: start_postprocess
            # Stage 4: interpolate bounding boxes to image resolution and filter bounding boxes using NMS
            out_tensor = postprocess(bboxes, probabilities, orig_tensor)

            # docs_tag: start_encode
            # encode the batch
            batch.data = out_tensor
            encoder(batch)
            batch_idx += 1

        nvtx.pop_range()  # for batch

    # Make sure encoder finishes any outstanding work
    encoder.join()

    nvtx.pop_range()  # for pipeline

    cuda_ctx.pop()
    # docs_tag: end_pipeline

    nvtx.pop_range()  # for this sample.


# docs_tag: begin_main_func
def main():
    parser = argparse.ArgumentParser(
        "Object Detection sample using CV-CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="./assets/images/peoplenet.jpg",
        type=str,
        help="Either a path to a JPEG image/MP4 video or a directory containing JPG images "
        "to use as input. When pointing to a directory, only JPG images will be read.",
    )

    parser.add_argument(
        "-th",
        "--target_img_height",
        default=544,
        type=int,
        help="The height to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-tw",
        "--target_img_width",
        default=960,
        type=int,
        help="The width to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="The batch size.",
    )

    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="The GPU device to use for this sample.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["info", "error", "debug", "warning"],
        default="info",
        help="Sets the desired logging level. Affects the std-out printed by the "
        "sample when it is run.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/tmp",
        type=str,
        help="The folder where the output detections are stored.",
    )

    parser.add_argument(
        "-e",
        "--engine_path",
        default="model.engine",
        type=str,
        help="The serialized engine path.",
    )

    parser.add_argument(
        "-c",
        "--conf_threshold",
        default=0.9,
        type=float,
        help="The confidence threshold for the detections.",
    )

    parser.add_argument(
        "-iou",
        "--iou_threshold",
        default=0.1,
        type=float,
        help="The iou threshold for NMS.",
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    logging.basicConfig(
        format="[%(name)s:%(lineno)d] %(asctime)s %(levelname)-6s %(message)s",
        level=getattr(logging, args.log_level.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.batch_size <= 0:
        raise ValueError("batch_size must be a value >=1.")

    if args.target_img_height < 10:
        raise ValueError("target_img_height must be a value >=10.")

    if args.target_img_width < 10:
        raise ValueError("target_img_width must be a value >=10.")

    if not os.path.isdir(args.output_dir):
        raise ValueError("Output directory not found: %s" % args.output_dir)

    if not os.path.isfile(args.engine_path):
        raise ValueError("TensorRT engine path not found: %s" % args.engine_path)
    # Run the sample.
    # docs_tag: start_call_run_sample
    run_sample(
        args.input_path,
        args.output_dir,
        args.engine_path,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
        args.conf_threshold,
        args.iou_threshold,
    )
    # docs_tag: end_call_run_sample


# docs_tag: end_main_func

if __name__ == "__main__":
    main()
