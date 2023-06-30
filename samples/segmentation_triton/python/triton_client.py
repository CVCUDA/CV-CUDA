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
import numpy as np
import time
import nvtx


# Import Triton modules
import tritonclient.grpc as tritongrpcclient
from tritonclient.utils import InferenceServerException
from functools import partial

# Bring the commons folder from the samples directory into our path so that
# we can import modules from it.
common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)

# Import data loader modules
from torch_utils import ImageBatchDecoderPyTorch, ImageBatchEncoderPyTorch  # noqa: E402

from vpf_utils import (  # noqa: E402
    VideoBatchDecoderVPF,
    VideoBatchEncoderVPF,
)

# docs_tag: end_python_imports


# Triton Inference callback
def callback(response, result, error):
    if error:
        response.append(error)
    else:
        response.append(result)


# Run Sample for a given input image or video
def run_sample(input_path, output_dir, batch_size, url, device_id):
    logger = logging.getLogger(__name__)

    logger.debug("Using batch size of %d" % batch_size)

    # docs_tag: begin_stream_setup
    nvtx.push_range("run_sample")

    # Define the cuda device, context and streams.
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    cvcuda_stream = cvcuda.Stream()
    torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    # docs_tag: end_stream_setup

    # docs_tag: begin_setup_triton_client
    # Create GRPC Triton Client
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print("Unable to create Triton GRPC Client: " + str(e))
        sys.exit(1)

    # Set input and output Triton buffer names
    input_name = "inputrgb"
    output_name = "outputrgb"
    # docs_tag: end_setup_triton_client

    # docs_tag: begin_init_dataloader
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
    # docs_tag: end_init_dataloader

    # Define and execute the processing pipeline ------------
    nvtx.push_range("pipeline")

    # Loop through all input frames

    batch_idx = 0
    while True:
        logger.info("Processing batch %d" % batch_idx)

        nvtx.push_range("batch_%d" % batch_idx)
        # Execute everything inside the streams.
        with cvcuda_stream, torch.cuda.stream(torch_stream):

            # docs_tag: begin_data_decode
            # Stage 1: decode
            batch = decoder()
            if batch is None:
                break  # No more frames to decode
            assert batch_idx == batch.batch_idx
            # docs_tag: end_data_decode

            # docs_tag: begin_create_triton_input
            # Stage 2: Create Triton Input request
            inputs = []
            outputs = []

            torch_arr = torch.as_tensor(batch.data.cuda(), device="cuda")
            numpy_arr = np.array(torch_arr.cpu())
            inputs.append(
                tritongrpcclient.InferInput(input_name, numpy_arr.shape, "UINT8")
            )
            outputs.append(tritongrpcclient.InferRequestedOutput(output_name))
            inputs[0].set_data_from_numpy(numpy_arr)
            # docs_tag: end_create_triton_input

            # docs_tag: begin_async_infer
            # Stage 3 : Run Async Inference
            response = []

            triton_client.async_infer(
                model_name="fcn_resnet101",
                inputs=inputs,
                callback=partial(callback, response),
                model_version="1",
                outputs=outputs,
            )
            # docs_tag: end_async_infer

            # docs_tag: begin_sync_output
            # Stage 4 : Wait until the results are available
            while len(response) == 0:
                time.sleep(0.001)

            # Stage 5 : Parse recieved Infer response
            if len(response) == 1:
                # Check for the errors
                if type(response[0]) == InferenceServerException:
                    cuda_ctx.pop()
                    sys.exit(1)
                else:
                    seg_output = response[0].as_numpy(output_name)

            # docs_tag: end_sync_output

            # docs_tag: begin_encode_output
            # Stage 6: encode output data
            seg_output = torch.as_tensor(seg_output)
            batch.data = seg_output.cuda()
            encoder(batch)
            # docs_tag: end_encode_output

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
        "Semantic segmentation sample using CV-CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="./assets/images/Weimaraner.jpg",
        type=str,
        help="Either a path to a JPEG image/MP4 video or a directory containing JPG images "
        "to use as input. When pointing to a directory, only JPG images will be read.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/tmp",
        type=str,
        help="The folder where the output segmentation overlay should be stored.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        default=4,
        type=int,
        help="The batch size.",
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
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )

    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="The GPU device to use for this sample.",
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    logging.basicConfig(
        format="[%(name)s:%(lineno)d] %(asctime)s %(levelname)-6s %(message)s",
        level=getattr(logging, args.log_level.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not os.path.isdir(args.output_dir):
        raise ValueError("Output directory not found: %s" % args.output_dir)

    if args.batch_size <= 0:
        raise ValueError("batch_size must be a value >=1.")

    # Run the sample.
    # docs_tag: start_call_run_sample
    run_sample(
        args.input_path, args.output_dir, args.batch_size, args.url, args.device_id
    )
    # docs_tag: end_call_run_sample


# docs_tag: end_main_func

if __name__ == "__main__":
    main()
