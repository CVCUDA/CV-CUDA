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
import queue
import uuid
import logging
import time
from functools import partial
import cvcuda
import torch
import tritonclient.grpc as tritongrpcclient
from tritonclient.utils import InferenceServerException
from pathlib import Path

# Bring module folders from the samples directory into our path so that
# we can import modules from it.
samples_dir = Path(os.path.abspath(__file__)).parents[2]  # samples/
sys.path.insert(0, os.path.join(samples_dir, ""))

from common.python.perf_utils import (  # noqa: E402
    CvCudaPerf,
    get_default_arg_parser,
    parse_validate_default_args,
)

from common.python.torch_utils import (  # noqa: E402
    ImageBatchDecoderPyTorch,
    ImageBatchEncoderPyTorch,
)

from common.python.vpf_utils import (  # noqa: E402
    VideoBatchDecoderVPF,
    VideoBatchEncoderVPF,
    VideoBatchStreamingDecoderVPF,
    VideoBatchStreamingEncoderVPF,
)

# docs_tag: end_python_imports


def callback(response, result, error):
    """
    Triton Inference callback
    """
    if error:
        response.append(error)
    else:
        response.append(result)


def run_sample(
    input_path,
    output_dir,
    batch_size,
    url,
    device_id,
    should_stream_video,
    cvcuda_perf,
):
    """
    Runs the Sample for a given input image or video.
    """
    logger = logging.getLogger(__name__)

    logger.debug("Using batch size of %d" % batch_size)
    logger.debug("Using CUDA device: %d" % device_id)

    # Check if video streaming was requested and that it is possible
    if should_stream_video:
        if os.path.splitext(input_path)[1] == ".jpg" or os.path.isdir(input_path):
            logger.warning("Video streaming mode is not available for image data.")
            should_stream_video = False  # Not possible in images use case.

    if should_stream_video:
        model_name = "fcn_resnet101_streaming"
        model_version = "1"
        logger.info("Using streaming video.")
    else:
        model_name = "fcn_resnet101"
        model_version = "1"

    # docs_tag: begin_stream_setup
    cvcuda_perf.push_range("run_sample")

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
        raise Exception("Unable to create Triton GRPC Client: " + str(e))

    # Set input and output Triton buffer names
    input_name = "inputrgb"
    output_name = "outputrgb"
    # docs_tag: end_setup_triton_client

    # docs_tag: begin_init_dataloader
    if os.path.splitext(input_path)[1] == ".jpg" or os.path.isdir(input_path):
        # Treat this as data modality of images
        decoder = ImageBatchDecoderPyTorch(
            input_path,
            batch_size,
            device_id,
            cuda_ctx,
            cvcuda_perf,
        )

        encoder = ImageBatchEncoderPyTorch(
            output_dir,
            fps=0,
            device_id=device_id,
            cuda_ctx=cuda_ctx,
            cvcuda_perf=cvcuda_perf,
        )
    else:
        # Treat this as data modality of videos.
        # Check if the user wanted to use streaming video or not.
        if should_stream_video:
            decoder = VideoBatchStreamingDecoderVPF(
                "client",
                None,
                cvcuda_perf,
                input_path,
                model_name,
                model_version,
            )
            file_name = os.path.splitext(os.path.basename(input_path))[0]
            video_id = uuid.uuid4()  # unique video id
            video_output_path = os.path.join(
                output_dir, f"{file_name}_output_{video_id}.mp4"
            )
            encoder = VideoBatchStreamingEncoderVPF(
                "client",
                None,
                cvcuda_perf,
                video_output_path,
                decoder.decoder.fps,
            )
        else:
            decoder = VideoBatchDecoderVPF(
                input_path,
                batch_size,
                device_id,
                cuda_ctx,
                cvcuda_perf,
            )

            encoder = VideoBatchEncoderVPF(
                output_dir, decoder.fps, device_id, cuda_ctx, cvcuda_perf
            )

    # Fire up encoder/decoder
    decoder.start()
    encoder.start()
    # docs_tag: end_init_dataloader

    # Define and execute the processing pipeline
    cvcuda_perf.push_range("pipeline")

    # Loop through all input frames
    batch_idx = 0

    # It is advisable to use client object within with..as clause
    # when sending streaming requests. This ensures the client
    # is closed when the block inside with exits.
    with tritongrpcclient.InferenceServerClient(url=url) as triton_client:
        if should_stream_video:
            # This is a streaming video use case. Most of the activities will
            # take place on the server.
            # The overall flow would be:
            # 1. Start sending decoder requests
            # 2. Wait for processed results
            # 3. Encode

            user_data = UserData()

            try:
                # Establish stream
                triton_client.start_stream(
                    callback=partial(callback_streaming, user_data)
                )

                # docs_tag: begin_streamed_decode
                # Stage 1: Begin streaming input data to the server for decoding
                decoder(triton_client, tritongrpcclient, video_id, batch_size)
                # docs_tag: end_streamed_decode

                # docs_tag: begin_async_receive
                # Stage 2: Begin receiving the output from server
                packet_count = 0
                packet_size = 0
                while True:
                    response = user_data._completed_requests.get()

                    # If the response was an error, we must raise it.
                    if isinstance(response, Exception):
                        raise response
                    else:
                        # The response was a data item.
                        data_item = response

                    packet = data_item.as_numpy("PACKET_OUT")
                    height, width = data_item.as_numpy("FRAME_SIZE")

                    packet_count += 1
                    packet_size += len(packet)
                    if packet_count % 50 == 0:
                        logger.debug(
                            "Received packet No. %d, size %d"
                            % (packet_count, len(packet))
                        )

                    # Check if this was the last packet by checking the terminal flag
                    if data_item.as_numpy("LAST_PACKET")[0]:
                        break
                    # docs_tag: end_async_receive

                    # docs_tag: begin_streamed_encode
                    # Stage 3: Begin streaming output data from server for encoding.
                    encoder(packet, height, width)
                    # docs_tag: end_streamed_encode

                # Close encoder/decoder
                decoder.join()
                encoder.join()

                logger.debug(
                    "%d packets received from server (including the EOF packet), total bytes = %d(MB)"
                    % (packet_count, packet_size / (1024 * 1024))
                )
                logger.info("Output video saved to %s" % video_output_path)
            except InferenceServerException as error:
                cuda_ctx.pop()
                raise error

            # docs_tag: end_pipeline

        else:
            while True:
                cvcuda_perf.push_range("batch", batch_idx=batch_idx)

                with cvcuda_stream, torch.cuda.stream(torch_stream):
                    # docs_tag: begin_data_decode
                    # Stage 1: decode
                    batch = decoder()
                    if batch is None:
                        cvcuda_perf.pop_range(total_items=0)  # for batch
                        break  # No more frames to decode
                    assert batch_idx == batch.batch_idx
                    # docs_tag: end_data_decode

                    logger.info("Processing batch %d" % batch_idx)

                    # docs_tag: begin_create_triton_input
                    # Stage 2: Create Triton Input request
                    inputs = []
                    outputs = []

                    cvcuda_perf.push_range("io_prep")
                    torch_arr = torch.as_tensor(
                        batch.data.cuda(), device="cuda:%d" % device_id
                    )
                    numpy_arr = torch_arr.cpu().numpy()
                    inputs.append(
                        tritongrpcclient.InferInput(
                            input_name, numpy_arr.shape, "UINT8"
                        )
                    )
                    outputs.append(tritongrpcclient.InferRequestedOutput(output_name))
                    inputs[0].set_data_from_numpy(numpy_arr)
                    cvcuda_perf.pop_range()
                    # docs_tag: end_create_triton_input

                    # docs_tag: begin_async_infer
                    # Stage 3 : Run async Inference
                    cvcuda_perf.push_range("async_infer")
                    response = []
                    triton_client.async_infer(
                        model_name=model_name,
                        inputs=inputs,
                        callback=partial(callback, response),
                        model_version=model_version,
                        outputs=outputs,
                    )
                    # docs_tag: end_async_infer

                    # docs_tag: begin_sync_output
                    # Stage 4 : Wait until the results are available
                    while len(response) == 0:
                        time.sleep(0.001)
                    cvcuda_perf.pop_range()

                    # Stage 5 : Parse received Infer response
                    cvcuda_perf.push_range("parse_response")
                    if len(response) == 1:
                        # Check for the errors
                        if type(response[0]) == InferenceServerException:
                            cuda_ctx.pop()
                            raise response[0]
                        else:
                            seg_output = response[0].as_numpy(output_name)

                    cvcuda_perf.pop_range()
                    # docs_tag: end_sync_output

                    # docs_tag: begin_encode_output
                    # Stage 6: encode output data
                    cvcuda_perf.push_range("encode_output")
                    seg_output = torch.as_tensor(seg_output)
                    batch.data = seg_output.cuda()
                    encoder(batch)
                    cvcuda_perf.pop_range()

                    # docs_tag: end_encode_output

                    batch_idx += 1

                cvcuda_perf.pop_range(total_items=batch.data.shape[0])  # for batch

            # Make sure encoder finishes any outstanding work
            encoder.join()
            # docs_tag: end_pipeline

        cvcuda_perf.pop_range()  # for pipeline

        cuda_ctx.pop()

        cvcuda_perf.pop_range()  # for this sample.

        # Once everything is over, we need to finalize the perf-numbers.
        cvcuda_perf.finalize()


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback_streaming(response, result, error):
    """
    Triton Inference callback for streaming, cache response in a queue
    """
    if error:
        response._completed_requests.put(error)
        # raise error
    else:
        response._completed_requests.put(result)


# docs_tag: begin_main_func
def main():
    parser = get_default_arg_parser(
        "Semantic segmentation sample using CV-CUDA with Triton."
    )
    parser.add_argument(
        "-c",
        "--class_name",
        default="__background__",
        type=str,
        help="The class to visualize the results for.",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-sv",
        "--stream_video",
        action="store_true",
        help="Enable Triton streaming (i.e. server-side decoding and encoding) of video data.",
    )
    args = parse_validate_default_args(parser)

    # Parse the command line arguments.
    args = parser.parse_args()

    logging.basicConfig(
        format="[%(name)s:%(lineno)d] %(asctime)s %(levelname)-6s %(message)s",
        level=getattr(logging, args.log_level.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run the sample.
    # docs_tag: start_call_run_sample
    cvcuda_perf = CvCudaPerf("segmentation_triton_sample", default_args=args)
    run_sample(
        args.input_path,
        args.output_dir,
        args.batch_size,
        args.url,
        args.device_id,
        args.stream_video,
        cvcuda_perf,
    )
    # docs_tag: end_call_run_sample


# docs_tag: end_main_func

if __name__ == "__main__":
    main()
