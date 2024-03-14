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
import cvcuda
import torch

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

from nvcodec_utils import (  # noqa: E402
    VideoBatchDecoder,
    ImageBatchDecoder,
)

from pipelines import (  # noqa: E402
    PreprocessorCvcuda,
    PostprocessorCvcuda,
)

from model_inference import (  # noqa: E402
    ClassificationPyTorch,
    ClassificationTensorRT,
)

# docs_tag: end_python_imports


def run_sample(
    input_path,
    output_dir,
    batch_size,
    target_img_height,
    target_img_width,
    device_id,
    backend,
    cvcuda_perf,
):
    logger = logging.getLogger("classification")

    logger.debug("Using batch size of %d" % batch_size)
    logger.debug("Using CUDA device: %d" % device_id)

    # docs_tag: begin_setup_gpu
    cvcuda_perf.push_range("run_sample")

    # Define the objects that handle the pipeline stages
    image_size = (target_img_width, target_img_height)

    # Define the cuda device, context and streams.
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    cvcuda_stream = cvcuda.Stream()
    torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    # docs_tag: end_setup_gpu

    # docs_tag: begin_setup_stages
    # Now define the object that will handle pre-processing
    preprocess = PreprocessorCvcuda(device_id, cvcuda_perf)

    if os.path.splitext(input_path)[1] == ".jpg" or os.path.isdir(input_path):
        # Treat this as data modality of images
        decoder = ImageBatchDecoder(
            input_path,
            batch_size,
            device_id,
            cuda_ctx,
            cvcuda_perf,
        )

    else:
        # Treat this as data modality of videos
        decoder = VideoBatchDecoder(
            input_path,
            batch_size,
            device_id,
            cuda_ctx,
            cvcuda_perf,
        )

    # Define the post-processor
    postprocess = PostprocessorCvcuda(
        "NCHW",
        device_id,
        cvcuda_perf,
    )

    # Setup the classification models.
    if backend == "pytorch":
        inference = ClassificationPyTorch(
            output_dir,
            batch_size,
            image_size,
            device_id,
            cvcuda_perf,
        )
    elif backend == "tensorrt":
        inference = ClassificationTensorRT(
            output_dir,
            batch_size,
            image_size,
            device_id,
            cvcuda_perf,
        )
    else:
        raise ValueError("Unknown backend: %s" % backend)
    # docs_tag: end_setup_stages

    # docs_tag: begin_pipeline
    # Define and execute the processing pipeline ------------
    cvcuda_perf.push_range("pipeline")

    # Fire up the decoder
    decoder.start()

    # Loop through all input frames
    batch_idx = 0
    while True:
        cvcuda_perf.push_range("batch", batch_idx=batch_idx)

        with cvcuda_stream, torch.cuda.stream(torch_stream):
            # Stage 1: decode
            batch = decoder()
            if batch is None:
                cvcuda_perf.pop_range(total_items=0)  # for batch
                break  # No more frames to decode
            assert batch_idx == batch.batch_idx

            logger.info("Processing batch %d" % batch_idx)

            # Stage 2: pre-processing
            orig_tensor, resized_tensor, normalized_tensor = preprocess(
                batch.data,
                out_size=image_size,
            )

            # Stage 3: inference
            probabilities = inference(normalized_tensor)

            # Stage 4: post-processing
            postprocess(
                probabilities,
                top_n=5,
                labels=inference.labels,
            )

            batch_idx += 1

        cvcuda_perf.pop_range(total_items=batch.data.shape[0])  # for batch

    cvcuda_perf.pop_range()  # for pipeline

    cuda_ctx.pop()
    # docs_tag: end_pipeline

    cvcuda_perf.pop_range()  # for this sample.

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
        "Classification sample using CV-CUDA.",
        input_path=os.path.join(assets_dir, "images", "tabby_tiger_cat.jpg"),
        target_img_height=224,
        target_img_width=224,
        supported_backends=["pytorch", "tensorrt"],
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
    cvcuda_perf = CvCudaPerf("classification_sample", default_args=args)
    run_sample(
        args.input_path,
        args.output_dir,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
        args.backend,
        cvcuda_perf,
    )
    # docs_tag: end_call_run_sample


# docs_tag: end_main_func


if __name__ == "__main__":
    main()
