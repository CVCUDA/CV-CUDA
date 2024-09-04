# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401
import os
import sys
import logging
import cvcuda
import nvcv
import torch

from pathlib import Path

# Bring module folders from the samples directory into our path so that
# we can import modules from it.
current_dir = Path(os.path.abspath(__file__)).parents[0]
samples_dir = os.path.join(Path(os.path.abspath(__file__)).parents[2], "samples")
common_dir = os.path.join(
    samples_dir,
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
    ImageBatchDecoder,
)

from bench_utils import get_benchmark_eligible_ops_info  # noqa: E402


def run_bench(
    input_path,
    output_dir,
    batch_size,
    target_img_height,
    target_img_width,
    device_id,
    num_iters,
    should_visualize,
    ops_filter_list,
    cvcuda_perf,
):
    """
    Runs the per operator benchmarks. It automatically discovers eligible operators for benchmarking,
    sets them up, runs them and saves the runtime numbers. benchmark.py is needed to actually perform any
    timing measurements.
    """
    logger = logging.getLogger("run_bench")
    logger.info("Benchmarking started.")

    # Set up various CUDA stuff.
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    # Use the the default stream for cvcuda and torch
    # Since we never created a stream current will be the CUDA default stream
    cvcuda_stream = cvcuda.Stream().current
    torch_stream = torch.cuda.default_stream(device=cuda_device)

    # Create an image batch decoder to supply us the input test data.
    decoder = ImageBatchDecoder(
        input_path,
        batch_size,
        device_id,
        cuda_ctx,
        cvcuda_stream,
        cvcuda_perf=cvcuda_perf,
    )

    # Get a list of (class names, class types) of all the ops that can be profiled.
    ops_info_list = get_benchmark_eligible_ops_info()
    logger.info("Found a total of %d operators for benchmarking." % len(ops_info_list))

    if ops_filter_list:
        # Filter based on user's criteria.
        ops_info_list_filtered = []
        for op_class_name, op_class in ops_info_list:
            for op_filter_name in ops_filter_list:
                if op_class_name.startswith(op_filter_name):
                    ops_info_list_filtered.append((op_class_name, op_class))
                    break

        ops_info_list = ops_info_list_filtered
        logger.info(
            "Filtered to a total of %d operators for benchmarking." % len(ops_info_list)
        )

    if should_visualize:
        logger.warning(
            "Visualization is turned ON. Run-times may increase drastically due to disk I/O."
        )

    #  Do everything in streams.
    with cvcuda_stream, torch.cuda.stream(torch_stream):

        # Start the decoder and get a batch.
        # NOTE: Currently, we will grab the first and only batch out of the decoder for
        #       performance benchmarking. All ops will receive this and only this batch.
        decoder.start()
        batch = decoder()
        batch.data = cvcuda.as_tensor(batch.data.cuda(), "NHWC")
        # Read input and create a batch

        for op_class_name, op_class in ops_info_list:
            logger.info("Running %s..." % op_class_name)
            cvcuda_perf.push_range(op_class_name)

            # Step 1: Initialize the operator...
            cvcuda_perf.push_range("init_op")
            try:
                op_instance = op_class(
                    device_id=device_id,
                    input=batch.data,
                    output_dir=output_dir,
                    should_visualize=should_visualize,
                )
                torch.cuda.current_stream().synchronize()
                cvcuda_perf.pop_range()  # For init_op
            except Exception as e:
                logger.error(
                    "Unable to init the op %s due to error: %s"
                    % (op_class_name, str(e))
                )
                cvcuda_perf.pop_range(delete_range=True)  # Deletes the init_op range
                cvcuda_perf.pop_range(
                    delete_range=True
                )  # Deletes the op_name range, too.
                continue  # Continue to the next operator.

            # Step 2: Run the operator.
            # Repeat for as many iterations as we wanted.
            cvcuda_perf.push_range("run_op")
            for i in range(num_iters):
                # Start the iteration.
                cvcuda_perf.push_range("iter", batch_idx=i)

                # Run the op
                success = op_instance(batch.data)
                torch.cuda.current_stream().synchronize()
                # Finish
                cvcuda_perf.pop_range(total_items=batch_size, delete_range=not success)

                # Get out of the loop if our operator invocation fails.
                if not success:
                    break

            cvcuda_perf.pop_range(delete_range=not success)  # For the run_op
            # reset the cache limit to not affect other operator benchmarks, in case a benchmark test
            # changed it
            if hasattr(nvcv, "set_cache_limit_inbytes"):
                total = torch.cuda.mem_get_info()[1]
                nvcv.set_cache_limit_inbytes(total // 2)

            # Step 3: log the parameters used by the operator, initialized during the setup call.
            if success:
                cvcuda_perf.push_range("op_params")
                cvcuda_perf.push_range(str(op_instance.get_params_info()))
                cvcuda_perf.pop_range()
                cvcuda_perf.pop_range()

                cvcuda_perf.pop_range()  # For the op_name
            else:
                cvcuda_perf.pop_range(
                    delete_range=True
                )  # Deletes the op_name range, too, if run_op failed

        cuda_ctx.pop()
        cvcuda_perf.finalize()
        logger.info("Finished run_bench.")


def main():
    # docs_tag: begin_parse_args
    parser = get_default_arg_parser(
        "Profiler for all ops of CV-CUDA.",
        input_path=os.path.join(current_dir, "assets", "brooklyn.jpg"),
        supports_video=False,
        batch_size=32,
    )
    parser.add_argument(
        "-n",
        "--num_iters",
        default=10,
        type=int,
        help="The number of iterations to run the benchmarks for.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Flag specifying whether outputs from the operators should be visualized"
        " on written on disk or not.",
    )
    parser.add_argument(
        "ops",
        nargs="*",
        help="Optional list of one or more operator names which you want to benchmark. "
        "When supplied, the benchmarking will be restricted to only the operators that starts "
        "with these names.",
    )
    args = parse_validate_default_args(parser)

    logging.basicConfig(
        format="[%(name)s:%(lineno)d] %(asctime)s %(levelname)-6s %(message)s",
        level=getattr(logging, args.log_level.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cvcuda_perf = CvCudaPerf("run_bench", default_args=args)
    run_bench(
        args.input_path,
        args.output_dir,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
        args.num_iters,
        args.visualize,
        args.ops,
        cvcuda_perf,
    )


if __name__ == "__main__":
    main()
