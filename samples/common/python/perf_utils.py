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

# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401

import os
import sys
import json
import logging
import argparse
import subprocess
from collections import deque
import cvcuda
import torch
import nvtx


class CvCudaPerf:
    """
    This class helps keep track of the CPU and GPU run-time performance of
    arbitrary python code spinets. This class do not have the functionality to
    actually perform any benchmarking since that is already done by NVIDIA NSYS.
    This must be used in conjunction with the script like benchmark.py to actually
    get the performance numbers.

    This class acts as an extension of the NVTX markers API. It allows us to capture
    application specific ranges with a few extra meta-data such as what range was
    inside a batch and what was the batch size, etc.
    """

    def __init__(
        self,
        obj_name,
        default_args,
    ):
        """
        Initializes a new instance of the `perf_utils` class.
        :param obj_name: The name of the object used for performance benchmarking.
        :param default_args: The usual set of command line arguments used to launch the CV-CUDA sample.
        """
        self.obj_name = obj_name
        self.command_line_args = default_args

        if hasattr(self.command_line_args, "device_id"):
            self.device_id = self.command_line_args.device_id
        else:
            raise ValueError("device_id must be provided in the default_args.")

        if hasattr(self.command_line_args, "output_dir"):
            self.output_dir = self.command_line_args.output_dir
        else:
            raise ValueError("output_dir must be provided in the default_args.")

        self.logger = logging.getLogger(__name__)
        # We will use a stack to record the push/pop range operations.
        self.stack = deque()
        self.stack_path = self.obj_name
        # We will maintain 3 different dictionaries to store the data.
        # 1. timing_info: to store CPU and GPU timings of NVTX ranges.
        # 2. batch_info: to store batch size and batch index of NVTX ranges.
        # 3. inside_batch_info: to store which NVTX ranges are part of a batch.
        self.timing_info = {}
        self.batch_info = {}
        self.inside_batch_info = []
        self.is_inside_batch = 0
        self.total_batches_processed = {}
        # Check if the benchmark.py script was used to run this. We do so
        # by checking whether an environment variable only set by that script is
        # present or not.
        if os.environ.get("BENCHMARK_PY"):
            self.should_benchmark = True
            self.logger.info("Benchmarking mode is turned on.")
        else:
            self.should_benchmark = False
            self.logger.warning(
                "perf_utils is used without benchmark.py. "
                "Benchmarking mode is turned off."
            )
        self.logger.info("Using CV-CUDA version: %s" % cvcuda.__version__)

    def push_range(
        self, message=None, color="blue", domain=None, category=None, batch_idx=None
    ):
        """
        Pushes a code range on to the stack for performance benchmarking.
        :param message: A message associated with the annotated code range.
        :param color: A color associated with the annotated code range.
        :param domain: Name of a domain under which the code range is scoped.
        :param category: A string or an integer specifying the category within the domain
        under which the code range is scoped. If unspecified, the code range
        is not associated with a category.
        :param batch_idx: If this range is associated with a batch, then its batch number.
         All the ranges pushed after this will be automatically associated with this batch.
        """
        if batch_idx is not None:
            message += "_%d" % batch_idx
            self.is_inside_batch += 1

        nvtx.push_range(message, color, domain, category)

        if self.should_benchmark:
            self.stack.append((message, batch_idx))
            self.stack_path = os.path.join(self.stack_path, message)

    def pop_range(self, domain=None, total_items=None):
        """
        Pops a code range off of the stack for performance benchmarking.
        :param domain: Name of a domain under which the code range is scoped.
        :param total_items: The number of items processed in this range.
        """
        if self.should_benchmark:
            # Grab the message and optional batch index from the stack.
            message, batch_idx = self.stack.pop()

            self.timing_info[self.stack_path] = (
                0,
                0,
            )  # Placeholders for CPU and GPU times respectively.
            # Actual timing information will be recorded and pulled from NSYS by a
            # script like benchmark.py.

            if self.is_inside_batch > 0:
                self.inside_batch_info.append(self.stack_path)

            # Record the batch information if it was present.
            if total_items is not None:
                if self.is_inside_batch <= 0:
                    raise ValueError(
                        "Non zero value for total_items in pop_range can only be "
                        "passed once inside a batch. No known batch was pushed previously. Please "
                        "push a batch first by using the batch_idx in the push_range()."
                    )

                self.batch_info[self.stack_path] = (batch_idx, total_items)
                self.is_inside_batch -= 1

                if total_items > 0:
                    batch_level_prefix = os.path.dirname(self.stack_path)

                    if batch_level_prefix not in self.total_batches_processed:
                        self.total_batches_processed[batch_level_prefix] = 0
                    self.total_batches_processed[batch_level_prefix] += 1

            # Unwind the stack to point to the previous path(i.e. directory like expression)
            # e.g. one level above.
            self.stack_path = os.path.dirname(self.stack_path)

        nvtx.pop_range(domain)

    def finalize(self):
        """
        Finalizes the performance benchmark data dictionary and saves it in the output folder
        as a JSON file. The benchmark data will be all zeros at this point. Actual data is
        captured and pulled from the NSYS reports when benchmark.py is run after this.
        """
        if self.should_benchmark:
            if len(self.stack):
                raise Exception(
                    "Unable to finalize timing info. The stack was non empty with %d"
                    " item(s) still not popped." % len(self.stack)
                )

            # Build a dictionary containing the timing information and some metadata
            # about this run.
            # The overall structure of this would be:
            # {
            #   "data" : {
            #       ...
            #   }
            #   "mean_data" : {
            #       ...
            #   }
            #   "batch_info" : {
            #       ...
            #   }
            #   "inside_batch_info" : [
            #       ...
            #   ]
            #   "meta" : {
            #       ...
            #   }
            # }
            #
            # The data field stores timing info of all batches keyed with raw flattened
            # names of the NVTX push/pop ranges.
            # The mean_data stores the mean timing info for NVTX ranges across
            # all the batches.
            # The batch_info stores the batch index and batch size of each batch.
            # The inside batch info is list of NVTX range names which executed inside
            # a batch.
            # The meta field stores various metadata about this run.
            #
            # NOTE: The numbers in the data/mean_batch_data field are all zero.
            #  i.e. They are only acting as placeholders. The actual numbers will be
            #       captured and pulled from NSYS when benchmarking is run with
            #       the benchmark.py script.
            #
            benchmark_dict = {
                "data": self.timing_info,
                "mean_data": {},
                "batch_info": self.batch_info,
                "inside_batch_info": self.inside_batch_info,
                "meta": {},
            }

            # Then we add basic details about this run and its configuration as meta.
            benchmark_dict["meta"] = {
                "obj_name": self.obj_name,
                "measurement_unit": "milliseconds",
                "total_batches": self.total_batches_processed,
                "cvcuda_version": cvcuda.__version__,
                "pytorch_version": torch.__version__,
                "python_version": sys.version,
            }
            if torch.cuda.device_count():
                benchmark_dict["meta"]["device"] = {
                    "id": self.device_id,
                    "name": torch.cuda.get_device_name(self.device_id),
                }
            else:
                benchmark_dict["meta"]["device"] = {
                    "id": self.device_id,
                    "name": "CPU",
                }

            benchmark_dict["meta"]["args"] = {}
            if self.command_line_args:
                for arg in vars(self.command_line_args):
                    benchmark_dict["meta"]["args"][arg] = getattr(
                        self.command_line_args, arg
                    )

            # The benchmark_dict is ready at this point. Convert it to JSON and write it.
            benchmark_json = json.dumps(benchmark_dict, indent=4)
            benchmark_file_path = os.path.join(self.output_dir, "benchmark.json")
            with open(benchmark_file_path, "w") as f:
                f.write(benchmark_json)
            self.logger.info(
                "Placeholder benchmark.json was written to: %s" % benchmark_file_path
            )

            return benchmark_dict

        else:
            return {}


def maximize_clocks(logger):
    """
    Maximizes the GPU clocks. Useful to do it before any type of performance
    benchmarking.
    """
    logger.info("Trying to maximize the GPU clocks...")

    gpu_available = torch.cuda.device_count() > 0

    was_persistence_mode_on = False
    current_power_limit = None

    if not gpu_available:
        logger.warning("No GPUs available to maximize the clocks.")
        return False, was_persistence_mode_on, current_power_limit

    # 1. Enable persistence mode if not already done.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=persistence_mode",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query persistence mode.")
        was_persistence_mode_on = False
    else:
        was_persistence_mode_on = proc_ret.stdout.decode() == "Enabled"

    if not was_persistence_mode_on:
        proc_ret = subprocess.run(
            ["nvidia-smi", "--persistence-mode=1"], stdout=subprocess.PIPE
        )
        if proc_ret.returncode:
            logger.error("Unable to set persistence mode.")
            return False, was_persistence_mode_on, current_power_limit
        else:
            logger.info("Turned on persistence mode.")

    # 2. Disable auto boost before locking clocks.
    proc_ret = subprocess.run(
        ["nvidia-smi", "--auto-boost-default=DISABLED"], stdout=subprocess.PIPE
    )
    if proc_ret.returncode:
        logger.warning("Unable to turn off auto boost mode.")
    else:
        logger.info("Turned off auto-boost mode.")

    # 3. Maximize the power limits.
    # Get the current value first and save it.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=power.limit",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query the current power limit.")
    else:
        current_power_limit = float(proc_ret.stdout.decode())

        # Get the maximum value after that.
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.max_limit",
                "--format=csv,nounits,noheader",
            ],
            stdout=subprocess.PIPE,
        )
        if proc_ret.returncode:
            logger.warning("Unable to query maximum power limit.")
        else:
            # Set the limit.
            max_power_limit = float(proc_ret.stdout.decode())
            proc_ret = subprocess.run(
                [
                    "nvidia-smi",
                    "--power-limit=%f" % max_power_limit,
                ],
                stdout=subprocess.PIPE,
            )

            if proc_ret.returncode:
                logger.warning("Unable to set maximum power limit.")
            else:
                logger.info("Set the maximum power limit to %f." % max_power_limit)

    # 4. Query the maximum allowed Graphics clock and lock it.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.max.graphics",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.error(
            "Unable to query the maximum graphics clock. Clocks were not maximized."
        )
        return False, was_persistence_mode_on, current_power_limit
    else:
        max_graphics_clock = float(proc_ret.stdout.decode())
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "--lock-gpu-clocks=%d,%d" % (max_graphics_clock, max_graphics_clock),
            ],
            stdout=subprocess.PIPE,
        )
        if proc_ret.returncode:
            logger.error("Unable to lock the GPU clock. Clocks were not maximized.")
            return False, was_persistence_mode_on, current_power_limit
        else:
            logger.info("Locked the GPU clock to %d." % (max_graphics_clock))

    # 5. Query the maximum allowed memory clock and lock it.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.max.memory",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.error(
            "Unable to query the maximum memory clock. Clocks were not maximized."
        )
        return False, was_persistence_mode_on, current_power_limit
    else:
        max_memory_clock = float(proc_ret.stdout.decode())
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "--lock-memory-clocks=%d,%d" % (max_memory_clock, max_memory_clock),
            ],
            stdout=subprocess.PIPE,
        )
        if proc_ret.returncode:
            logger.warning(
                "Unable to lock the memory clock. Clocks were not maximized."
            )
        else:
            logger.info("Locked the memory clock to %d." % max_memory_clock)

    # 6. Lock the application clocks. Specifies <memory,graphics> clocks as a pair
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "--applications-clocks=%d,%d" % (max_memory_clock, max_graphics_clock),
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to lock the application clocks.")
    else:
        logger.info(
            "Locked the application clocks to %d, %d."
            % (max_memory_clock, max_graphics_clock)
        )

    # 7. Get the GPU Performance State. P0 state means the most performance.
    proc_ret = subprocess.run(
        ["nvidia-smi", "--query-gpu=pstate", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query the performance state of the GPU.")

        return False, was_persistence_mode_on, current_power_limit
    else:
        gpu_perf_state = proc_ret.stdout.decode().strip()
        logger.info("Current GPU performance state is %s." % gpu_perf_state)

        if gpu_perf_state == "P0":
            logger.info("Clocks are now maximized.")
            return True, was_persistence_mode_on, current_power_limit
        else:
            logger.info("Unable to maximize all GPU clocks to reach the P0 state.")
            return False, was_persistence_mode_on, current_power_limit


def reset_clocks(
    logger,
    was_persistence_mode_on=False,
    current_power_limit=None,
):
    """
    Resets the GPU clocks.
    """
    logger.info("Trying to reset the GPU clocks...")

    gpu_available = torch.cuda.device_count() > 0

    if not gpu_available:
        logger.warning("No GPUs available to reset the clocks.")

    # 1. Reset the memory clock.
    proc_ret = subprocess.run(
        ["nvidia-smi", "--reset-memory-clocks"], stdout=subprocess.PIPE
    )
    if proc_ret.returncode:
        logger.warning("Unable to reset the memory clock back to normal.")
    else:
        logger.info("Reset the memory clock back to normal.")

    # 2. Reset GPU clock.
    proc_ret = subprocess.run(
        ["nvidia-smi", "--reset-gpu-clocks"], stdout=subprocess.PIPE
    )
    if proc_ret.returncode:
        logger.warning("Unable to reset the GPU clock back to normal.")
    else:
        logger.info("Reset the graphics clock back to normal.")

    # 3. Reset application clocks.
    proc_ret = subprocess.run(
        ["nvidia-smi", "--reset-applications-clocks"], stdout=subprocess.PIPE
    )
    if proc_ret.returncode:
        logger.warning("Unable to reset the application clocks back to normal.")
    else:
        logger.info("Reset the application clocks back to normal.")

    # 4. Reset the power limit.
    if current_power_limit:
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "--power-limit=%f" % current_power_limit,
            ],
            stdout=subprocess.PIPE,
        )

        if proc_ret.returncode:
            logger.warning("Unable to reset the power limit back to normal.")
        else:
            logger.info("Reset the power limit to normal.")

    # 5. Enable auto boost back..
    proc_ret = subprocess.run(
        ["nvidia-smi", "--auto-boost-default=ENABLED"], stdout=subprocess.PIPE
    )
    if proc_ret.returncode:
        logger.warning("Unable to turn the auto boost mode back on.")
    else:
        logger.info("Turned the auto-boost mode back on.")

    # 6. Turn off persistence mode if it was enabled by us.
    if not was_persistence_mode_on:
        proc_ret = subprocess.run(
            ["nvidia-smi", "--persistence-mode=0"], stdout=subprocess.PIPE
        )
        if proc_ret.returncode:
            logger.warning("Unable to turn off the persistence mode.")
        else:
            logger.info("Turned off the persistence mode.")

    # 7. Get GPU Performance State
    proc_ret = subprocess.run(
        ["nvidia-smi", "--query-gpu=pstate", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query the performance state of the GPU.")

        return False, was_persistence_mode_on, current_power_limit
    else:
        gpu_perf_state = proc_ret.stdout.decode().strip()
        logger.info("Current GPU performance state is %s." % gpu_perf_state)


def get_default_arg_parser(
    message,
    supports_video=True,
    input_path=None,
    output_dir="/tmp",
    target_img_height=224,
    target_img_width=224,
    batch_size=4,
    device_id=0,
    supported_backends=["tensorrt", "pytorch"],
    backend="tensorrt",
    log_level="info",
    parser_type="vision",
):
    """
    Prepares and returns an argparse command line argument parser for the scripts
    that supports auto-benchmarking. This parser guarantees that all the scripts which can be
    benchmarked supports a basic set of command line arguments which allows us to run
    them in a uniform and consistent fashion.
    """

    # Check what kind of parser the user needs.
    # 1. A vision parser:
    #       Adds all of the most commonly used command-line arguments for a typical
    #       computer vision pipeline.
    # 2. A minimal parser:
    #       Only adds the arguments needed for performance benchmarking.
    #
    if parser_type not in ["vision", "minimal"]:
        raise ValueError("parser_type must either be 'vision' or 'minimal.")

    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets",
    )

    if not input_path:
        input_path = os.path.join(assets_dir, "images", "Weimaraner.jpg")

    parser = argparse.ArgumentParser(
        message,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if parser_type == "vision":
        if not supports_video:
            parser.add_argument(
                "-i",
                "--input_path",
                default=input_path,
                type=str,
                help="The path to a JPEG image or a directory containing JPG images "
                "to use as input. When pointing to a directory, only *.jpg images will be read.",
            )
        else:
            parser.add_argument(
                "-i",
                "--input_path",
                default=input_path,
                type=str,
                help="Either a path to a JPEG image/MP4 video or a directory containing JPG images "
                "to use as input. When pointing to a directory, only *.jpg images will be read.",
            )

    if parser_type in ["vision", "minimal"]:
        parser.add_argument(
            "-o",
            "--output_dir",
            default=output_dir,
            type=str,
            help="The folder where the output results should be stored.",
        )

    if parser_type == "vision":
        parser.add_argument(
            "-th",
            "--target_img_height",
            default=target_img_height,
            type=int,
            help="The height to which you want to resize the input_image before "
            "running inference.",
        )

        parser.add_argument(
            "-tw",
            "--target_img_width",
            default=target_img_width,
            type=int,
            help="The width to which you want to resize the input_image before "
            "running inference.",
        )

        parser.add_argument(
            "-b",
            "--batch_size",
            default=batch_size,
            type=int,
            help="The batch size.",
        )

    if parser_type in ["vision", "minimal"]:
        parser.add_argument(
            "-d",
            "--device_id",
            default=device_id,
            type=int,
            help="The GPU device to use for this sample.",
        )

    if parser_type == "vision":
        parser.add_argument(
            "-bk",
            "--backend",
            type=str,
            choices=supported_backends,
            default=backend,
            help="The inference backend to use. Currently supports %s."
            % ", ".join(supported_backends),
        )

    if parser_type in ["vision", "minimal"]:
        parser.add_argument(
            "-ll",
            "--log_level",
            type=str,
            choices=["info", "error", "debug", "warning"],
            default=log_level,
            help="Sets the desired logging level. Affects the std-out printed by the "
            "sample when it is run.",
        )

    return parser


def parse_validate_default_args(parser):
    """
    Parses and validates the values of the default command line arguments.
    """
    args = parser.parse_args()

    if hasattr(args, "input_path"):
        if not os.path.isdir(args.input_path) and not os.path.isfile(args.input_path):
            raise ValueError(
                "input_path is neither a valid file not a directory: %s"
                % args.input_path
            )

    if hasattr(args, "output_dir"):
        if not os.path.isdir(args.output_dir):
            raise ValueError(
                "output_dir is not a valid directory: %s" % args.output_dir
            )

    if hasattr(args, "batch_size"):
        if args.batch_size <= 0:
            raise ValueError("batch_size must be a value >=1.")

    if hasattr(args, "device_id"):
        if torch.cuda.device_count():
            if args.device_id < 0 or args.device_id >= torch.cuda.device_count():
                raise ValueError(
                    "device_id must be a valid value from 0 to %d."
                    % (torch.cuda.device_count() - 1)
                )

    if hasattr(args, "target_img_height"):
        if args.target_img_height < 10:
            raise ValueError("target_img_height must be a value >=10.")

    if hasattr(args, "target_img_width"):
        if args.target_img_width < 10:
            raise ValueError("target_img_width must be a value >=10.")

    return args
