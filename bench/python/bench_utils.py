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
import torch
from PIL import Image
import os
import logging
import inspect
import argparse
import importlib
from glob import glob
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import json
import pandas

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(name)s:%(lineno)d] %(asctime)s %(levelname)-6s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AbstractOpBase(ABC):
    """
    This is an abstract base class of all the operators that can be benchmarked.
    It provides basic functionality and guarantees uniformity across operator test cases.
    Concrete implementation of all the abstract methods of this class must be provided by
    the class inheriting from this.
    """

    def __init__(self, device_id, input, output_dir=None, should_visualize=False):
        """
        Initializes a new instances of this class.
        :param device_id: The GPU device id that to use by this operator.
        :param input: The input tensor to run the operator on.
        :param output_dir: The directory where artifacts should be stored.
        :param should_visualize: A flag specifying whether the output from the operator
         should be visualized and written to the disk or not.
        """
        self.device_id = device_id
        self.input = input
        self.output_dir = output_dir
        self.should_visualize = should_visualize
        if self.output_dir:
            if not os.path.isdir(self.output_dir):
                raise ValueError("A valid output_dir must be given.")
        self.op_output = None

        self.assets_dir = os.path.join(
            Path(os.path.abspath(__file__)).parents[0], "assets"
        )
        self.setup(self.input)

    def __call__(self, input):
        """
        Runs the operator on a given input. Also visualizes the output if visualization was set to True.
        :param input: The input tensor to run the operator on.
        :returns: True if the operator executed successfully, False otherwise.
        """
        try:
            self.op_output = self.run(input)

            if self.should_visualize and self.output_dir:
                self.visualize()

            return True
        except Exception as e:
            logger.error(
                "Unable to run the op %s due to error: %s"
                % (self.__class__.__name__, str(e))
            )
            return False

    @abstractmethod
    def setup(self, input):
        """
        Performs various setup activities to set this operator before it can be run.
        :param input: The input tensor to run the operator on.
        """
        pass

    @abstractmethod
    def run(self, input):
        """
        Runs the operator and returns the result.
        :param input: The input tensor to run the operator on.
        :returns: The result from the operator's run.
        """
        pass

    def get_params_info(self, primitive_types_only=True):
        """
        Returns a dictionary with keys being the variable names initialized exclusively during the setup call
        # and values being their values. Useful to log if someone wants to know what parameters were used to
        initialize the operator in the setup function call.
        :param primitive_types_only: Only includes attributes with primitive data-types if True. Primitive
         data types are bool, str, int, float, tuple and None.
        """
        primitives = (bool, str, int, float, tuple, type(None))

        # Get all global names (e.g variables + function names) used by the setup function.
        all_global_names_setup_func = set(self.setup.__code__.co_names)

        # Get all the global names (e.g variables + function names) used by the __init__ function.
        all_global_names_init_func = set(self.__init__.__code__.co_names)

        # Remove the names already used by __init__ from the ones used by setup to get a list of names
        # which are exclusively used by setup
        all_global_names_setup_func -= all_global_names_init_func

        # Get all the variables of this class.
        all_vars_info = vars(self)
        all_vars_names = set(all_vars_info.keys())

        # Figure out all global variables only by intersecting the all_vars_names with
        # all_global_names_setup_func.
        # That will eliminate the global function names from all_global_names_setup_func.
        vars_names_of_setup_function = all_vars_names.intersection(
            all_global_names_setup_func
        )

        if primitive_types_only:
            vars_info_of_setup_function = {
                v: all_vars_info[v]
                for v in vars_names_of_setup_function
                if isinstance(all_vars_info[v], primitives)
            }
        else:
            vars_info_of_setup_function = {
                v: all_vars_info[v] for v in vars_names_of_setup_function
            }

        return vars_info_of_setup_function

    def _setup_clear_output_dir(self, filename_ends_with):
        output_dir = os.path.join(self.output_dir, self.__class__.__name__)

        # Clear out the output directory or create it
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        else:
            for file in os.listdir(output_dir):
                if os.path.isfile(file) and file.endswith(filename_ends_with):
                    os.remove(file)

        return output_dir

    def visualize(self):
        """
        Attempts to visualize the output produced by the operator as an image by writing it
        down to the disk. May raise exceptions if visualization is not successful.
        """
        output_dir = self._setup_clear_output_dir(filename_ends_with="_op_out.jpg")
        if self.op_output is None:
            raise TypeError(
                "Visualization Error: Operator did not return any value as output to visualize."
            )

        op_output_npy = (
            torch.as_tensor(self.op_output.cuda(), device="cuda:%d" % self.device_id)
            .cpu()
            .numpy()
        )
        if op_output_npy.dtype == np.uint8:
            for i, npy_img in enumerate(op_output_npy):
                if npy_img.shape[-1] == 1:
                    # Need to drop the 1 from the channels dimension if dealing with
                    # grayscale in PIL
                    npy_img = npy_img[..., 0]
                out_file_name = "img_%d_op_out.jpg" % i
                # Visualize as image
                pil_img = Image.fromarray(npy_img)
                pil_img.save(os.path.join(output_dir, out_file_name))

        else:
            raise TypeError(
                "Visualization Error: Unsupported dtype for visualization: %s"
                % str(op_output_npy.dtype)
            )


def get_benchmark_eligible_ops_info():
    """
    Prepares list of tuples : op-class-name (str) and class for all the operators that can be benchmarked.
    """
    class_members = []

    for file in glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_ops", "*.py")
    ):
        name = os.path.splitext(os.path.basename(file))[0]
        module = importlib.import_module("all_ops." + name)
        all_members = inspect.getmembers(module, inspect.isclass)
        op_members = [x for x in all_members if x[0].startswith("Op")]

        class_members.extend(op_members)

    return class_members


def summarize_runs(
    baseline_run_json_path,
    baseline_run_name="baseline",
    compare_run_json_paths=[],
    compare_run_names=[],
):
    """
    Summarizes one or more benchmark runs and prepares a pandas table showing the per operator run-time
     and speed-up numbers.
    :param baseline_run_json_path: Path to where the benchmark.py styled JSON of the first run is stored.
    :param baseline_run_name: The display name of the column representing the first run in the table.
    :param compare_run_json_paths: Optional. A list of path to where the benchmark.py styled JSON of
     the other runs are stored. These runs are compared with the baseline run.
    :param compare_run_names: A list of display names of the column representing the comparison runs
     in the table. This must be of the same length as the `compare_run_json_paths`.
    :returns: A pandas table with the operator name, its run time from the baseline run and the params.
     used to launch those runs. If compare runs are given, it also returns their run times and the speed-up
     compared to the baseline run. The speedup is simply the run time of an operator from the compare run
     divided by its run time from the baseline run. If an operator's run time or speedup factor is not
     available, it simply puts "N/A".
    """
    if os.path.isfile(baseline_run_json_path):
        with open(baseline_run_json_path, "r") as f:
            baseline_perf = json.loads(f.read())
    else:
        raise ValueError(
            "baseline_run_json_path does not exist: %s" % baseline_run_json_path
        )

    if len(compare_run_json_paths) != len(compare_run_names):
        raise ValueError(
            "Length mismatch between the number of given JSON paths for comparison and"
            "their run names. %d v/s %d. Each JSON must have its corresponding run name."
            % (len(compare_run_json_paths), len(compare_run_names))
        )

    # Read all the comparison related JSON files, one by one, if any.
    compare_perfs = {}
    for compare_json_path, compare_run_name in zip(
        compare_run_json_paths, compare_run_names
    ):
        if os.path.isfile(compare_json_path):
            with open(compare_json_path, "r") as f:
                compare_perfs[compare_run_name] = json.loads(f.read())
        else:
            raise ValueError("compare_json_path does not exist: %s" % compare_json_path)

    results = []

    for op in baseline_perf["data_mean_all_procs"]["run_bench"]:
        if op.startswith("Op"):
            op_name = op[2:]

            row_dict = {}

            # Fetch the time and parameters from the JSON for baseline run.
            baseline_run_time = baseline_perf["data_mean_all_procs"]["run_bench"][op][
                "run_op"
            ]["cpu_time_minus_warmup_per_item"]["mean"]

            op_params = list(
                baseline_perf["data_mean_all_procs"]["run_bench"][op][
                    "op_params"
                ].keys()
            )[0]

            row_dict["operator name"] = op_name
            row_dict["%s time (ms)" % baseline_run_name] = baseline_run_time

            if compare_perfs:
                # Fetch the time from the JSON for all comparison runs.
                for compare_run_name in compare_perfs:
                    # Check if the OP was present.
                    if (
                        op
                        in compare_perfs[compare_run_name]["data_mean_all_procs"][
                            "run_bench"
                        ]
                    ):
                        compare_run_time = compare_perfs[compare_run_name][
                            "data_mean_all_procs"
                        ]["run_bench"][op]["run_op"]["cpu_time_minus_warmup_per_item"][
                            "mean"
                        ]
                    else:
                        compare_run_time = None

                    row_dict["%s time (ms)" % compare_run_name] = (
                        compare_run_time if compare_run_time else "N/A"
                    )

                    if baseline_run_time and compare_run_time:
                        speedup = round(compare_run_time / baseline_run_time, 3)
                    else:
                        speedup = "N/A"
                    row_dict[
                        "%s v/s %s speed-up" % (compare_run_name, baseline_run_name)
                    ] = speedup

            row_dict["run time params"] = op_params

            results.append(row_dict)

    pandas.set_option("display.max_colwidth", 100)

    df = pandas.DataFrame.from_dict(results)

    return df


def main():
    """
    The main function. This will run the comparison function to compare two benchmarking runs.
    """
    parser = argparse.ArgumentParser("Summarize and compare benchmarking runs.")

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="The output directory where you want to store the result summary as a CSV file.",
    )

    parser.add_argument(
        "-b",
        "--baseline-json",
        type=str,
        required=True,
        help="Path where the benchmark.py styled JSON of the baseline run is stored.",
    )
    parser.add_argument(
        "-bn",
        "--baseline-name",
        type=str,
        required=True,
        help="The name of the column representing the baseline run in the output table.",
    )
    parser.add_argument(
        "-c",
        "--compare-jsons",
        action="append",
        required=False,
        help="Optional. List of paths where the benchmark.py styled JSON of the comparison run are stored.",
    )
    parser.add_argument(
        "-cn",
        "--compare-names",
        action="append",
        required=False,
        help="Optional. List of names of the column representing the comparison runs in the output table.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise ValueError("output-dir does not exist: %s" % args.output_dir)

    if not os.path.isfile(args.baseline_json):
        raise ValueError("baseline-json does not exist: %s" % args.baseline_json)

    args.compare_jsons = args.compare_jsons if args.compare_jsons else []
    args.compare_names = args.compare_names if args.compare_names else []

    if len(args.compare_jsons) != len(args.compare_names):
        raise ValueError(
            "Length mismatch between the number of given JSON paths for comparison and"
            "their run names. %d v/s %d. Each JSON must have its corresponding run name."
            % (len(args.compare_jsons), len(args.compare_names))
        )

    logger.info(
        "Summarizing a total of %d runs. All times are in milliseconds"
        % (len(args.compare_jsons) + 1)
    )

    df = summarize_runs(
        baseline_run_json_path=args.baseline_json,
        baseline_run_name=args.baseline_name,
        compare_run_json_paths=args.compare_jsons,
        compare_run_names=args.compare_names,
    )

    csv_path = os.path.join(
        args.output_dir,
        "summarize_runs.%s.csv" % datetime.now(),
    )
    df.to_csv(csv_path)

    logger.info("Wrote comparison CSV to: %s" % csv_path)


if __name__ == "__main__":
    # If this was called on its own, we will run the summarize_runs function to summarize
    # and compare two runs.
    main()
