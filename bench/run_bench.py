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

import os
import sys
import time
import subprocess
import pandas as pd


BENCH_PREFIX = "cvcuda_bench_"
BENCH_OUTPUT = "out.csv"
BENCH_COMMAND = "{} {} --csv {}"
BENCH_COLNAME = "Benchmark"
BENCH_RESULTS = "bench_output.csv"
BENCH_COLUMNS = {"Benchmark", "BWUtil", "Skipped"}
BANDWIDTH_COLNAME = "BWUtil"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "E At least one argument must be provided: benchmark folder"
            f"I Usage: {sys.argv[0]} bench_folder [extra args for benchmarks]"
        )
        sys.exit(1)

    bench_args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
    bench_folder = sys.argv[1]
    bench_files = [fn for fn in sorted(os.listdir(bench_folder)) if BENCH_PREFIX in fn]

    if len(bench_files) == 0:
        print(f"E No benchmarks found in {bench_folder}")
        sys.exit(1)

    print(f"I Found {len(bench_files)} benchmark(s) in {bench_folder} to run")

    l_df = []

    for filename in bench_files:
        filepath = os.path.join(bench_folder, filename)

        cmd = BENCH_COMMAND.format(filepath, bench_args, BENCH_OUTPUT)

        print(f'I Running "{cmd}"', end=" ")

        beg = time.time()
        subprocess.run(cmd.split(), stdout=subprocess.PIPE)
        end = time.time()

        print(f"took {end - beg:.03f} sec")

        if os.path.exists(BENCH_OUTPUT) is False or os.path.getsize(BENCH_OUTPUT) == 0:
            print("W Skipping as benchmark output does not exist or is empty")
            continue

        df = pd.read_csv(BENCH_OUTPUT)

        if not BENCH_COLUMNS.issubset(df.columns):
            print(f"W Skipping as benchmark output does not have: {BENCH_COLUMNS}")
            continue

        df = df[df["Skipped"] == "No"]

        os.remove(BENCH_OUTPUT)

        if len(df) > 0:
            l_df.append(df)

    df = pd.concat(l_df, axis=0)
    df = df.reset_index(drop=True)

    filepath = os.path.join(bench_folder, BENCH_RESULTS)

    df.to_csv(filepath)

    print(f"I Full results written to {filepath}")

    df = df.groupby("Benchmark")["BWUtil"].mean()

    pd.options.display.float_format = "{:.2%}".format

    print(f"I Summary results:\n{df}")
