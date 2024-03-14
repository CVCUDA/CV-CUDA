
[//]: # "SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."


# Python Operator Performance Benchmarking

Using various performance benchmarking scripts that ships with CV-CUDA samples, we can measure and report the performance of various CV-CUDA operators from Python.

The following scripts are part of the performance benchmarking tools in CV-CUDA.

1. `samples/scripts/benchmark.py`
2. `samples/common/python/perf_utils.py`
3. `bench/python/bench_utils.py`

We use NVIDIA NSYS internally to benchmark Python code for its CPU and GPU run-times.


## About the Operator Benchmarks

Operators for which a test case has been implemented in the `all_ops` folder can be benchmarked. The following statements are true for all such test cases:

1. All inherit from a base class called `AbstractOpBase` which allows them to expose benchmarking capabilities in a consistent manner. They all have a setup stage, a run stage and an optional visualization stage. By default, the visualization is turned off.
2. All receive the same input image. Some operators may need to read additional data. Such data is always read from the `assets` directory.
3. All run for a number of iterations (default is set to 10) and a batch size (default is set to 32).
4. The script `benchmark.py` handles overall benchmarking. It launches the runs, monitors it, communicates with NSYS and saves the results of a run in a JSON file. Various settings such as using warm-up (default is set to 1 iteration) are handled here.
5. One or more benchmark runs can be compared and summarized in a table showing only the important information from the detailed JSON files.

## Setting up the environment

1. Follow [Setting up the environment](../../samples/README.md#setting-up-the-environment) section of the CV-CUDA samples. Note: The step asking to install dependencies can be ignored if you are only interested in benchmarking the operators (and not the samples).


## Running the benchmark

The script `run_bench.py` together with `benchmark.py` can be used to automatically benchmark all supported CV-CUDA operators in Python. Additionally, one or more runs can be summarized and compared in a table using the functionality provided by `bench_utils.py`


### To run the operator benchmarks

```bash
python3 samples/scripts/benchmark.py -o <OUT_DIR> bench/python/run_bench.py
```
- Where:
    1. An `OUTPUT_DIR` must be given to store various benchmark artifacts.
- Upon running it will:
    1. Ask the `benchmark.py` to launch the `run_bench.py`.
    2. `run_bench.py` will then find out all the operators that can be benchmarked.
    3. Run those one by one, through all the stages, such as setup, run and visualization (if enabled).
    4. Store the artifacts in the output folder. This is where the `benchmark.py` style `benchmark_mean.json` would be stored.

Once a run is completed, one can use the `bench_utils.py` to summarize it. Additionally, we can use the same script to compare multiple different runs.

### To summarize one run only

```bash
python3 bench/python/bench_utils.py -o <OUTPUT_DIR> -b <benchmark_mean_json_path> -bn baseline
```
- Where:
    1. A `OUTPUT_DIR` must be given to store the summary table as a CSV file.
    2. The first run's `benchmark_mean.json` path must be given as `b`.
    3. The display name of the first run must be given as `bn`.
- Upon running it will:
    1. Grab appropriate values from the JSON file for all the operators and put it in a table format.
    2. Save the table as a CSV file.

The output CSV file will be stored in the `OUTPUT_DIR` with current date and time on it.

NOTE: `benchmark.py` will produce additional JSON files (and visualization files if it was enabled). These files provide way more detailed information compared to the CSV and is usually only meant for debugging purposes.


### To summarize and compare multiple runs

```bash
python3 bench/python/bench_utils.py -o <OUTPUT_DIR> -b <benchmark_mean_json_path> -bn baseline \
       -c <benchmark_mean_2_json_path> -cn run_2 \
       -c <benchmark_mean_3_json_path> -cn run_3
```
- Where:
    1. An `OUTPUT_DIR` must be given to store the summary table as a CSV file.
    2. The first run's `benchmark_mean.json` path is given as `b`.
    3. The display name of the first run is given as `bn`.
    4. The second run's `benchmark_mean.json` path is given as `c`.
    5. The display name of the second run is given as `cn`.
    6. The third run's `benchmark_mean.json` path is given as `c`.
    7. The display name of the third run must be given as `cn`.
    8. Options `c` and `cn` can be repeated as zero or more times to cover all the runs.
- Upon running it will:
    1. Grab appropriate values from the JSON file for all the operators and put it in a table format.
    2. Save the table as a CSV file.


## Interpreting the results

Upon a successful completion of the `bench_utils.py` script, we would get a CSV file.

- If you ran it only on one run, your CSV will only have four columns - showing data only from that run:
    1. `index`: from 0 to N-1 for all the N operators benchmarked
    2. `operator name` The name of the operator
    3. `baseline run time (ms)`: The first run's time in milliseconds, averaged across M iterations (default is 10, with warm-up runs discarded)
    4. `run time params`: Any helpful parameters supplied to the operator as it ran in first run. Only lists primitive data-types.

- If you ran it on more than one runs, your CSV file will have additional columns - comparing data of those runs with the baseline run. Additional columns, per run, would be:
    1. `run i time (ms)`: The ith run's time in milliseconds, averaged across M iterations (default is 10, with warm-up runs discarded)
    2. `run i v/s baseline speed-up`: The speed-up factor. This is calculated by dividing `run i time (ms)` by `baseline run time (ms)`.
