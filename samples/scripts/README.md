# Performance Benchmarking

CV-CUDA samples ships with the following scripts that can help track and report the performance of the Python samples.

1. `scripts/benchmark.py`
2. `common/python/perf_utils.py`

We use NVIDIA NSYS to profile the code for its CPU and GPU run-times. Profiling is done at the sample level and also at each stage present in a sample's pipeline. NVTX style markers are used to annotate the code using `push_range()` and `pop_range()` methods and then benchmarking is run to collect the timing information of all such ranges. Please refer to the NVIDIA NSIGHT user guide to learn more about NSIGHT and NVTX trace (https://docs.nvidia.com/nsight-systems/UserGuide/index.html)


## benchmark.py

This is the main launcher script responsible for launching a sample in a multi-CPU multi-GPU fashion. After launching the sample it:

1. Coordinates the entire process of benchmarking with NSYS.
2. Parses various results returned by NSYS.
3. Stores per-process results in a JSON file.
4. Automatically calculates average numbers across all processes and also saves them in a JSON file.


## perf_utils.py

This file holds the data structures and functions most commonly used during the benchmarking process:

1. It contains the `CvCudaPerf` class to hold performance data with an API similar to NVTX.
2. Provides a way to maximize the GPU clocks before benchmarking.
3. Provides a command-line argument parser that can be shared across all samples to maintain uniformity in the way of passing the inputs.


## The Benchmarking Flow

With these tools, the benchmarking flow involves the following two steps:

1. Annotating the code of the sample using classes and functions from the `perf_utils.py` so that it can be profiled.
    1. Import the necessary classes and functions first
        ```python
        from perf_utils import CvCudaPerf, get_default_arg_parser, parse_validate_default_args
        ```

        1. The `CvCudaPerf` class is used to used to mark the portions of code that one wants to benchmark using its `push_range()` and `pop_range()` methods. These methods are similar to NVTX except that it adds two new features on the top that allows us to compute more detailed numbers:
            1. It can record the start of a batch. A batch is a logical group of operations that often repeats.
            2. It can record the end of a batch, with its batch size.
        2. The `get_default_arg_parser` function is used to create an argument parser with the most frequently used command-line arguments for a computer vision pipeline already added in it. One can also add additional arguments if needed.
        3. The `parse_validate_default_args` is used to parse and validate the command-line arguments.

    2. Create an instance of `CvCudaPerf` and use it. We use the `push_range()` method to start a new level and `pop_range()` to end it. One can have any level of nested hierarchy with this. To indicate a start of a batch, we simply pass the `batch_idx` inside the `push_range()` and to indicate the end of a batch we pass the `total_items` in the `pop_range()`. The following is a very simple example showing the overall flow in the code of a toy sample:

       ```python
       # Create the objects.
       parser = get_default_arg_parser("perf_utils_test")
       args = parse_validate_default_args(parser)
       cvcuda_perf = CvCudaPerf("perf_utils_test", default_args=args)

       # Create a simple pipeline with two stages.
       # The stages does not use any GPU code in this example.
       cvcuda_perf.push_range("pipeline")

       for i in range(3):
           cvcuda_perf.push_range("batch", batch_idx=i)

           cvcuda_perf.push_range("stage_1")
           # Do some work
           time.sleep(0.3)
           cvcuda_perf.pop_range()

           cvcuda_perf.push_range("stage_2")
           # Do some work
           time.sleep(0.3)
           cvcuda_perf.pop_range()

           cvcuda_perf.pop_range(total_items=1)

       cvcuda_perf.pop_range()

       # Once everything is done, we must call the finalize().
       cvcuda_perf.finalize()
       ```
2. Use the sample with the `benchmark.py` to launch the benchmarking. `benchmark.py` can launch any script that uses `perf_utils`'s functionality and benchmark it using NSYS. It can also launch it in a multi-CPU multi-GPU fashion to compute the throughput.

    1. To benchmark the object detection sample, for example, we can use the following command:

       ```bash
       clear && python scripts/benchmark.py -np 2 -w 1 -o /tmp object_detection/python/main.py -b 4 -i assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4
       ```
        - This will:
            1. Launch two CPU process, each using the same GPU using the `-np 2` option
            2. Use the first and last batch as the warmup batch and discard its results using the `-w 1` option
            3. Write the artifacts from two processes in the `/tmp` directory using the `-o /tmp` option
            4. Launch the object detection script with batch size 4 and using a video from the assets folder.
            5. Wait for the completion of all processes. Once completed, it stores the per process `benchmark.json` in the two folders `/tmp/proc_0_gpu_0` and `/tmp/proc_1_gpu_0` and one JSON at `/tmp/benchmark_mean.json` containing the average results.

        **NOTE**: You may need to run the above command twice if you are running it for the first time or with a different output directory as some samples may spend quite a bit of time in creating a TensorRT model on the first run. This may impact the performance numbers of the first run. Subsequent runs will use a pre-existing model (if available).


## Interpreting the results


Upon a successful completion of the `benchmark.py` script, we get the following files:

1. Per process statistics in `benchmark.json` files. These are stored in `<OUTPUT_DIR>/proc_X_gpu_Y` where `OUTPUT_DIR` is the directory used to store the output, `X` is the CPU index and `Y` is the GPU index. If one ran the `benchmark.py` with `-np 3`, there will be a total of 3 such `benchmark.json` files, one per each process.
    In each `benchmark.json` file one can see an overall structure like this:
    ```json
    {
        "data": {},
        "mean_data": {},
        "meta": {}
    }
    ```
    1. The `data` key stores the per batch data maintaining the hierarchy of the pipeline. At each non-batch level, it stores the following information:
        ```json
        "pipeline": {
            "batch_0": {
                "stage_1": {
                    "cpu_time": 300.797,
                    "gpu_time": 0.0
                },
                "stage_2": {
                    "cpu_time": 300.416,
                    "gpu_time": 0.0
                },
                "cpu_time": 601.373,
                "gpu_time": 0.0,
                "total_items": 1,
                "cpu_time_per_item": 601.373,
                "gpu_time_per_item": 0.0
            }
        }
        ```

        1. One can see the `cpu_time` and `gpu_time` per stage.
        2. The `total_items` (i.e. the batch size) is also tracked at the batch level and per item numbers are computed from it.

    2. At the batch level, the statistics are aggregated from all the batches and reported with considering the warm-up batches. The `*_minus_warmup` timings are the ones which ignore the warm-up batches from the computation.
        ```json
        {
            "cpu_time": 1805.027,
            "gpu_time": 0.0,
            "cpu_time_per_item": 601.676,
            "gpu_time_per_item": 0.0,
            "total_items": 3,
            "cpu_time_minus_warmup": 601.759,
            "gpu_time_minus_warmup": 0.0,
            "cpu_time_per_item_minus_warmup": 601.759,
            "gpu_time_per_item_minus_warmup": 0.0,
            "total_items_minus_warmup": 1
        }
        ```
    3. The `mean_data` key stores the average of all numbers across all the batches.
    4. The `meta` key stores various metadata about the run. This may be useful for reproducibility purposes.


2. Overall statistics in the `benchmark_mean.json` file. This file will be stored in `<OUTPUT_DIR>` where `<OUTPUT_DIR>` is the directory used to store the output.
    In `benchmark_mean.json` file one can see an overall structure like this:
    ```json
    {
        "mean_all_batches": {},
        "mean_data": {},
        "meta": {}
    }
    ```
    1. The `mean_all_batches` key stores average per batch numbers from all processes launched by the `benchmark.py`.  These are essentially the mean of the `data` field reported in the the per process' `benchmark.json` file and maintains the overall pipeline hierarchy.
    2. The `mean_data` key stores the average numbers from all batches from all processes. These are essentially the mean of the `mean_data` reported in the the per process' `benchmark.json` file.
    3. The `meta` key stores various metadata about the run. This may be useful for reproducibility purposes.


## Regarding maximizing the clocks

Often during the GPU benchmarking process one would like to set the GPU clocks and power to their maximum settings. While the `nvidia-smi` command and `nvml` APIs both provide various options to do so, we have consolidated these into a convenient function call `maximize_clocks()` in the `perf_utils.py` script. Once can easily turn it on during the benchmarking process by passing the `--maximize_clocks` flag to the `benchmark.py` script. This will also bring the clocks down to its original values once the process is over.
