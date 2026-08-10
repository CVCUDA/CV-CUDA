<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CV-CUDA Benchmarks

CV-CUDA provides C++ and Python operator benchmarks built on
[nvbench](https://github.com/NVIDIA/nvbench). They report GPU time, measurement
noise, and bandwidth utilization; paired C++ and Python runs also check
performance parity. Committed performance baselines are available for NVIDIA
A100 and H100 GPUs.

## Choose a Workflow

| Goal | Command | Main output |
|---|---|---|
| Run C++ and/or Python benchmarks | `run_bench.py` | `bench_output.csv` or JSON |
| Compare two Python wheels | `compare_wheels.py` | `summary.md` and `comparison.csv` |
| Compare a run with committed baselines | `compare_to_baseline.py` | Console, Markdown, or JUnit report |

Use `--help` on any command for its complete option list.

## Setup

The benchmarks require Python 3.10+, CUDA Toolkit 12 or 13, and nvbench. The
development container provides the expected environment. Otherwise, install the
Python benchmark dependencies from the repository root:

```bash
bench/python/install_bench_dependencies.sh
```

Build the benchmark component by following the
[installation guide](../docs/sphinx/installation.rst). The examples below assume
the build output is `build-rel/bin`.

`run_bench.py` and `compare_wheels.py` are copied into build and install
directories. `compare_to_baseline.py` and the baseline-maintenance helpers are
source-tree tools.

## Run Benchmarks

From the benchmark build directory, a command with no selection options runs the
`basic` tier for both C++ and Python:

```bash
cd build-rel/bin

# Default basic suite, both languages
python3 run_bench.py

# Select exact operators or one language
python3 run_bench.py --operator resize,gaussian
python3 run_bench.py --lang python --operator resize

# Select deeper profiles or one exact configuration
python3 run_bench.py --tier basic,advanced
python3 run_bench.py \
  --config-key resize_expand_linear_1080p_uchar3_basic

# Discover valid operator names
python3 run_bench.py --list-operators
```

Operator names come from `config/bench_params.json`. `--config-key` bypasses
operator and tier selection.

The default result is `bench_output.csv`. Use JSON when the result will be
compared with or imported into committed baselines:

```bash
python3 run_bench.py --operator resize --output bench_output.json
```

The runner fails noisy results. When both languages run, it also checks that C++
and Python timings agree within the configured relative and absolute thresholds.
`--skip-validation` retains row statuses but prevents those checks from failing
the final run.

Output from `python3 run_bench.py --lang python --config-key
resize_expand_linear_1080p_uchar3_basic` is shown below. It is abridged, and
timings vary by GPU:

```text
=== Running Benchmarks ===
→ GPU: NVIDIA Graphics Device
→ Found 1 benchmarks

[1/1] resize_expand_linear_1080p_uchar3_basic
| Benchmark | config_key                              | ... | Py (µs) | Py Noise      | Status |
| resize    | resize_expand_linear_1080p_uchar3_basic | ... | 1,972.7 | ±0.07% (±1µs) | PASS   |

✓ All validations passed (noise <10.0%, perf diff <10.0% AND <100us)
✓ Results written to bench_output.csv

--- Summary ---
  Operators: 1, Configurations: 1 (1 passed, 0 failed)
  Py noise: ±0.07% (±1.4µs)
```

Exit status is `0` for a successful validated run, `1` for an execution or
validation failure, and `2` for invalid arguments or configuration.

## Compare Two Python Wheels

`compare_wheels.py` installs the reference and candidate wheels into separate
temporary environments, runs the same current Python benchmark harness against
each, and compares the results sequentially on the same GPU.

Both wheels must use the same distribution and CUDA channel, such as
`cvcuda-cu12`. The selected Python interpreter must support both wheel tags and
already provide the benchmark dependencies. The output directory must be new or
empty.

From the repository root:

```bash
python3 bench/compare_wheels.py \
  /path/to/reference/cvcuda_cu12-*.whl \
  /path/to/candidate/cvcuda_cu12-*.whl \
  --operator resize,gaussian \
  --output-dir wheel-comparison
```

From a build or installed benchmark directory, omit the `bench/` prefix.

The default selection is the complete `basic` tier. Use `--tier
basic,advanced` for both tiers, `--operator` for exact operator names, or
`--config-key` for one or more exact configurations.

The output directory contains:

```text
wheel-comparison/
├── baseline/bench_output.csv
├── candidate/bench_output.csv
├── comparison.csv
└── summary.md
```

The baseline and candidate directories also contain logs, wheel metadata, and
GPU diagnostics.

This abridged example is from the tested v0.16.0 versus v0.17.0-pre comparison:

```text
- baseline: `cvcuda-cu12 0.16.0 [...]`
- candidate: `cvcuda-cu12 0.17.0rc0 [...]`
- comparison status: **FAIL** — reference `run_bench.py` exited 1; 213 configurations only in candidate; 1 configuration non-PASS
- artifacts: `wheel-comparison`
  - reference benchmark CSV: `baseline/bench_output.csv`
  - candidate benchmark CSV: `candidate/bench_output.csv`
  - comparison CSV: `comparison.csv`

## Summary

- overall candidate speedup vs baseline: **1.8258x** (mean of per-operator geomeans across matched benchmarks)
- valid compatible operators: 49/60
- valid compatible configurations: 146/360
- configuration regressions over threshold: 0
- reference `run_bench.py`: **FAIL** (exit code 1)
- candidate `run_bench.py`: **PASS** (exit code 0)

## Compatibility

- operators partially compatible: 43/60
- operators fully compatible: 7/60
- expanded benchmark configurations incompatible or not present: 213/360
- present in both but non-PASS: 1

## Top 5 operator improvements

| Operator | Speedup | Candidate time delta | Matched configurations |
|---|---:|---:|---:|
| gaussiannoise | 13.0905x | -92.36% | 3 |
...

## Top 5 operator regressions

_none_
```

This fails because the older reference is incomplete relative to the current
harness, not because a configuration regressed. The overall result is the
arithmetic mean of per-operator geometric speedups, so every operator has equal
weight. Compatibility counts include present non-PASS rows; speedups exclude
them.

Exit status is `0` when both runs and the comparison pass, `1` when a run or
comparison check fails after producing the report, and `2` for setup or artifact
errors.

## Compare with Committed Baselines

Committed baselines are embedded in `bench/config/operators/*.json` for these
reference GPU identities:

- `A100_PCIE_40GB_250W_1095MHz`
- `H100_PCIe_350W_1095MHz`

First create a JSON result, then compare it from the repository root:

```bash
cd build-rel/bin
python3 run_bench.py --output bench_output.json
cd ../..

python3 bench/compare_to_baseline.py \
  --current build-rel/bin/bench_output.json
```

For a scoped comparison, add `--operator resize` to both commands. The JSON
input must contain only the selected operators.

The tool resolves the SKU from the JSON result. It fails on regressions,
unexpected improvements that may indicate stale baselines, missing or new rows,
and missing SKU data. The default regression and improvement thresholds are
both 10%.

Abridged passing output:

```text
Resolved SKU: A100_PCIE_40GB_250W_1095MHz (current JSON SKU key)
matched=720 regressions=0 improvements=0 missing=0 new=0 missing_sku=0

- thresholds: regression=0.1, improvement=0.1
- all-rows |Delta|: median 0.30%, max 5.61%

## Regressions (0)
_none_

## Unexpected improvements (0)
_none_

```

The report prints to the console by default. It can also be written as Markdown
or JUnit:

```bash
python3 bench/compare_to_baseline.py \
  --current build-rel/bin/bench_output.json \
  --markdown comparison.md --junit comparison.xml
```

Exit status is `0` when all expected rows match within the thresholds, `1` when
the comparison finds an incompatibility or timing failure, and `2` for invalid
input or configuration.

## Advanced Benchmark Usage and Configuration

### Directory Structure

```text
bench/
├── config/
│   ├── bench_params.json        # Operator manifest
│   ├── operators/               # Per-operator cases and embedded baselines
│   └── sku_map.json             # Supported benchmark GPU identities
├── cpp/ops/                     # C++ operator benchmarks
├── python/ops/                  # Python operator benchmarks
├── _internal/                   # Shared helpers and baseline maintenance
├── run_bench.py                 # Run C++ and/or Python benchmarks
├── compare_wheels.py            # Compare two Python wheels
└── compare_to_baseline.py       # Compare a run with committed baselines
```

The three top-level scripts are the supported benchmark commands. Files under
`_internal/` are implementation and maintenance helpers.

### Individual Benchmark Drivers

Individual drivers expose their nvbench axes directly. Python drivers also
require a configuration key:

```bash
cd build-rel/bin
./bench_resize --list
python3 bench_resize.py \
  --config-key resize_expand_linear_1080p_uchar3_basic --list
```

Remove `--list` and use nvbench `--axis` filters to run a direct selection.
Prefer `run_bench.py` for combined output and validation.

### Configuration Files

`config/bench_params.json` is the operator manifest. Each operator points to a
file under `config/operators/` and its matching C++ and Python benchmarks:

```json
{
  "operators": {
    "resize": {
      "config": "operators/resize.json",
      "cpp": "bench_resize",
      "python": "bench_resize.py"
    }
  }
}
```

Each operator file defines stable configuration keys, tiers, data types, axes,
warmup behavior, and machine-owned baselines. This schema example is abridged:

```json
{
  "benchmark": "resize",
  "configs": {
    "resize_basic": {
      "tier": "basic",
      "dtypes": ["uint8", "float32"],
      "string_axes": {
        "shape": ["1x1080x1920"],
        "interpolation": ["LINEAR", "CUBIC"],
        "inputKind": ["Tensor", "VarShape"]
      },
      "int64_axes": {"kernelSize": [3, 5]},
      "float64_axes": {"sigma": [1.2]},
      "warmup_iterations": 100,
      "baselines": {
        "resize_basic[InOutDataType=uint8][shape=1x1080x1920][interpolation=LINEAR][inputKind=Tensor][kernelSize=3][sigma=1.2]": {
          "H100_PCIe_350W_1095MHz": {
            "n_runs": 20,
            "gpu_time_us_cpp": 100.0,
            "gpu_time_us_python": 103.0,
            "gpu_noise_us_cpp": 1.0,
            "gpu_noise_us_python": 2.0,
            "gpu_bwutil_cpp": 0.42,
            "gpu_bwutil_python": 0.41,
            "gpu_gap_stddev_us": 3.0
          }
        }
      }
    }
  }
}
```

### Field Reference

| Field | Description |
|---|---|
| Manifest `config` | Per-operator JSON file under `config/operators/`. |
| Manifest `cpp` / `python` | Matching C++ binary and Python benchmark script. |
| `benchmark` | The `bench_<op>` target for this operator file. |
| `configs` | Configuration entries keyed by stable `config_key`. |
| `tier` | `basic` runs by default; `advanced` is explicitly selected. |
| `dtypes` | Input/output data types used by the benchmark. |
| `string_axes` | String-valued axes such as shape, mode, layout, and `inputKind`. |
| `int64_axes` / `float64_axes` | Integer and floating-point benchmark axes. |
| `warmup_iterations` | Warmup count before measurement; default is zero. |
| `baselines` | Machine-owned case/SKU measurements keyed by the fully expanded case identity. |

A baseline metric payload records the number of imported runs, C++ and Python
GPU time, measurement noise, bandwidth utilization, and optionally the
cross-run C++/Python gap standard deviation.

`inputKind` selects the input container: `Tensor` uses a dense tensor and
`VarShape` uses `ImageBatchVarShape`. Operators with a distinct batch-of-tensors
API may also implement `TensorBatch`.

Python benchmarks load the manifest and operator files at registration.
`run_bench.py` passes selected axis values to C++ benchmarks at runtime. Value
changes do not require a rebuild; rebuild the affected `bench_<op>` target when
adding a new axis name, data type, or operator:

```bash
cmake --build build-rel --target bench_resize
```

`--warmup-cap N` limits configured warmups for one run without modifying the
configuration. Passing zero disables warmup.

## Advanced Baseline Maintenance

Baseline blocks are generated data. Do not hand-edit them except for surgical
recovery; keep the source run or CI link with every reviewed update.


### Import, Validate, and Recheck

Dry-run the import before writing operator files:

```bash
python3 bench/_internal/update_baseline.py --from /path/to/artifacts \
  --operator resize --dry-run

python3 bench/_internal/update_baseline.py --from /path/to/artifacts \
  --operator resize \
  --write-summary baseline-update.md
```

Validate the updated baselines, including same-key timing changes against the
branch base:

```bash
python3 bench/_internal/validate_baselines.py \
  --operator resize --reject-regressions-from origin/main
```

Finally, create a fresh JSON run and compare it with the updated baselines, then
run the full MR matrix.

`--allow-regressions` waives only the same-key slowdown check and is reserved
for intentional, reviewed baseline resets. Schema, SKU, noise, C++/Python
parity, and fresh-run comparison checks remain enforced.

## Resources

- [CV-CUDA documentation](https://docs.nvidia.com/cvcuda/)
- [nvbench documentation](https://github.com/NVIDIA/nvbench)
- [CV-CUDA installation guide](../docs/sphinx/installation.rst)
