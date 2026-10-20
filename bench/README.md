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
| Measure Python binding overhead | `binding_bench.py` | Raw-window JSON and a console summary |
| Compare Python binding overhead | `compare_binding_bench.py` | Console comparison and JUnit report |

Use `--help` on any command for its complete option list.

## Setup

The operator benchmarks require Python 3.10+, CUDA Toolkit 12 or 13, and
nvbench. The development container provides the expected environment.
Otherwise, install the Python benchmark dependencies from the repository root:

```bash
bench/python/install_bench_dependencies.sh
```

Build the benchmark component by following the
[installation guide](../docs/sphinx/installation.rst). The examples below assume
the build output is `build-rel/bin`.

`run_bench.py`, `compare_wheels.py`, and `binding_bench.py` are copied into
build and install directories. `compare_to_baseline.py`, `compare_binding_bench.py`,
and the baseline-maintenance helpers are source-tree tools.

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

For an operator change, run the full `basic,advanced` surface for both languages,
including affected sibling operators when shared code changes:

```bash
python3 run_bench.py --operator resize --tier basic,advanced \
  --cuda-major 13 --output bench_output.json
```

Use committed JSON as the before measurements; do not regenerate existing
baselines in pre-change CI. Missing rows and candidate baseline updates follow
the import workflow below.

The default result is `bench_output.csv`. Use JSON when the result will be
compared with or imported into committed baselines:

```bash
python3 run_bench.py --operator resize --cuda-major 13 --output bench_output.json
```

Non-local `--run-context` values require `--cuda-major`. When the reported GPU
model is present in `config/sku_map.json`, these contexts require
its configured SM clock to be requested successfully and checked under device
load immediately before every timed state. This is a pre-state gate rather than
an in-kernel measurement of the timed operation. An unmapped GPU model retains
the local, best-effort clock policy, but its results cannot be compared with a
committed SKU baseline until the model is added to the map.

Local and unmapped-GPU runs record `sm_clock_device_verified: false` because an
accepted lock request is not device verification. The stability analysis
rejects those artifacts, and `update_baseline.py` applies that analysis when
`--stable-only` is used. Use that option when updating committed baselines.
Artifacts created before device verification was recorded are intentionally
rejected and must be recollected for `--stable-only` updates.

`CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ` is the runner-to-driver enforcement
contract. C++ and Python benchmark drivers that inherit it hard-exit when the
value is invalid or the active clock cannot be verified. `run_bench.py` scopes
the value to benchmark child processes that require strict verification.
The runner interprets the driver protocol as follows:

- `SM clock verified:` accepts a successful run.
- `SM clock mismatch:` retries after reapplying the requested clock, then exits
  `3` if the clock does not stabilize.
- `SM clock device error:` exits `3` immediately so CI can exclude the node.
- `SM clock verification error:` exits `1` for a driver or configuration error.
- A successful child that omits `SM clock verified:` exits `1`, unless its
  scratch output confirms that every benchmark state was skipped.

The C++ and Python drivers use the same outcome mapping.

The other clock-control environment variables are:

- `BENCH_LOCK_SM_CLOCK_MHZ`: ordered lock preferences used only on the
  best-effort path (local runs and unmapped GPU models in any run context).
  The SKU preflight ignores this setting.
- `BENCH_CLOCK_SAMPLE_INTERVAL_S`: live-diagnostic sampling interval.
- `BENCH_CLOCK_LOG`: override for the JSONL diagnostics path.

The runner fails noisy results. When both languages run, it also checks that C++
and Python timings agree within the configured relative and absolute thresholds.
`--skip-validation` retains row statuses but prevents those checks from failing
the final run.

### Retain native measurement evidence

Pass `--raw-output-dir /path/to/evidence` to retain each actual C++ or Python
benchmark subprocess in its own `attempt-*` directory. This includes native CSV,
JSONBIN and per-sample binary files, stdout/stderr, invocation metadata, and clock
outcomes. Failed processes preserve whatever output they produced; successful
processes must produce complete native output or the capture fails visibly.

Capture is off by default. It does not change selected workloads, validation
thresholds, warmup, or the requested retry policy. Existing retries are retained
as separate attempts rather than overwriting earlier evidence. Evidence is not
automatically imported into baselines, and capture adds serialization and storage
overhead outside the benchmark's timed region.

Native sample filenames in JSON may be absolute. When moving evidence, preserve
the original JSON and resolve filenames under the corresponding captured attempt
directory in analysis code.

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
validation failure, `2` for invalid arguments or configuration, and `3` when
the node cannot satisfy or hold its baseline-backed clock profile and should be
retried elsewhere.

## Measure Python Binding Overhead

`binding_bench.py` measures host-side Python binding costs with
`time.perf_counter_ns()`. It complements the CUDA-event operator benchmarks,
which intentionally include GPU execution and cannot resolve sub-microsecond
wrapper costs.

The benchmark preallocates persistent inputs, warms reusable caches, and keeps
blocking completion waits and window-boundary stream synchronization outside the
timed regions. Synchronization commands intrinsic to the API call, such as DLPack
cross-stream dependency setup, remain included. Returned Python references are
immediately discarded, so their decref is included. In the reshape and wrapping
cases, the next timed call also includes the previous wrapper cleanup. Tensor
construction measures warmed allocation-cache reuse. Submission-resource
reclamation occurs at untimed chunk boundaries instead.
The composed wrap-and-submit workflows use the explicit release drain when
available, without an extra untimed submission before each chunk. Older wheels
retain that priming submission to clear legacy callback closures outside timing.
Submission windows are split into at most 64 calls so the bounded pending-release
queue does not turn a host-overhead measurement into a GPU-backpressure
measurement. Cases cover tensor construction, reshape, and CAI/DLPack
``as_tensor`` wrapping with string and ``TensorLayout`` arguments; a composed
wrap-and-submit workflow at serial and queued depth; fixed-tensor and VarShape
submission at three and five retained resources; batch-1 and configurable
(default 256) VarShape endpoints; ResourceGuard retention; repeated
wrapped-image export; legacy and versioned DLPack capsule lifetimes;
cross-stream DLPack synchronization; and the empty release-drain path.
A Python no-op case records the host noise floor. The DLPack-only input delegates
both protocol calls through fixed Python frames to hide CAI, so compare that row
with the same harness rather than treating it as the absolute cost of a
third-party producer.

The separate `export-metadata` suite measures a fresh owned Tensor/Image
`.cuda()` export and its CAI metadata on every call. It covers packed and padded
rows, scalar integer/float/complex tensors, and single-plane, planar RGB, and
NV12 images. The allocation stays live outside timing; exported buffers and
metadata are recreated and released inside timing. This also exercises lazy
metadata creation on free-threaded Python. It does not measure GPU execution or
Torch conversion. Use the same suite for both builds:

```bash
python3 bench/binding_bench.py --suite export-metadata --output exports.json
```

```bash
python3 bench/binding_bench.py --output python_binding_microbenchmark.json

# Run a focused subset while iterating.
python3 bench/binding_bench.py \
  --case submission.flip.tensor.explicit \
  --case interop.image.cuda.wrapped_writer \
  --output focused_binding_microbenchmark.json
```

The JSON contains every raw timing window plus median, mean, sample standard
deviation, p90, median absolute deviation, and robust noise. Synchronization and
setup failures are fatal, but timing regressions are initially diagnostic: use
same-host artifacts to establish the noise floor before choosing a gate.

Dedicated CI comparisons halve the export-metadata iteration count while
retaining all cases, 20 timed windows, five warmup windows and five fresh processes
per wheel. Standalone defaults are unchanged. See the [CI sampling settings](../ci/README.md#python-binding-speed-comparison)
for the exact per-suite overrides; reference and candidate always use the same
settings, and the comparator rejects mismatched measurement parameters.

Each case also retains `thread_cpu_window_ns`, summed over the timed chunks using
`time.thread_time_ns()`. These diagnostic CPU intervals surround the wall-time
intervals and include clock overhead, which can be significant for very small
chunks. Explicit synchronization waits are excluded from both. Use these values
to investigate noisy windows; CPU time does not replace or normalize the wall-time
regression metric, and small wall-minus-CPU differences need not be positive.

Exit status is `0` after a complete measurement, `1` for a setup or execution
failure, and `2` for invalid arguments.

### Compare Binding Measurements

Run the complete harness against a reference and candidate wheel in separate
clean environments on the same host, with the same Python build, GIL mode,
GPU, CPU affinity, and arguments. Set `CVCUDA_BINDING_BENCH_IMAGE` to the runtime
image identity and `CVCUDA_BINDING_BENCH_CUDA_MAJOR` to `12` or `13` for both
measurements. Then compare the reports from the source tree:

```bash
python3 bench/compare_binding_bench.py \
  --reference reference.json --candidate candidate.json --junit comparison.xml
```

For a single report per wheel, the comparator recomputes medians and robust
noise from raw timing windows. A binding case more than 10% slower fails;
improvements pass. Noise above 5%, incomplete reports, or incompatible
environments make the comparison invalid. The Python no-op is diagnostic and
is not subtracted. A quiet individual run does not rule out drift between the
two measurements. Use the fixed five-process comparison for shared CI workers.

For a fixed five-process comparison, run five fresh processes per wheel in
reference/candidate order, alternating between the two environments. Supply all
five reports for each role:

```bash
python3 bench/compare_binding_bench.py \
  --reference reference-{1..5}.json --candidate candidate-{1..5}.json \
  --junit comparison.xml
```

By default, each process discards five warm-up windows and records 20 timed
windows per case. Each window batches repeated calls and reports elapsed time
divided by its call count. The median of those 20 window averages is that
process's observation. Each case
compares the arithmetic mean of its five process medians. It reports
a regression only when the candidate is more than 10% slower **and** the mean
difference exceeds six standard errors:

```text
candidate_mean > 1.10 * reference_mean
candidate_mean - reference_mean > 6 * sqrt(candidate_variance / 5 + reference_variance / 5)
```

The variances are sample variances of the five process medians. The fixed
six-standard-error guard accounts for process variation without adding a
statistics dependency. It is conservative under the independent-sample Welch
approximation; shared-worker drift can violate that assumption. It is not an
empirical false-alarm guarantee.
Each complete suite is compared separately and contains at most 25 non-control
cases. The statistical family bound is per suite, not combined across suites or
CUDA channels. Partial suites and mixed-suite comparisons are invalid. The CI
wheel runner runs both suites, retains separate reports/JUnit, and fails if
either comparison fails; the slowdown and noise thresholds are identical.

Within-process and between-process robust noise remain diagnostics in this
mode. Every process must have complete, compatible measurements and the same
wheel identity within its role. Counts other than one or five per role are
invalid. Raw windows remain in their original reports; they are not pooled as
independent samples, and no run is replaced or discarded.

A passing comparison means no supported regression was detected, not that the
wheels have equivalent performance. High process variance reduces sensitivity:
a 10% slowdown may not trigger, and noisier cases can require a substantially
larger change. Inspect the retained process medians when investigating a pass
with a large measured slowdown.

Exit status is `0` for a valid passing comparison, `1` for a regression, and
`2` for invalid measurements. JUnit distinguishes regressions from invalid
measurements. There are no automatic timing retries.

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

- `A100_PCIE_40GB_250W_1005MHz`
- `H100_PCIe_350W_1005MHz`
- `A100_PCIE_40GB_250W_1005MHz_CUDA12`
- `H100_PCIe_350W_1005MHz_CUDA12`

First create a JSON result, then compare it from the repository root:

```bash
cd build-rel/bin
python3 run_bench.py --cuda-major 13 --output bench_output.json
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
Resolved SKU: A100_PCIE_40GB_250W_1005MHz (current JSON SKU key)
matched=720 regressions=0 improvements=0 missing=0 new=0 missing_sku=0

- thresholds: regression=0.1, improvement=0.1
- all-rows |Delta|: median 0.30%, max 5.61%

## Regressions (0)
_none_

## Unexpected improvements (0)
_none_

```

The report always prints to the console, so a CI job log names the rows it
failed on rather than only counting them. It can also be written as Markdown or
JUnit; the printed copy then caps each section at 25 rows, while the file keeps
every row:

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
│   └── sku_map.json             # GPU profiles and their required clock targets
├── cpp/ops/                     # C++ operator benchmarks
├── python/ops/                  # Python operator benchmarks
├── _internal/                   # Shared helpers and baseline maintenance
├── run_bench.py                 # Run C++ and/or Python benchmarks
├── compare_wheels.py            # Compare two Python wheels
├── compare_to_baseline.py       # Compare a run with committed baselines
├── binding_bench.py             # Measure Python binding host overhead
└── compare_binding_bench.py     # Compare paired binding measurements
```

The five top-level scripts are the supported benchmark commands. Files under
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

Each `config/sku_map.json` entry routes a device name, power cap, locked SM
clock, and CUDA major to a baseline stem. A `(gpu_name, power_cap_w,
cuda_major)` profile must declare exactly one locked SM clock; CUDA 12 and CUDA
13 profiles may declare different targets, but every `(gpu_name, power_cap_w)`
profile must include both CUDA majors. Strict SKU-map loading rejects multiple
clock targets for one profile, so validation, preflight, and benchmark execution
cannot choose different targets.

An explicit `"power_cap_w": null` makes a profile power-independent. H100 PCIe
uses this policy for both CUDA majors: 310W and 350W boards share the same
1005MHz baseline target, and power remains recorded in measurement metadata.
The existing `H100_PCIe_350W_1005MHz` stems are retained as historical identifiers,
not power requirements. A100 retains its 250W requirement. Wildcard and fixed
power profiles cannot overlap for the same model and CUDA major. Regenerate and
check mixed-power measurements for stability before importing new baselines;
power-independent routing does not waive clock verification or stability gates.

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
          "H100_PCIe_350W_1005MHz": {
            "n_runs": 5,
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
  },
  "baseline_refresh_incomplete": true
}
```

### Field Reference

| Field | Description |
|---|---|
| Manifest `config` | Per-operator JSON file under `config/operators/`. |
| Manifest `cpp` / `python` | Matching C++ binary and Python benchmark script. |
| `benchmark` | The `bench_<op>` target for this operator file. |
| `baseline_refresh_incomplete` | Optional importer-owned marker: one or more rows were retained; a complete stable import clears it. |
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

The reference-baseline rule is: **collect ten runs per GPU/CUDA lane, select the
largest stable cluster of at least five matching runs for each case/SKU, and
average that cluster**. C++ and Python use the same selected runs. The cohort
size (`BASELINE_RUNS_PER_SKU = 10`), minimum cluster size
(`BASELINE_MIN_CLUSTER_SIZE = 5`), and range limit are defined in
`bench/_internal/quality.py`; CI and the importer read them from there.

Collection, selection, and acceptance are separate steps:

| Step | Requirement |
|---|---|
| Collect | Exactly ten raw artifacts per configured SKU, from one measured revision and git ref, with matching device identity and verified clocks. Each case must have all ten C++/Python pairs. |
| Select | For each case/SKU, choose the largest group of at least five matching runs where `(max - min) / median * 100 <= 2` separately for C++ and Python. |
| Accept | Check the selected aggregate's noise and C++/Python parity, then reject same-key slowdowns beyond the import threshold (1% by default with `--stable-only`). |

For equal-sized stable clusters, the current tie-breaker chooses the largest
sum of C++ and Python mean times, then uses source names to break an exact tie.
This is a deterministic preference; it does not prove the entire reference pool
is stable or guarantee a conservative baseline for each language. Retain the
raw artifacts and review the node, driver, and vBIOS provenance when timings
form separate groups.

Before importing, preserve and check the complete raw cohort:

```bash
python3 bench/_internal/baseline_stability.py \
  --from /path/to/artifacts \
  --operator resize --clustered \
  --markdown baseline-stability.md \
  --json-out baseline-stability.json
```

The read-only report validates provenance, device-to-SKU identity, and
declared-case coverage for the selected operators. Its Markdown summarizes
selected sample counts and tabulates min/median/mean/max for flagged rows;
`--json-out` records statistics and selected source paths for every row. The
original artifacts retain the excluded measurements. A row is unstable when
no qualifying cluster exists. The report flags an existing baseline as
optimistic when the analyzed mean is more than one percent slower and every
analyzed sample exceeds it; that label is an audit signal, not an import gate.
Under-sampled committed rows flag fewer than five runs for maintenance review;
they remain valid campaign references.
Input or device-identity issues block import. Coverage issues make the
standalone report exit nonzero. A stable-only import reports missing declared
rows and retains their committed values; measured unstable rows are withheld
row-by-row when a committed value exists. A first baseline seed has no value to
retain, so every declared row must be stable; rerun the wave if a new row is
unstable. Retry missing slots and preserve completed measurements. Replace
superseded artifacts when rerunning; do not choose five convenient files before
analysis. A failed comparison job can still upload a complete, usable artifact;
judge its provenance and measurements rather than its job status alone.

Use `--clustered` on both the report and import commands for these ten-run CI
cohorts. Omitting it retains the earlier mode: exactly five runs per SKU, all
five stable. This compatibility mode is not the current CI collection workflow.

After reviewing the stability report, dry-run a stable-only import. This keeps
the existing baseline for an entry when either its C++ or Python timing is
unstable:

```bash
python3 bench/_internal/update_baseline.py --from /path/to/artifacts \
  --operator resize --stable-only --clustered --expect-revision <measured-sha> \
  --dry-run

python3 bench/_internal/update_baseline.py --from /path/to/artifacts \
  --operator resize --stable-only --clustered --expect-revision <measured-sha> \
  --write-summary baseline-update.md
```

The two thresholds answer different questions: the two-percent range checks
agreement within the selected cluster, while the one-percent regression limit
checks its mean against the committed value. A stable cluster can still reveal
an optimistic committed baseline and require an explicitly reviewed reset.

Add `--withhold-regressions` to create a temporary investigation snapshot, not a
merge-ready baseline policy. Before treating a refresh as final, explain and
resolve every withheld slowdown through a replacement measurement, a code fix,
or an explicitly reviewed baseline update. Repeatedly importing improvements
while permanently retaining slowdowns would ratchet the committed baseline
toward optimistic measurements. Any non-dry-run `--stable-only` import requires
`--write-summary` so retained rows and coverage gaps remain auditable. The
stable-only regression threshold defaults to one percent. Put the generated
summary in the MR description or a review discussion; do not commit it. The
importer marks partial refreshes for review; the marker does not block reusing
committed numbers as a campaign reference. `--expect-revision` must match every
artifact's measured revision. For merged-result pipelines, use the merge SHA.
An import-tool or documentation cleanup that leaves measurement behavior
unchanged does not require recollecting those raw measurements; preserve their
original revision identity. The importer exits nonzero when no selected entry
remains, whether every entry was withheld or the input had no matching rows,
because that is a failed regeneration rather than a successful no-op.

Validate the updated baselines, including same-key timing changes against the
branch base. A passing schema/quality validator alone does not prove the refresh
is complete: resolve every retained row, clear partial-refresh markers through
a complete stable import, and check the selected sample counts before declaring
the reference evidence final.

```bash
python3 bench/_internal/validate_baselines.py \
  --operator resize --reject-regressions-from origin/main \
  --max-regression-pct 1
```

Finally, create a fresh JSON run and compare it with the updated baselines, then
run the full MR matrix.

The local `benchmark-baseline-regression` pre-commit hook and GitLab `lint` job
run this one-percent, slowdown-only check for changed operator JSON files.
Improvements and rows without a matching config key, case key, SKU, and language
remain subject to the existing validation but are not timing regressions.

`--allow-regressions` waives only the same-key slowdown check and is reserved
for intentional, reviewed baseline resets. Schema, SKU, noise, C++/Python
parity, and fresh-run comparison checks remain enforced.

## Benchmark Drift Reporting

The committed-baseline gate compares a run against a fixed +/-10% band. Measured
spread on the reference SKUs is far tighter than that -- around 0.6% across a day
and 1.5% within a stable stretch -- so a row can slide several percent and still
pass. A config that moves from sitting at +/-1% to hovering at -5% is invisible
today.

`bench/_internal/bench_drift.py` reads a window of nightly artifacts and reports
rows whose recent level sits outside their own historical spread:

```bash
python3 bench/_internal/bench_drift.py --from nightly_wave --markdown drift.md
```

A row is reported when the recent stretch differs from the earlier one by more
than `--min-shift-pct` *and* by more than `--sigma` multiples of the earlier
stretch's own robust spread. Rows are split into quiet drift and large shifts,
because a large shift is almost always a landed change while a quiet one is the
case this exists to surface.

It is deliberately read-only: it writes no baseline and fails no gate. A shift is
a prompt to look, not a verdict, so a false positive costs a glance rather than a
red pipeline, and the tool cannot alter the reference it measures.

Only nightly runs of the default branch are analysed. A merge request run
measures its own branch, so it says nothing about where the default branch sits.
Rows the current config no longer declares are skipped and counted -- declared
axis values change over time, so any window longer than the last config edit
contains some.


## Resources

- [CV-CUDA documentation](https://docs.nvidia.com/cvcuda/)
- [nvbench documentation](https://github.com/NVIDIA/nvbench)
- [CV-CUDA installation guide](../docs/sphinx/installation.rst)
