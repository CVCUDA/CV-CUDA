#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Compare paired binding_bench.py reports (0: pass, 1: slowdown, 2: invalid)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

from binding_bench import SUITES, summarize

# For five observations per role, Welch degrees of freedom are at least four.
# Six exceeds t(df=4, 1-0.05/25)=5.9514 for each suite of at most 25 cases.
# This is a per-suite bound, not a combined bound across suites or CUDA channels.
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm
SIGNIFICANCE_MULTIPLIER = 6.0

MATCH_METADATA = (
    "benchmark_script_sha256",
    "python_version",
    "python_build",
    "gil_enabled",
    "runtime_image",
    "worker_hostname",
    "cpu_model",
    "cpu_affinity",
    "gpu_inventory",
    "cuda_major",
    "cuda_visible_devices",
    "nvidia_visible_devices",
)


def limit(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("thresholds must be finite and nonnegative")
    return number


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _report(report):
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported binding report schema")
    parameters = report["parameters"]
    cases = parameters["selected_cases"]
    if cases not in [list(suite) for suite in SUITES.values()]:
        raise ValueError("report must select one complete binding suite in order")
    samples = _integer(parameters["samples"], "samples", 2)
    _integer(parameters["warmups"], "warmups")
    metadata = report["metadata"]
    for key in MATCH_METADATA:
        value = metadata[key]
        if key in ("cuda_visible_devices", "nvidia_visible_devices"):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"invalid {key}")
        elif key == "gil_enabled":
            if type(value) is not bool:
                raise ValueError("gil_enabled must be a boolean")
        elif key == "cpu_affinity":
            if not isinstance(value, list) or len(value) != 1:
                raise ValueError("benchmark must be pinned to one CPU")
            _integer(value[0], "CPU", 0)
        elif key == "gpu_inventory":
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item for item in value)
            ):
                raise ValueError("missing GPU inventory")
        elif not isinstance(value, str) or not value or value in ("unknown", "null"):
            raise ValueError(f"missing {key}")
    if metadata["cuda_major"] not in ("12", "13"):
        raise ValueError("unsupported CUDA channel")

    rows = report["results"]
    if len(rows) != len(cases) or {row["name"] for row in rows} != set(cases):
        raise ValueError("missing, duplicate, or unexpected binding cases")
    result = {}
    for row in rows:
        name = row["name"]
        iterations = _integer(row["iterations_per_window"], f"{name} iterations")
        windows = row["window_ns"]
        if row["samples"] != samples or len(windows) != samples:
            raise ValueError(f"{name}: inconsistent sample count")
        for duration in windows:
            _integer(duration, f"{name} duration")
        stats = summarize(windows, iterations)
        if not all(math.isfinite(stats[key]) for key in ("p50_ns", "robust_noise_pct")):
            raise ValueError(f"{name}: nonfinite measurement")
        if stats["p50_ns"] <= 0:
            raise ValueError(f"{name}: nonpositive median")
        result[name] = stats
    return result


def _compatible(reference, candidate):
    if reference["parameters"] != candidate["parameters"]:
        raise ValueError("measurement parameters differ")
    for key in MATCH_METADATA:
        if reference["metadata"][key] != candidate["metadata"][key]:
            raise ValueError(f"incompatible metadata: {key}")


def _aggregate(reports):
    measurements = [_report(report) for report in reports]
    first = reports[0]
    timestamps = set()
    for report in reports:
        _compatible(first, report)
        if len(reports) > 1:
            timestamp = report["metadata"].get("timestamp_utc")
            if not isinstance(timestamp, str) or not timestamp:
                raise ValueError("missing measurement timestamp")
            # Copies of one report are not independent process observations.
            if timestamp in timestamps:
                raise ValueError("duplicate measurement within a wheel")
            timestamps.add(timestamp)
            for key in ("cvcuda_version", "cvcuda_extension_sha256"):
                value = report["metadata"].get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"missing repeated-wheel identity: {key}")
                if value != first["metadata"].get(key):
                    raise ValueError(f"wheel identity differs between processes: {key}")
    result = {}
    for name in first["parameters"]["selected_cases"]:
        runs = [measurement[name] for measurement in measurements]
        iterations = runs[0]["iterations_per_window"]
        if any(run["iterations_per_window"] != iterations for run in runs):
            raise ValueError(f"{name}: iteration counts differ")
        medians = [run["p50_ns"] for run in runs]
        median = statistics.median(medians)
        mad = statistics.median(abs(value - median) for value in medians)
        between_noise = 100.0 * 1.4826 * mad / median
        result[name] = {
            "iterations_per_window": iterations,
            "latency_ns": statistics.mean(medians),
            "variance_of_mean_ns2": (
                statistics.variance(medians) / len(medians) if len(medians) > 1 else 0
            ),
            # Retain dispersion as a diagnostic for repeated measurements.
            "robust_noise_pct": max(
                between_noise, *(run["robust_noise_pct"] for run in runs)
            ),
        }
    return result


def compare(reference, candidate, threshold=10.0, max_noise=5.0):
    threshold, max_noise = limit(threshold), limit(max_noise)
    references = reference if isinstance(reference, list) else [reference]
    candidates = candidate if isinstance(candidate, list) else [candidate]
    if len(references) != len(candidates) or len(references) not in (1, 5):
        raise ValueError("expected one or five reports per wheel, with equal counts")
    baseline, current = _aggregate(references), _aggregate(candidates)
    _compatible(references[0], candidates[0])
    results = []
    for name in baseline:
        before, after = baseline[name], current[name]
        if before["iterations_per_window"] != after["iterations_per_window"]:
            raise ValueError(f"{name}: iteration counts differ")
        delta = 100.0 * (after["latency_ns"] / before["latency_ns"] - 1.0)
        if not math.isfinite(delta):
            raise ValueError(f"{name}: nonfinite slowdown")
        noise = max(before["robust_noise_pct"], after["robust_noise_pct"])
        margin = SIGNIFICANCE_MULTIPLIER * math.sqrt(
            before["variance_of_mean_ns2"] + after["variance_of_mean_ns2"]
        )
        if len(references) == 1 and noise > max_noise:
            status = "INVALID (noise)"
        elif name == "python.noop":
            status = "CONTROL"
        elif delta > threshold + 1e-9 and (
            len(references) == 1 or after["latency_ns"] - before["latency_ns"] > margin
        ):
            status = "REGRESSION"
        else:
            status = "PASS"
        results.append(
            (name, before["latency_ns"], after["latency_ns"], delta, noise, status)
        )
    return results


def write_junit(path, rows, error=None):
    failures = sum(row[-1] == "REGRESSION" for row in rows)
    errors = sum(row[-1].startswith("INVALID") for row in rows) + bool(error)
    suite = ET.Element(
        "testsuite",
        name="python-binding-speed",
        tests=str(len(rows) + bool(error)),
        failures=str(failures),
        errors=str(errors),
    )
    for name, before, after, delta, noise, status in rows:
        case = ET.SubElement(suite, "testcase", name=name, classname="binding_bench")
        detail = (
            f"{before:.2f} -> {after:.2f} ns/call; {delta:+.2f}%; noise {noise:.2f}%"
        )
        ET.SubElement(case, "system-out").text = detail
        if status == "REGRESSION" or status.startswith("INVALID"):
            tag = "failure" if status == "REGRESSION" else "error"
            ET.SubElement(case, tag, message=f"{status}: {detail}")
    if error:
        case = ET.SubElement(suite, "testcase", name="measurement-validity")
        ET.SubElement(case, "error", message=error)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  compare_binding_bench.py --reference reference.json --candidate candidate.json --junit comparison.xml

Exit status:
  0: valid comparison passes; 1: binding slowdown; 2: invalid or noisy measurements.
""",
    )
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--regression-threshold-pct", type=float, default=10.0)
    parser.add_argument(
        "--max-noise-pct",
        type=float,
        default=5.0,
        help="Noise limit for single-report comparisons; diagnostic only with five reports.",
    )
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    rows, error = [], None
    try:
        for paths in (args.reference, args.candidate):
            if len({path.resolve() for path in paths}) != len(paths):
                raise ValueError("duplicate report path within a wheel")
        references = [json.loads(path.read_text()) for path in args.reference]
        candidates = [json.loads(path.read_text()) for path in args.candidate]
        rows = compare(
            references,
            candidates,
            args.regression_threshold_pct,
            args.max_noise_pct,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as exc:
        error = f"Invalid binding comparison: {exc}"
        print(error, file=sys.stderr)
    if rows:
        print(
            f"Processes per wheel: {len(args.reference)}; comparing means of process medians"
        )
        print(
            f"{'case':58} {'reference ns':>13} {'candidate ns':>13} {'change':>9} {'noise':>8} status"
        )
        for name, before, after, delta, noise, status in rows:
            print(
                f"{name:58} {before:13.2f} {after:13.2f} {delta:+8.2f}% {noise:7.2f}% {status}"
            )
        if len(references) == 5:
            print(
                f"Five-process rule: >{args.regression_threshold_pct:g}% slowdown "
                "and difference >6 standard errors; noise is diagnostic."
            )
            for label, reports in (
                ("reference", references),
                ("candidate", candidates),
            ):
                processes = [_report(report) for report in reports]
                for name in reports[0]["parameters"]["selected_cases"]:
                    medians = ", ".join(
                        f"{process[name]['p50_ns']:.2f}" for process in processes
                    )
                    print(f"{label} {name}: process medians [{medians}] ns/call")
    if args.junit:
        write_junit(args.junit, rows, error)
    if error or any(row[-1].startswith("INVALID") for row in rows):
        return 2
    return int(any(row[-1] == "REGRESSION" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
