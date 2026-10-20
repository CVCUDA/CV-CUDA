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
"""Complete, valid evidence objects shared by the summary and gate tests."""

import copy

CANDIDATE = "2" * 40
DIGEST = "a" * 64
CATEGORY = ["Tensor", "NHWC", "u8"]


def run(exit_code=0, *command):
    return {
        "command": list(command) or ["tool"],
        "exit_code": exit_code,
        "output_sha256": DIGEST,
    }


def tests(commit=CANDIDATE, **overrides):
    record = {
        "commit": commit,
        "required_categories": [CATEGORY],
        "cases": [
            {
                "id": "OpExample/correct_output/0",
                "framework": "gtest",
                "outcome": "passed",
                "reference": "GoldExample",
                "assertions": ["EXPECT_EQ"],
                "categories": [CATEGORY],
            }
        ],
        "gtest": run(0, "cvcuda_test_system"),
    }
    record.update(overrides)
    return record


def attempt(outcome="kept", speedup=1.5, lead="Vectorized loads", commit=CANDIDATE):
    return {
        "lead": lead,
        "outcome": outcome,
        "commit": commit,
        "before_artifact_sha256": DIGEST,
        "after_artifact_sha256": "b" * 64,
        "cases": 4,
        "median_speedup": speedup,
        "min_speedup": speedup,
        "max_speedup": speedup,
    }


def complete(commit=CANDIDATE):
    """Evidence that satisfies every checklist gate."""

    return {
        "schema": 1,
        "tests": tests(commit),
        "profile": {
            "hot_kernel": "ExampleKernel",
            "hot_share_pct": 97.0,
            "memory_pct": 93.0,
            "compute_pct": 21.0,
            "report_sha256": DIGEST,
            "findings": [{"rule": "UncoalescedGlobalAccess", "speedup_pct": 40.0}],
        },
        "attempts": [attempt(commit=commit)],
        "memory": {
            "workload": "bench_example --axis shape=1x64x64",
            "before_report_sha256": DIGEST,
            "after_report_sha256": "b" * 64,
            "sites": [
                {
                    "site": "cvcuda::priv::Example::operator()",
                    "before_peak_bytes": 1024,
                    "after_peak_bytes": 1024,
                }
            ],
            "before_unattributed_bytes": 0,
            "after_unattributed_bytes": 0,
        },
        "review": {
            "commit": commit,
            "assess": run(0, "refactor_op.py"),
            "assess_verdict": "PASS",
            "applied": False,
        },
        "baselines": {
            "base": "origin/main",
            "import": {
                "dry_run": run(0, "update_baseline.py", "--dry-run"),
                "run": run(0, "update_baseline.py"),
                "measured_commit": commit,
                "summary_sha256": DIGEST,
                "files": {"bench/config/operators/example.json": DIGEST},
            },
            "validation": run(0, "validate_baselines.py"),
        },
    }


def with_(evidence, **sections):
    result = copy.deepcopy(evidence)
    for key, value in sections.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result
