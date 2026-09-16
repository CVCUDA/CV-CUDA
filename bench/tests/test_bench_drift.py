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

"""Unit tests for the read-only benchmark drift report."""

from __future__ import annotations

import json
import random

import pytest


from _internal import bench_drift
from _internal.baselines import build_baseline_document, case_key_from_axis_values

SKU = "TESTSKU"
CONFIG_KEY = "resize_basic"
BASE_US = 1000.0


def _case():
    return case_key_from_axis_values(
        CONFIG_KEY,
        [("InOutDataType", "uint8"), ("shape", "1x2x3"), ("inputKind", "Tensor")],
    )


def _series(values):
    """Feed raw numbers through the analyser the way a wave would arrive."""
    split = max(bench_drift.DEFAULT_MIN_SAMPLES, int(len(values) * 0.7))
    return bench_drift.analyse_row(values, split=split)


# ---------------------------------------------------------------------------
# The gap the report exists to close: the committed band is +/-10%, but a row's
# own spread is around 1%, so a few percent of drift is invisible today.
# ---------------------------------------------------------------------------


def test_a_quiet_drift_inside_the_committed_band_is_reported():
    rng = random.Random(7)
    steady = [BASE_US * (1 + rng.uniform(-0.005, 0.005)) for _ in range(30)]
    drifted = [BASE_US * 1.05 * (1 + rng.uniform(-0.005, 0.005)) for _ in range(15)]

    outcome = _series(steady + drifted)

    assert outcome is not None
    baseline, recent, shift_pct, spread_pct, sigmas = outcome
    # Well inside the +/-10% gate, so CI would say nothing about it.
    assert 4.0 < shift_pct < 6.0
    assert spread_pct < 1.0
    assert sigmas > bench_drift.DEFAULT_SIGMA


def test_a_steady_row_is_not_reported():
    rng = random.Random(11)
    values = [BASE_US * (1 + rng.uniform(-0.01, 0.01)) for _ in range(45)]

    assert _series(values) is None


def test_a_noisy_row_needs_a_bigger_shift_to_register():
    """A row that is genuinely noisy must not cry wolf on ordinary variation."""
    rng = random.Random(13)
    noisy = [BASE_US * (1 + rng.uniform(-0.12, 0.12)) for _ in range(30)]
    slightly_up = [BASE_US * 1.03 * (1 + rng.uniform(-0.12, 0.12)) for _ in range(15)]

    assert _series(noisy + slightly_up) is None


def test_a_shift_below_the_floor_is_ignored_however_tight_the_row():
    """A perfectly repeatable row must not report a fraction of a percent."""
    steady = [BASE_US] * 30
    nudged = [BASE_US * 1.005] * 15

    assert _series(steady + nudged) is None


def test_a_speedup_is_reported_as_a_negative_shift():
    rng = random.Random(17)
    before = [BASE_US * (1 + rng.uniform(-0.005, 0.005)) for _ in range(30)]
    after = [BASE_US * 0.90 * (1 + rng.uniform(-0.005, 0.005)) for _ in range(15)]

    outcome = _series(before + after)

    assert outcome is not None
    assert outcome[2] < -9.0


def test_too_little_history_reports_nothing():
    assert _series([BASE_US] * 6 + [BASE_US * 1.2] * 6) is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n")


def _write_config_tree(root):
    _write_json(
        root / "sku_map.json",
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 350,
                    "locked_sm_clock_mhz": 1095,
                    "cuda_major": 13,
                    "stem": SKU,
                }
            ]
        },
    )
    _write_json(
        root / "operators" / "resize.json",
        {
            "benchmark": "resize",
            "configs": {
                CONFIG_KEY: {
                    "tier": "basic",
                    "dtypes": ["uint8"],
                    "string_axes": {"shape": ["1x2x3"], "inputKind": ["Tensor"]},
                    "baselines": {
                        _case(): {
                            SKU: {
                                "n_runs": 5,
                                "gpu_time_us_cpp": BASE_US,
                                "gpu_time_us_python": BASE_US * 1.05,
                                "gpu_noise_us_cpp": 1.0,
                                "gpu_noise_us_python": 1.0,
                                "gpu_bwutil_cpp": 0.42,
                                "gpu_bwutil_python": 0.41,
                            }
                        }
                    },
                }
            },
        },
    )


def _wave(tmp_path, values, *, nightly=True):
    directory = tmp_path / "wave"
    directory.mkdir(parents=True, exist_ok=True)
    for idx, cpp in enumerate(values):
        _write_json(
            directory / f"run{idx:03d}.json",
            build_baseline_document(
                {
                    _case(): {
                        SKU: {
                            "n_runs": 1,
                            "gpu_time_us_cpp": cpp,
                            "gpu_time_us_python": cpp * 1.05,
                            "gpu_noise_us_cpp": 1.0,
                            "gpu_noise_us_python": 1.0,
                            "gpu_bwutil_cpp": 0.42,
                            "gpu_bwutil_python": 0.41,
                        }
                    }
                },
                run_metadata={
                    "run_context": "nightly" if nightly else "mr",
                    "is_nightly": nightly,
                    "git_ref": "main",
                    "timestamp_utc": f"2026-08-01T{idx % 24:02d}:00:00Z",
                },
            ),
        )
    return directory


def test_cli_reports_drift_and_still_succeeds(tmp_path, capsys):
    """A shift is a prompt to look, never a failing gate."""
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = _wave(tmp_path, [BASE_US] * 30 + [BASE_US * 1.05] * 15)

    rc = bench_drift.main(["--from", str(wave), "--config-dir", str(config_dir)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Quiet drift" in out
    assert "+5.0" in out


def test_cli_is_quiet_when_nothing_moved(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = _wave(tmp_path, [BASE_US] * 45)

    rc = bench_drift.main(["--from", str(wave), "--config-dir", str(config_dir)])

    assert rc == 0
    assert "No row moved beyond its own spread." in capsys.readouterr().out


def test_cli_ignores_non_nightly_artifacts(tmp_path, capsys):
    """A merge request run measures its own branch, not the default branch."""
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = _wave(tmp_path, [BASE_US] * 30 + [BASE_US * 1.05] * 15, nightly=False)

    rc = bench_drift.main(["--from", str(wave), "--config-dir", str(config_dir)])

    assert rc == 1
    assert "not nightly runs" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--sigma", "0"),
        ("--min-shift-pct", "0"),
        ("--min-samples", "0"),
        ("--recent-fraction", "1"),
    ],
)
def test_cli_refuses_threshold_values_the_analysis_cannot_use(
    tmp_path, capsys, flag, value
):
    """A zero divisor should be refused up front, not raised mid-wave."""
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = _wave(tmp_path, [BASE_US] * 45)

    rc = bench_drift.main(
        ["--from", str(wave), "--config-dir", str(config_dir), flag, value]
    )

    assert rc == 1
    assert flag in capsys.readouterr().err


def test_cli_creates_nested_output_directories(tmp_path):
    """A CI job routinely writes into a directory it also creates."""
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = _wave(tmp_path, [BASE_US] * 30 + [BASE_US * 1.05] * 15)
    nested = tmp_path / "reports" / "nightly" / "drift.md"
    nested_json = tmp_path / "reports" / "nightly" / "drift.json"

    rc = bench_drift.main(
        [
            "--from",
            str(wave),
            "--config-dir",
            str(config_dir),
            "--markdown",
            str(nested),
            "--json-out",
            str(nested_json),
        ]
    )

    assert rc == 0
    assert "# Benchmark drift report" in nested.read_text()
    assert nested_json.read_text().lstrip().startswith("[")


def test_cli_writes_a_markdown_report(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = _wave(tmp_path, [BASE_US] * 30 + [BASE_US * 1.05] * 15)
    md = tmp_path / "drift.md"

    bench_drift.main(
        ["--from", str(wave), "--config-dir", str(config_dir), "--markdown", str(md)]
    )

    assert "# Benchmark drift report" in md.read_text()
