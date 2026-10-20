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

"""Tests for the pre-import baseline stability check."""

import json
import os
import subprocess
import sys

import pytest

from _internal import baseline_stability
from _internal.baselines import (
    BaselineError,
    build_baseline_document,
    case_key_from_axis_values,
    collect_artifact_jsons,
)

SKU = "TEST_GPU"
SECOND_SKU = "TEST_GPU_2"
CONFIG = "resize_basic"
SHA = "a" * 40


def test_internal_command_help_is_self_contained(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, baseline_stability.__file__, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--from JSON_OR_DIR" in result.stdout
    assert "Exit status:" in result.stdout


def case_key(shape="1x8x8"):
    return case_key_from_axis_values(
        CONFIG, [("InOutDataType", "uint8"), ("shape", shape)]
    )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_config(root, *, shapes=("1x8x8",), baseline=100.0, n_runs=5):
    write_json(
        root / "sku_map.json",
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 300,
                    "locked_sm_clock_mhz": 1000,
                    "cuda_major": 13,
                    "stem": SKU,
                }
            ]
        },
    )
    write_json(
        root / "operators" / "resize.json",
        {
            "benchmark": "resize",
            "configs": {
                CONFIG: {
                    "tier": "basic",
                    "dtypes": ["uint8"],
                    "string_axes": {"shape": list(shapes)},
                    "baselines": {
                        case_key(): {
                            SKU: {
                                "n_runs": n_runs,
                                "gpu_time_us_cpp": baseline,
                                "gpu_time_us_python": baseline + 1,
                                "gpu_noise_us_cpp": 0.1,
                                "gpu_noise_us_python": 0.1,
                                "gpu_bwutil_cpp": 0.5,
                                "gpu_bwutil_python": 0.5,
                            }
                        }
                    },
                }
            },
        },
    )


def add_second_sku(root):
    path = root / "sku_map.json"
    payload = json.loads(path.read_text())
    payload["entries"].append(
        {
            "gpu_name": "NVIDIA Test GPU 2",
            "power_cap_w": 250,
            "locked_sm_clock_mhz": 900,
            "cuda_major": 13,
            "stem": SECOND_SKU,
        }
    )
    write_json(path, payload)


def write_wave(root, values, *, include_second_case=False, n_runs=1):
    paths = []
    for index, cpp in enumerate(values, 1):
        baselines = {
            case_key(): {
                SKU: {
                    "n_runs": n_runs,
                    "gpu_time_us_cpp": cpp,
                    "gpu_time_us_python": cpp + 1,
                    "gpu_noise_us_cpp": 0.1,
                    "gpu_noise_us_python": 0.1,
                    "gpu_bwutil_cpp": 0.5,
                    "gpu_bwutil_python": 0.5,
                }
            }
        }
        if include_second_case:
            baselines[case_key("1x16x16")] = baselines[case_key()]
        path = root / f"bench-burnin-{index}-{SHA[:8]}.json"
        write_json(
            path,
            build_baseline_document(
                baselines,
                run_metadata={
                    "git_sha": SHA,
                    "git_ref": "test/baselines",
                    "run_context": "mr",
                    "cuda_major": 13,
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 300,
                    "locked_sm_clock_mhz": 1000,
                    "sm_clock_device_verified": True,
                    "vbios_version": "1.0",
                    "driver_version": "600.0",
                    "k8s_node_name": f"node-{index}",
                },
            ),
        )
        paths.append(path)
    return paths


def rewrite_wave_as_second_sku(paths):
    for index, path in enumerate(paths, 1):
        payload = json.loads(path.read_text())
        for case_payload in payload["baselines"].values():
            case_payload[SECOND_SKU] = case_payload.pop(SKU)
        metadata = payload["run_metadata"]
        metadata.update(
            {
                "gpu_name": "NVIDIA Test GPU 2",
                "power_cap_w": 250,
                "locked_sm_clock_mhz": 900,
                "k8s_node_name": f"second-node-{index}",
            }
        )
        write_json(path, payload)


def cpp_row(report):
    return next(row for row in report.rows if row.language == "cpp")


def python_row(report):
    return next(row for row in report.rows if row.language == "python")


def test_stable_wave_flags_an_optimistic_under_sampled_baseline(tmp_path):
    config = tmp_path / "config"
    write_config(config, baseline=98.0, n_runs=2)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])

    report = baseline_stability.analyse(paths, config_dir=config)
    row = cpp_row(report)
    py_row = python_row(report)

    assert report.issues == []
    assert len(report.rows) == 2
    assert row.mean_us == pytest.approx(100.2)
    assert row.committed_us == 98.0
    assert row.stable and row.optimistic and row.under_sampled
    assert py_row.mean_us == pytest.approx(101.2)
    assert py_row.committed_us == 99.0
    assert py_row.stable and py_row.optimistic and py_row.under_sampled
    assert len(report.under_sampled_rows) == 1
    assert "under-sampled committed rows: 1" in baseline_stability.format_report(report)


@pytest.mark.parametrize("bad_n_runs", [None, True])
def test_committed_invalid_run_count_is_under_sampled(tmp_path, bad_n_runs):
    config = tmp_path / "config"
    write_config(config)
    config_path = config / "operators" / "resize.json"
    payload = json.loads(config_path.read_text())
    metrics = payload["configs"][CONFIG]["baselines"][case_key()][SKU]
    if bad_n_runs is None:
        metrics.pop("n_runs")
    else:
        metrics["n_runs"] = bad_n_runs
    write_json(config_path, payload)

    row = cpp_row(
        baseline_stability.analyse(
            write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4]),
            config_dir=config,
        )
    )

    assert row.committed_n_runs is None
    assert row.under_sampled


def test_artifact_gpu_name_whitespace_is_normalized(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    payload = json.loads(paths[0].read_text())
    payload["run_metadata"]["gpu_name"] = " NVIDIA Test GPU "
    write_json(paths[0], payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert report.issues == []
    assert report.artifact_counts_by_sku == {SKU: 5}


def test_power_independent_profile_accepts_mixed_power_wave(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    map_path = config / "sku_map.json"
    sku_map = json.loads(map_path.read_text())
    sku_map["entries"][0]["power_cap_w"] = None
    write_json(map_path, sku_map)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    for index, path in enumerate(paths):
        payload = json.loads(path.read_text())
        payload["run_metadata"]["power_cap_w"] = 310 if index % 2 else 350
        write_json(path, payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert report.issues == []
    assert report.artifact_counts_by_sku == {SKU: 5}


def test_unstable_wave_still_flags_an_optimistic_old_baseline(tmp_path):
    config = tmp_path / "config"
    write_config(config, baseline=98.0)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 110])

    row = cpp_row(baseline_stability.analyse(paths, config_dir=config))

    assert not row.stable
    assert row.optimistic


@pytest.mark.parametrize(
    ("values", "stable", "range_pct"),
    [
        ([99, 100, 100, 100, 101], True, 2.0),
        ([98.99, 100, 100, 100, 101], False, 2.01),
    ],
)
def test_stability_range_boundary(tmp_path, values, stable, range_pct):
    config = tmp_path / "config"
    write_config(config)

    row = cpp_row(
        baseline_stability.analyse(
            write_wave(tmp_path / "wave", values), config_dir=config
        )
    )

    assert row.range_pct == pytest.approx(range_pct)
    assert row.stable is stable
    assert not row.under_sampled


@pytest.mark.parametrize(
    ("baseline", "expected_delta", "optimistic"),
    [
        (99.0, pytest.approx(1.0101, rel=1e-4), True),
        (100 / 1.01, pytest.approx(1.0), False),
        (99.1, pytest.approx(0.90817, rel=1e-4), False),
        (100.1, pytest.approx(-0.0999, rel=1e-4), False),
    ],
)
def test_optimistic_requires_bias_over_one_percent_and_below_wave(
    tmp_path, baseline, expected_delta, optimistic
):
    config = tmp_path / "config"
    write_config(config, baseline=baseline)

    row = cpp_row(
        baseline_stability.analyse(
            write_wave(tmp_path / "wave", [100] * 5), config_dir=config
        )
    )

    assert row.candidate_delta_pct == expected_delta
    assert row.stable
    assert row.optimistic is optimistic


def test_optimistic_requires_every_run_to_exceed_the_committed_value(tmp_path):
    config = tmp_path / "config"
    write_config(config, baseline=99.0)

    row = cpp_row(
        baseline_stability.analyse(
            write_wave(tmp_path / "wave", [99, 101, 101, 101, 101]),
            config_dir=config,
        )
    )

    assert row.candidate_delta_pct > 1.0
    assert row.minimum_us == row.committed_us
    assert row.stable
    assert not row.optimistic


def test_unstable_wave_fails_the_pre_import_gate(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    wave = tmp_path / "wave"
    write_wave(wave, [100, 100.1, 100.2, 100.3, 110])

    assert (
        baseline_stability.main(["--from", str(wave), "--config-dir", str(config)]) == 1
    )


def test_ten_run_wave_accepts_shared_five_run_cluster(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(
        tmp_path / "wave",
        [100, 100.1, 100.2, 100.3, 100.4, 120, 121, 122, 123, 124],
    )

    report = baseline_stability.analyse(
        paths,
        config_dir=config,
        clustered=True,
    )
    row = cpp_row(report)

    assert report.issues == []
    assert row.stable
    assert row.values_us == [100, 100.1, 100.2, 100.3, 100.4]
    assert {os.path.basename(source) for source in row.sources} == {
        f"bench-burnin-{index}-{SHA[:8]}.json" for index in range(1, 6)
    }
    text = baseline_stability.format_report(report)
    assert f"| {SKU} | 5 of 10 | 1 |" in text
    assert "import quality and regression checks still apply" in text


def test_cluster_must_use_same_runs_for_cpp_and_python(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(
        tmp_path / "wave",
        [100, 100.1, 100.2, 100.3, 100.4, 130, 131, 132, 133, 134],
    )
    for index, path in enumerate(paths):
        payload = json.loads(path.read_text())
        metrics = payload["baselines"][case_key()][SKU]
        metrics["gpu_time_us_python"] = (
            140 + index if index < 5 else 101 + (index - 5) / 10
        )
        write_json(path, payload)

    report = baseline_stability.analyse(
        paths,
        config_dir=config,
        clustered=True,
    )

    assert not cpp_row(report).stable
    assert not python_row(report).stable
    assert "## Stable selections" not in baseline_stability.format_report(report)


def test_cluster_prefers_slower_of_two_stable_regimes(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(
        tmp_path / "wave",
        [10, 10.01, 10.02, 10.03, 10.04, 15, 15.01, 15.02, 15.03, 15.04],
    )

    report = baseline_stability.analyse(
        paths,
        config_dir=config,
        clustered=True,
    )

    assert cpp_row(report).values_us == [15, 15.01, 15.02, 15.03, 15.04]


def test_cluster_mode_rejects_an_incomplete_cohort(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100 + index / 10 for index in range(9)])

    report = baseline_stability.analyse(
        paths,
        config_dir=config,
        clustered=True,
    )

    assert any(
        "expected 10 artifacts exactly, found 9" in issue for issue in report.issues
    )
    assert not cpp_row(report).stable
    assert not python_row(report).stable


def test_incomplete_or_copied_runs_fail_input_validation(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3])
    report = baseline_stability.analyse(paths, config_dir=config)
    assert any("expected 5 artifacts" in issue for issue in report.issues)

    duplicate = tmp_path / "copy" / paths[0].name
    duplicate.parent.mkdir()
    payload = json.loads(paths[0].read_text())
    payload["run_metadata"]["k8s_node_name"] = "retry-node"
    duplicate.write_text(json.dumps(payload))
    second_duplicate = tmp_path / "copy-2" / paths[0].name
    second_duplicate.parent.mkdir()
    second_duplicate.write_text(json.dumps(payload))
    report = baseline_stability.analyse(
        paths + [duplicate, second_duplicate], config_dir=config
    )
    assert any(
        f"{duplicate}: duplicates artifact {paths[0]}" == issue
        for issue in report.issues
    )
    assert any(
        f"{second_duplicate}: duplicates artifact {paths[0]}" == issue
        for issue in report.issues
    )


def test_sku_scope_narrows_the_cohort_without_relaxing_it(tmp_path):
    """A scoped regen measures one SKU on purpose; an unscoped wave still owes all.

    jetson-burnin selects a single SKU's legs, so requiring artifacts from every
    mapped SKU would make its waves permanently unimportable. Narrowing is explicit
    so the completeness rule keeps protecting an ordinary wave.
    """
    config = tmp_path / "config"
    write_config(config)
    add_second_sku(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])

    unscoped = baseline_stability.analyse(paths, config_dir=config)
    assert any(
        f"{SECOND_SKU}: expected 5 artifacts exactly, found 0" in issue
        for issue in unscoped.issues
    ), "an unscoped wave still owes every mapped SKU"

    scoped = baseline_stability.analyse(paths, config_dir=config, sku_scope={SKU})
    assert not any(SECOND_SKU in issue for issue in scoped.issues)

    with pytest.raises(BaselineError):
        baseline_stability.analyse(paths, config_dir=config, sku_scope={"NO_SUCH_SKU"})


def test_multi_sku_counts_expected_rows_and_provenance(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    add_second_sku(config)
    first_paths = write_wave(tmp_path / "first-wave", [100, 100.1, 100.2, 100.3, 100.4])

    partial = baseline_stability.analyse(first_paths, config_dir=config)

    assert any(
        f"{SECOND_SKU}: expected 5 artifacts exactly, found 0" in issue
        for issue in partial.issues
    )
    assert any(
        "missing declared case/SKU entries: 1" in issue for issue in partial.row_issues
    )
    assert partial.provenance_by_sku[SECOND_SKU]["nodes"] == []

    second_paths = write_wave(
        tmp_path / "second-wave", [200, 200.1, 200.2, 200.3, 200.4]
    )
    rewrite_wave_as_second_sku(second_paths)
    complete = baseline_stability.analyse(first_paths + second_paths, config_dir=config)

    assert complete.issues == []
    assert complete.row_issues == []
    assert complete.artifact_counts_by_sku == {SKU: 5, SECOND_SKU: 5}
    assert complete.provenance_by_sku[SKU]["nodes"] == [
        "node-1",
        "node-2",
        "node-3",
        "node-4",
        "node-5",
    ]
    assert complete.provenance_by_sku[SECOND_SKU]["nodes"] == [
        "second-node-1",
        "second-node-2",
        "second-node-3",
        "second-node-4",
        "second-node-5",
    ]


def test_superseded_sixth_artifact_is_rejected_explicitly(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4, 100.5])

    report = baseline_stability.analyse(paths, config_dir=config)

    assert any(
        "expected 5 artifacts exactly, found 6; remove superseded artifacts" in issue
        for issue in report.issues
    )


def test_declared_but_unmeasured_case_is_rejected(tmp_path):
    config = tmp_path / "config"
    write_config(config, shapes=("1x8x8", "1x16x16"))
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])

    report = baseline_stability.analyse(paths, config_dir=config)

    assert any(
        "missing declared case/SKU entries: 1" in issue for issue in report.row_issues
    )
    assert set(report.row_issue_keys) == {
        (CONFIG, case_key("1x16x16"), SKU, language) for language in ("cpp", "python")
    }
    assert (
        baseline_stability.main(
            ["--from", str(tmp_path / "wave"), "--config-dir", str(config)]
        )
        == 1
    )


def test_preaveraged_artifacts_report_one_issue_per_artifact(tmp_path):
    config = tmp_path / "config"
    write_config(config, shapes=("1x8x8", "1x16x16"))
    paths = write_wave(
        tmp_path / "wave",
        [100, 100.1, 100.2, 100.3, 100.4],
        include_second_case=True,
        n_runs=5,
    )

    report = baseline_stability.analyse(paths, config_dir=config)

    raw_issues = [
        issue for issue in report.issues if "requires raw per-run artifacts" in issue
    ]
    assert len(raw_issues) == 5
    assert all("n_runs=1, found [5]" in issue for issue in raw_issues)


def test_provenance_guards_reject_untrusted_artifact(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    payload = json.loads(paths[0].read_text())
    metadata = payload["run_metadata"]
    metadata.pop("driver_version")
    metadata["sm_clock_device_verified"] = False
    metadata["git_ref"] = "test/other"
    metadata["git_sha"] = "b" * 40
    metadata["gpu_name"] = "NVIDIA Other GPU"
    metadata["power_cap_w"] = "not-a-number"
    write_json(paths[0], payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert any(
        "missing metadata ['driver_version']" in issue for issue in report.issues
    )
    assert any("SM clock was not device-verified" in issue for issue in report.issues)
    assert any("does not match metadata SKU None" in issue for issue in report.issues)
    assert any("expected one revision identity" in issue for issue in report.issues)
    assert any("expected one git ref" in issue for issue in report.issues)
    assert (
        baseline_stability.main(
            ["--from", str(tmp_path / "wave"), "--config-dir", str(config)]
        )
        == 1
    )


def test_legacy_lock_request_metadata_is_not_device_verification(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    for path in paths:
        payload = json.loads(path.read_text())
        payload["run_metadata"].pop("sm_clock_device_verified")
        payload["run_metadata"]["sm_clock_locked"] = True
        write_json(path, payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert any("SM clock was not device-verified" in issue for issue in report.issues)


def test_missing_provenance_on_valid_sku_is_reported_without_traceback(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    payload = json.loads(paths[0].read_text())
    payload["run_metadata"].pop("driver_version")
    write_json(paths[0], payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert any(
        "missing metadata ['driver_version']" in issue for issue in report.issues
    )
    assert report.artifact_counts_by_sku == {SKU: 5}
    assert report.provenance_by_sku[SKU]["drivers"] == ["600.0"]


def test_revision_falls_back_to_artifact_filename(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    renamed = []
    for index, path in enumerate(paths, 1):
        payload = json.loads(path.read_text())
        payload["run_metadata"].pop("git_sha")
        renamed_path = (
            tmp_path
            / "filename-wave"
            / f"bench-burnin-1-cu13-TEST_GPU-{index}-{SHA[:8]}.json"
        )
        write_json(renamed_path, payload)
        renamed.append(renamed_path)

    report = baseline_stability.analyse(renamed, config_dir=config)

    assert report.revisions == [SHA[:8]]
    assert not any("missing revision identity" in issue for issue in report.issues)


def test_full_and_filename_revision_identities_match(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    payload = json.loads(paths[0].read_text())
    payload["run_metadata"].pop("git_sha")
    renamed_path = (
        tmp_path / "filename-wave" / f"bench-burnin-1-TEST_GPU-1-{SHA[:8]}.json"
    )
    write_json(renamed_path, payload)
    paths[0] = renamed_path

    payload = json.loads(paths[1].read_text())
    payload["run_metadata"]["git_sha"] = SHA.upper()
    write_json(paths[1], payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert report.revisions == [SHA]
    assert not any("expected one revision identity" in issue for issue in report.issues)


def test_short_metadata_revision_survives_renaming(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    renamed = []
    for index, path in enumerate(paths, 1):
        payload = json.loads(path.read_text())
        payload["run_metadata"]["git_sha"] = SHA[:8].upper()
        renamed_path = tmp_path / "renamed" / f"run-{index}.json"
        write_json(renamed_path, payload)
        renamed.append(renamed_path)

    report = baseline_stability.analyse(renamed, config_dir=config)

    assert report.revisions == [SHA[:8]]
    assert not any("missing revision identity" in issue for issue in report.issues)


def test_metadata_revision_ignores_surrounding_whitespace(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    for path in paths:
        payload = json.loads(path.read_text())
        payload["run_metadata"]["git_sha"] = f"  {SHA}\n"
        write_json(path, payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert report.revisions == [SHA]
    assert report.issues == []


def test_metadata_and_filename_revisions_must_match(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    payload = json.loads(paths[0].read_text())
    payload["run_metadata"]["git_sha"] = "b" * 40
    renamed_path = tmp_path / "mismatch" / f"bench-burnin-1-TEST_GPU-1-{SHA[:8]}.json"
    write_json(renamed_path, payload)
    paths[0] = renamed_path

    report = baseline_stability.analyse(paths, config_dir=config)

    assert any(
        f"metadata revision {'b' * 40} does not match filename revision aaaaaaaa"
        in issue
        for issue in report.issues
    )


def test_invalid_metadata_revision_does_not_fall_back_to_filename(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    payload = json.loads(paths[0].read_text())
    payload["run_metadata"]["git_sha"] = "unknown"
    write_json(paths[0], payload)

    report = baseline_stability.analyse(paths, config_dir=config)

    assert any(
        "metadata git_sha is not an 8-40 digit hex revision" in issue
        for issue in report.issues
    )
    assert any(
        "bench-burnin-1-aaaaaaaa.json: missing revision identity" in issue
        for issue in report.issues
    )


def test_revision_rejects_non_burnin_hex_suffix(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    renamed = []
    for index, path in enumerate(paths, 1):
        payload = json.loads(path.read_text())
        payload["run_metadata"].pop("git_sha")
        renamed_path = tmp_path / "without-revision" / f"run-{index}-20260315.json"
        write_json(renamed_path, payload)
        renamed.append(renamed_path)

    report = baseline_stability.analyse(renamed, config_dir=config)

    assert report.revisions == []
    assert any("missing revision identity" in issue for issue in report.issues)


def test_missing_revision_identity_is_rejected(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])
    renamed = []
    for index, path in enumerate(paths, 1):
        payload = json.loads(path.read_text())
        payload["run_metadata"].pop("git_sha")
        renamed_path = tmp_path / "without-revision" / f"run-{index}.json"
        write_json(renamed_path, payload)
        renamed.append(renamed_path)

    report = baseline_stability.analyse(renamed, config_dir=config)

    assert any(
        "run-1.json: missing revision identity" in issue for issue in report.issues
    )


def test_operator_scope_ignores_unselected_declared_cases(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    gaussian_config = "gaussian_basic"
    gaussian_case = case_key_from_axis_values(
        gaussian_config, [("InOutDataType", "uint8"), ("shape", "1x8x8")]
    )
    write_json(
        config / "operators" / "gaussian.json",
        {
            "benchmark": "gaussian",
            "configs": {
                gaussian_config: {
                    "tier": "basic",
                    "dtypes": ["uint8"],
                    "string_axes": {"shape": ["1x8x8"]},
                    "baselines": {gaussian_case: {}},
                }
            },
        },
    )
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])

    scoped = baseline_stability.analyse(paths, config_dir=config, operators={"resize"})
    unscoped = baseline_stability.analyse(paths, config_dir=config)

    assert scoped.issues == []
    assert any(
        "missing declared case/SKU entries" in issue for issue in unscoped.row_issues
    )

    absent_operator = baseline_stability.analyse(
        paths, config_dir=config, operators={"gaussian"}
    )
    assert absent_operator.issues == []
    assert any(
        "missing declared case/SKU entries" in issue
        for issue in absent_operator.row_issues
    )
    assert not any("payload SKU []" in issue for issue in absent_operator.issues)


def test_unknown_operator_is_rejected(tmp_path, capsys):
    config = tmp_path / "config"
    write_config(config)
    wave = tmp_path / "wave"
    write_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])

    assert (
        baseline_stability.main(
            [
                "--from",
                str(wave),
                "--config-dir",
                str(config),
                "--operator",
                "unknown",
            ]
        )
        == 1
    )
    assert "unknown --operator value(s): ['unknown']" in capsys.readouterr().err


def test_custom_sku_map_is_honoured(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    custom_sku_map = tmp_path / "custom-sku-map.json"
    (config / "sku_map.json").replace(custom_sku_map)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])

    assert (
        baseline_stability.main(
            [
                "--from",
                str(paths[0].parent),
                "--config-dir",
                str(config),
                "--sku-map",
                str(custom_sku_map),
            ]
        )
        == 0
    )


def test_report_renders_statistics_nodes_findings_and_json(tmp_path):
    config = tmp_path / "config"
    write_config(config, baseline=98.0, n_runs=2)
    wave = tmp_path / "wave"
    write_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])
    markdown = tmp_path / "report.md"
    json_out = tmp_path / "report.json"

    assert (
        baseline_stability.main(
            [
                "--from",
                str(wave),
                "--config-dir",
                str(config),
                "--markdown",
                str(markdown),
                "--json-out",
                str(json_out),
            ]
        )
        == 0
    )
    text = markdown.read_text()
    assert "5: node-1, node-2, node-3, node-4, node-5" in text
    assert "| Old | Min | Median | Mean | Max | Runs (us) |" in text
    assert "| 98.000 | 100.000 | 100.200 | 100.200 | 100.400 |" in text
    assert "## Under-sampled committed rows" in text
    under_sampled = text.split("## Under-sampled committed rows", 1)[1].split(
        "## Unstable rows", 1
    )[0]
    assert "| n_runs | SKU | Config | Case |" in under_sampled
    assert f"| 2 | {SKU} | `{CONFIG}` |" in under_sampled
    assert "| Lang |" not in under_sampled
    payload = json.loads(json_out.read_text())
    assert payload["rows"][0]["minimum_us"] == 100.0
    assert payload["rows"][0]["median_us"] == 100.2
    assert payload["rows"][0]["mean_us"] == pytest.approx(100.2)
    assert payload["rows"][0]["maximum_us"] == 100.4
    assert len(payload["rows"]) == 2


def test_new_row_renders_without_a_committed_baseline(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    config_path = config / "operators" / "resize.json"
    payload = json.loads(config_path.read_text())
    payload["configs"][CONFIG]["baselines"] = {}
    write_json(config_path, payload)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 100.4])

    row = cpp_row(baseline_stability.analyse(paths, config_dir=config))
    rendered = "\n".join(baseline_stability.format_rows([row], label="new"))

    assert row.committed_us is None
    assert row.candidate_delta_pct is None
    assert "| new | 100.000 | 100.200 | 100.200 | 100.400 |" in rendered
    assert "| new | TEST_GPU | cpp |" in rendered


def test_unstable_report_renders_finding_with_nonpositive_old_value(tmp_path):
    config = tmp_path / "config"
    write_config(config, baseline=0.0)
    paths = write_wave(tmp_path / "wave", [100, 100.1, 100.2, 100.3, 110])

    text = baseline_stability.format_report(
        baseline_stability.analyse(paths, config_dir=config)
    )

    assert "## Unstable rows" in text
    assert "| 0.000 | 100.000 | 100.200 |" in text
    assert "| new |" in text


def test_artifact_collection_rejects_each_empty_directory(tmp_path):
    good = tmp_path / "good"
    empty = tmp_path / "empty"
    good.mkdir()
    empty.mkdir()
    write_json(good / "run.json", {})

    with pytest.raises(BaselineError, match="directory has no .*json files"):
        collect_artifact_jsons([good, empty])


def test_main_reports_artifact_collection_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert baseline_stability.main(["--from", str(empty)]) == 1
    error = capsys.readouterr().err
    assert "ERROR:" in error
    assert "directory has no *.json files" in error


def test_main_reports_unreadable_artifact(tmp_path, capsys):
    artifact = tmp_path / "broken.json"
    artifact.write_text("{not json")

    assert baseline_stability.main(["--from", str(artifact)]) == 1
    assert "unreadable benchmark artifact" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "text"])
def test_artifact_collection_rejects_invalid_input(tmp_path, kind):
    path = tmp_path / ("missing.json" if kind == "missing" else "notes.txt")
    if kind == "text":
        path.write_text("not an artifact")

    with pytest.raises(BaselineError, match="does not exist|not a .json"):
        collect_artifact_jsons([path])


def test_artifact_collection_deduplicates_directory_and_file(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact = artifacts / "run.json"
    write_json(artifact, {})

    assert collect_artifact_jsons([artifacts, artifact]) == [artifact]


def test_cli_help_documents_inputs_and_exit_status(capsys):
    with pytest.raises(SystemExit) as exc:
        baseline_stability.parse_args(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--from JSON_OR_DIR" in output
    assert "--operator NAME" in output
    assert "Exit status:" in output


def test_cli_accepts_comma_separated_operator_values(tmp_path):
    config = tmp_path / "config"
    write_config(config)
    wave = tmp_path / "wave"
    write_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])

    assert (
        baseline_stability.main(
            [
                "--from",
                str(wave),
                "--config-dir",
                str(config),
                "--operator",
                "resize, resize",
            ]
        )
        == 0
    )
