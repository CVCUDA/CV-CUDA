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

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / ".agents" / "tools"))
sys.path.insert(0, str(REPO / "bench"))

from _internal.baselines import expected_case_keys_for_entry  # noqa: E402
from optimization_summary import (  # noqa: E402
    CHECKLIST_LABELS,
    SummaryError,
    SummaryMetadata,
    generate_summary,
    parse_summary,
    refresh_summary,
    validate_summary,
)
from optimization_summary import _config_surface, _logical_type  # noqa: E402


A100 = "A100_PCIE_40GB_250W_1095MHz"
H100 = "H100_PCIe_350W_1095MHz"
BASE = "1" * 40
CANDIDATE = "2" * 40


def _entry(tier, dtype, layout, before, after, *, extra_axes=None):
    axes = {
        "shape": ["4x1080x1920"],
        "layout": [layout],
        "inputKind": ["Tensor"],
        **(extra_axes or {}),
    }
    entry = {
        "tier": tier,
        "dtypes": [dtype],
        "string_axes": axes,
        "baselines": {},
    }
    return entry, before, after


def _surface():
    specs = {
        "example_rgb_interleaved_basic": _entry("basic", "uchar3", "NHWC", 20, 10),
        "example_rgb_planar_basic": _entry("basic", "uchar3", "NCHW", 10, 5),
        "example_rgb_fake_advanced": _entry("advanced", "uchar3", "NCHW_FAKE", 30, 20),
        "example_rgb_planar_pair_advanced": _entry("advanced", "uchar3", "NCHW", 10, 5),
        "example_float_interleaved_advanced": _entry(
            "advanced", "float3", "NHWC", 40, 20
        ),
        "example_float_planar_advanced": _entry("advanced", "float3", "NCHW", 20, 10),
    }
    before = {"benchmark": "example", "configs": {}}
    after = {"benchmark": "example", "configs": {}}
    for key, (entry, old, new) in specs.items():
        before_entry = copy.deepcopy(entry)
        after_entry = copy.deepcopy(entry)
        case_key = expected_case_keys_for_entry(key, entry)[0]
        before_entry["baselines"] = {
            case_key: {
                A100: {"gpu_time_us_cpp": old},
                H100: {"gpu_time_us_cpp": old * 2},
            }
        }
        after_entry["baselines"] = {
            case_key: {
                A100: {"gpu_time_us_cpp": new},
                H100: {"gpu_time_us_cpp": new * 2},
            }
        }
        before["configs"][key] = before_entry
        after["configs"][key] = after_entry
    return before, after


def _python_overhead_surface():
    before, after = _surface()
    before_gaps = (100, 80, 60, 40, 20, 10)
    after_gaps = (20, 30, 40, 30, 10, 15)
    for index, key in enumerate(before["configs"]):
        before_payload = next(iter(before["configs"][key]["baselines"].values()))
        after_payload = next(iter(after["configs"][key]["baselines"].values()))
        for stem, scale in ((A100, 1), (H100, 2)):
            before_payload[stem]["gpu_time_us_python"] = (
                before_payload[stem]["gpu_time_us_cpp"] + before_gaps[index] * scale
            )
            after_payload[stem]["gpu_time_us_python"] = (
                after_payload[stem]["gpu_time_us_cpp"] + after_gaps[index] * scale
            )
    return before, after


def _surface_with_valid_advanced_fake_pair():
    return _surface()


def _metadata(candidate):
    keys = [
        next(iter(candidate["configs"]["example_rgb_interleaved_basic"]["baselines"])),
        next(iter(candidate["configs"]["example_float_planar_advanced"]["baselines"])),
    ]
    return SummaryMetadata("Example", "final", BASE, CANDIDATE, tuple(keys))


def _sku_map():
    return {"entries": [{"stem": H100}, {"stem": A100}]}


def _checklist():
    evidence = {
        "Pixelwise equality to reference": (
            "cvcuda_test_system used EXPECT_EQ against the CPU reference; "
            "12/12 passed."
        ),
        "Memory-footprint checks": "ODO-9 `compute-sanitizer` memory checks passed 12/12.",
        "Baselines updated": (
            "bench/_internal/update_baseline.py imported A100/H100 artifacts "
            "and committed generated files."
        ),
        "Baseline validation": (
            "bench/_internal/validate_baselines.py passed against the baseline "
            "with zero regressions."
        ),
        "Lead exhaustion": "ncu measured 93% Memory SOL at-ridge; PASS.",
        "Review/refactor gate": "refactor_op.py --phase assess PASS with 0 recommendations.",
    }
    return {label: (True, evidence[label]) for label in CHECKLIST_LABELS}


def _render(*, state="final", prefix="", suffix="", secondary_operators=()):
    before, after = _surface()
    source = _metadata(after)
    metadata = SummaryMetadata(
        source.operator,
        source.state,
        source.baseline_commit,
        source.candidate_commit,
        source.optimized_cases,
        impact_metric=source.impact_metric,
        secondary_operators=secondary_operators,
    )
    if state != "final":
        metadata = SummaryMetadata(
            metadata.operator,
            state,
            metadata.baseline_commit,
            metadata.candidate_commit,
            metadata.optimized_cases,
            impact_metric=metadata.impact_metric,
            secondary_operators=metadata.secondary_operators,
        )
    return generate_summary(
        metadata,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        checklist=_checklist(),
        learnings=(
            "Vectorized RGB loads made Tensor Interleaved RGB8 cases 2.00x faster.",
        ),
        prefix=prefix,
        suffix=suffix,
    )


def test_generate_parse_and_validate_full_basic_advanced_surface():
    before, after = _surface()
    text = _render()
    parsed = parse_summary(text)

    assert parsed.optimized_count == 2
    assert parsed.total_count == 6
    assert parsed.categories == (
        "Tensor Interleaved RGB8",
        "Tensor Planar RGBF32",
    )
    assert [row.sku for row in parsed.impact_rows[:4]] == [
        "A100",
        "A100",
        "H100",
        "H100",
    ]
    assert "| A100 | Full operator | 6 | 1.50x | 2.00x | 2.00x |" in text
    assert "| A100 | Interleaved / Planar | 2.00x / 2.00x / 2.00x" in text
    assert "| A100 | FakePlanar / Planar | 3.00x / 3.00x / 3.00x | 4.00x" in text

    result = validate_summary(
        parsed,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        expected_operator="Example",
        expected_candidate_commit=CANDIDATE,
        evidence_results={label: "PASS" for label in CHECKLIST_LABELS},
    )
    assert result.ok, result.errors


def test_default_cpp_impact_metadata_remains_backward_compatible():
    text = _render()

    assert '"impact_metric"' not in text.splitlines()[0]
    assert parse_summary(text).metadata.impact_metric == "cpp_time"

    explicit = text.replace(
        '"state":"final"',
        '"state":"final","impact_metric":"cpp_time"',
        1,
    )
    assert parse_summary(explicit).metadata.impact_metric == "cpp_time"


def test_secondary_operator_metadata_round_trips_and_refreshes():
    before, after = _surface()
    text = _render(secondary_operators=("BndBox",))

    assert '"secondary_operators":["BndBox"]' in text.splitlines()[0]
    assert parse_summary(text).metadata.secondary_operators == ("BndBox",)

    refreshed = refresh_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        state="provisional",
    )
    assert parse_summary(refreshed).metadata.secondary_operators == ("BndBox",)


def test_secondary_operator_metadata_supports_bounded_shared_surfaces():
    secondary = tuple(f"Shared{index}" for index in range(34))
    text = _render(secondary_operators=secondary)

    assert parse_summary(text).metadata.secondary_operators == secondary

    with pytest.raises(SummaryError, match="at most 64 names"):
        _render(secondary_operators=tuple(f"Shared{index}" for index in range(65)))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('"BndBox"', "must be an array"),
        ('["BndBox","bndbox"]', "duplicate operators"),
        ('["Example"]', "must not repeat the primary"),
    ],
)
def test_secondary_operator_metadata_rejects_invalid_declarations(payload, message):
    text = _render().replace(
        '"operator":"Example"',
        f'"operator":"Example","secondary_operators":{payload}',
        1,
    )
    with pytest.raises(SummaryError, match=message):
        parse_summary(text)


def test_python_overhead_impact_is_derived_and_validated():
    before, after = _python_overhead_surface()
    metadata = SummaryMetadata(
        "Example",
        "final",
        BASE,
        CANDIDATE,
        _metadata(after).optimized_cases,
        impact_metric="python_overhead",
    )
    text = generate_summary(
        metadata,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="nsys measured 93% memory throughput and Python launch gaps.",
        checklist=_checklist(),
        learnings=("Caching metadata reduced measured Python overhead by 80.00 µs.",),
    )
    parsed = parse_summary(text)

    assert parsed.metadata.impact_metric == "python_overhead"
    assert '"impact_metric":"python_overhead"' in text.splitlines()[0]
    assert (
        "Python overhead is `gpu_time_us_python - gpu_time_us_cpp`; "
        "reduction is `before - after`" in text
    )
    assert (
        "| A100 | Optimized | 2 | 10.00 µs / 55.00 µs / 100.00 µs | "
        "15.00 µs / 17.50 µs / 20.00 µs | "
        "-5.00 µs / 37.50 µs / 80.00 µs |" in text
    )
    assert (
        "| H100 | Full operator | 6 | 20.00 µs / 100.00 µs / 200.00 µs | "
        "20.00 µs / 50.00 µs / 80.00 µs | "
        "-10.00 µs / 30.00 µs / 160.00 µs |" in text
    )

    result = validate_summary(
        parsed,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        evidence_results={label: "PASS" for label in CHECKLIST_LABELS},
    )
    assert result.ok, result.errors

    refreshed = refresh_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        state="provisional",
    )
    assert parse_summary(refreshed).metadata.impact_metric == "python_overhead"

    edited = text.replace(
        "-5.00 µs / 37.50 µs / 80.00 µs",
        "-5.00 µs / 37.50 µs / 81.00 µs",
        1,
    )
    result = validate_summary(
        edited,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
    )
    assert any(item.code == "impact-statistics" for item in result.errors)


def test_python_overhead_impact_requires_python_and_cpp_timings():
    before, after = _python_overhead_surface()
    first = next(iter(after["configs"].values()))
    del next(iter(first["baselines"].values()))[A100]["gpu_time_us_python"]

    with pytest.raises(SummaryError, match="gpu_time_us_python"):
        generate_summary(
            SummaryMetadata(
                "Example",
                "provisional",
                BASE,
                CANDIDATE,
                _metadata(after).optimized_cases,
                impact_metric="python_overhead",
            ),
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="nsys measured 93% memory throughput and Python launch gaps.",
        )


def test_metadata_rejects_unknown_impact_metric():
    text = _render().replace(
        '"state":"final"',
        '"state":"final","impact_metric":"wall_clock"',
        1,
    )
    with pytest.raises(SummaryError, match="impact_metric"):
        parse_summary(text)


def test_tensor_batch_category_generates_and_validates():
    key = "example_tensor_batch_advanced"
    entry, old, new = _entry(
        "advanced",
        "uchar3",
        "NCHW",
        10,
        5,
        extra_axes={"inputKind": ["TensorBatch"]},
    )
    case_key = expected_case_keys_for_entry(key, entry)[0]
    before_entry = copy.deepcopy(entry)
    after_entry = copy.deepcopy(entry)
    before_entry["baselines"] = {
        case_key: {
            A100: {"gpu_time_us_cpp": old},
            H100: {"gpu_time_us_cpp": old * 2},
        }
    }
    after_entry["baselines"] = {
        case_key: {
            A100: {"gpu_time_us_cpp": new},
            H100: {"gpu_time_us_cpp": new * 2},
        }
    }
    before = {"benchmark": "example", "configs": {key: before_entry}}
    after = {"benchmark": "example", "configs": {key: after_entry}}
    metadata = SummaryMetadata("Example", "final", BASE, CANDIDATE, (case_key,))

    text = generate_summary(
        metadata,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="nsys measured 88% memory throughput in the TensorBatch path.",
        checklist=_checklist(),
        learnings=(
            "Caching TensorBatch metadata improved the measured path by 2.00x.",
        ),
    )
    parsed = parse_summary(text)

    assert parsed.categories == ("TensorBatch Planar RGB8",)
    result = validate_summary(
        parsed,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        expected_operator="Example",
        expected_candidate_commit=CANDIDATE,
        evidence_results={label: "PASS" for label in CHECKLIST_LABELS},
    )
    assert result.ok, result.errors


def test_generator_refuses_to_label_incomplete_evidence_final():
    before, after = _surface()
    with pytest.raises(SummaryError, match="invalid final summary"):
        generate_summary(
            _metadata(after),
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        )


def test_full_surface_rejects_omitted_advanced_case():
    before, after = _surface()
    text = _render()
    del after["configs"]["example_float_planar_advanced"]

    result = validate_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
    )

    assert not result.ok
    assert "candidate omits" in " ".join(item.message for item in result.errors)


def test_refresh_can_migrate_cpp_summary_to_python_overhead():
    before, after = _python_overhead_surface()
    text = generate_summary(
        _metadata(after),
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="nsys measured 93% memory throughput and Python launch gaps.",
        checklist=_checklist(),
        learnings=("Caching metadata reduced measured Python overhead by 80.00 µs.",),
    )

    migrated = refresh_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        impact_metric="python_overhead",
    )

    assert parse_summary(migrated).metadata.impact_metric == "python_overhead"
    assert "Before min / median / max" in migrated


def test_generator_rejects_fake_planar_whose_native_match_is_cross_tier():
    before, after = _surface_with_valid_advanced_fake_pair()
    for config in (before, after):
        del config["configs"]["example_rgb_planar_pair_advanced"]

    with pytest.raises(SummaryError, match=r"unmatched FakePlanar.*same-tier"):
        generate_summary(
            _metadata(after),
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
            checklist=_checklist(),
            learnings=("Vector loads improved the measured path by 2.00x.",),
        )


def test_generator_rejects_partial_fake_planar_pairing():
    before, after = _surface_with_valid_advanced_fake_pair()
    key = "example_rgba_fake_advanced"
    entry, old, new = _entry("advanced", "uchar4", "NCHW_FAKE", 50, 40)
    case_key = expected_case_keys_for_entry(key, entry)[0]
    for config, value in ((before, old), (after, new)):
        item = copy.deepcopy(entry)
        item["baselines"] = {
            case_key: {
                A100: {"gpu_time_us_cpp": value},
                H100: {"gpu_time_us_cpp": value * 2},
            }
        }
        config["configs"][key] = item

    with pytest.raises(SummaryError, match="1 unmatched FakePlanar"):
        generate_summary(
            _metadata(after),
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
            checklist=_checklist(),
            learnings=("Vector loads improved the measured path by 2.00x.",),
        )


def test_validator_rejects_unmatched_fake_planar_in_supplied_artifacts():
    before, after = _surface_with_valid_advanced_fake_pair()
    text = generate_summary(
        _metadata(after),
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        checklist=_checklist(),
        learnings=("Vector loads improved the measured path by 2.00x.",),
    )
    for config in (before, after):
        del config["configs"]["example_rgb_planar_pair_advanced"]

    result = validate_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
    )

    assert not result.ok
    assert "unmatched FakePlanar" in " ".join(item.message for item in result.errors)


def test_operator_that_declares_only_basic_tier_is_valid():
    before, after = _surface()
    before["configs"] = {
        key: value
        for key, value in before["configs"].items()
        if value["tier"] == "basic"
    }
    after["configs"] = {
        key: value
        for key, value in after["configs"].items()
        if value["tier"] == "basic"
    }
    key = next(iter(after["configs"]["example_rgb_interleaved_basic"]["baselines"]))
    text = generate_summary(
        SummaryMetadata("Example", "provisional", BASE, CANDIDATE, (key,)),
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
    )
    assert parse_summary(text).total_count == 2


def test_no_layout_operator_gets_verified_na_rows():
    entry = {
        "tier": "basic",
        "dtypes": ["uint8"],
        "string_axes": {"shape": ["4x16x16"]},
        "baselines": {},
    }
    key = "example_basic"
    case_key = expected_case_keys_for_entry(key, entry)[0]
    before_entry = copy.deepcopy(entry)
    after_entry = copy.deepcopy(entry)
    before_entry["baselines"] = {
        case_key: {
            A100: {"gpu_time_us_cpp": 10},
            H100: {"gpu_time_us_cpp": 8},
        }
    }
    after_entry["baselines"] = {
        case_key: {
            A100: {"gpu_time_us_cpp": 5},
            H100: {"gpu_time_us_cpp": 4},
        }
    }
    before = {"benchmark": "example", "configs": {key: before_entry}}
    after = {"benchmark": "example", "configs": {key: after_entry}}
    text = generate_summary(
        SummaryMetadata("Example", "provisional", BASE, CANDIDATE, (case_key,)),
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Compute-bound",
        profile_evidence="ncu measured 91% Compute SOL and 20% Memory SOL.",
    )
    parsed = parse_summary(text)
    assert parsed.categories == ("Tensor NoLayout U8",)
    assert all(row.before is None and row.after is None for row in parsed.layout_rows)


def test_non_image_point_layouts_are_categorized_as_no_layout():
    entry = {
        "tier": "basic",
        "dtypes": ["float32"],
        "string_axes": {"shape": ["512x1024"], "layout": ["NW", "NWC"]},
        "baselines": {},
    }
    key = "example_basic"
    case_keys = expected_case_keys_for_entry(key, entry)
    before_entry = copy.deepcopy(entry)
    after_entry = copy.deepcopy(entry)
    before_entry["baselines"] = {
        case_key: {
            A100: {"gpu_time_us_cpp": 10},
            H100: {"gpu_time_us_cpp": 8},
        }
        for case_key in case_keys
    }
    after_entry["baselines"] = {
        case_key: {
            A100: {"gpu_time_us_cpp": 5},
            H100: {"gpu_time_us_cpp": 4},
        }
        for case_key in case_keys
    }
    before = {"benchmark": "example", "configs": {key: before_entry}}
    after = {"benchmark": "example", "configs": {key: after_entry}}

    text = generate_summary(
        SummaryMetadata("Example", "provisional", BASE, CANDIDATE, case_keys),
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Compute-bound",
        profile_evidence="ncu measured 91% Compute SOL and 20% Memory SOL.",
    )

    parsed = parse_summary(text)
    assert parsed.categories == ("Tensor NoLayout F32",)
    assert all(row.before is None and row.after is None for row in parsed.layout_rows)


def test_full_surface_rejects_missing_expanded_baseline_key():
    before, after = _surface()
    text = _render()
    entry = after["configs"]["example_float_planar_advanced"]
    entry["baselines"].clear()

    result = validate_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
    )

    assert not result.ok
    assert "missing 1 expanded case" in " ".join(item.message for item in result.errors)


def test_full_surface_rejects_missing_reference_sku_timing():
    before, after = _surface()
    text = _render()
    case = next(
        iter(after["configs"]["example_float_planar_advanced"]["baselines"].values())
    )
    del case[H100]

    result = validate_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
    )

    assert not result.ok
    assert "lacks full before/after" in " ".join(item.message for item in result.errors)


def test_refresh_preserves_human_fields_and_surrounding_description():
    before, after = _surface()
    memory_section = (
        "\n\n## Memory footprint\n"
        "Peak attributable increase: 0 B\n"
        "New runtime CUDA allocations/frees: no\n"
        "Evidence: Source inspection found no new runtime allocations.\n"
    )
    text = _render(prefix="Intro\n\n", suffix=memory_section)
    refreshed = refresh_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        state="provisional",
        candidate_commit="3" * 40,
    )
    parsed = parse_summary(refreshed)

    assert refreshed.startswith("Intro\n\n")
    assert refreshed.endswith(memory_section)
    assert parsed.bottleneck == "Memory-bound"
    assert parsed.profile_evidence == "ncu measured 93% Memory SOL and 21% Compute SOL."
    assert parsed.checklist[0].evidence.endswith("12/12 passed.")
    assert parsed.learnings[0].startswith("Vectorized RGB loads")
    assert parsed.metadata.state == "provisional"
    assert parsed.metadata.candidate_commit == "3" * 40


def test_parser_ignores_fenced_example_but_rejects_duplicate_live_blocks():
    live = _render()
    fenced = "```markdown\n" + live + "\n```\n\n"
    assert parse_summary(fenced + live).metadata.operator == "Example"
    four_tick_fence = "````markdown\n```\n" + live + "\n```\n````\n\n"
    assert parse_summary(four_tick_fence + live).metadata.operator == "Example"
    with pytest.raises(SummaryError, match="exactly one"):
        parse_summary(live + "\n" + live)


def test_visible_schema_rejects_extra_detail_content():
    scope_detail = _render().replace(
        "\n### Impact\n",
        "\nExtra per-case detail must stay in artifacts.\n\n### Impact\n",
    )
    with pytest.raises(SummaryError, match="Scope may contain only"):
        parse_summary(scope_detail)

    impact_detail = _render().replace(
        "\n### Layout comparison\n",
        "\n| Extra | Per-case | Detail |\n\n### Layout comparison\n",
    )
    with pytest.raises(SummaryError, match="Impact table"):
        parse_summary(impact_detail)


@pytest.mark.parametrize("bad_state", [{}, [], 7, None])
def test_untrusted_metadata_type_errors_are_bounded(bad_state):
    text = _render()
    text = text.replace('"state":"final"', f'"state":{json.dumps(bad_state)}', 1)
    with pytest.raises(SummaryError, match="state"):
        parse_summary(text)


def test_checked_item_requires_hard_evidence_and_matching_gate():
    before, after = _surface()
    text = _render().replace(
        "cvcuda_test_system used EXPECT_EQ against the CPU reference; 12/12 passed.",
        "PASS",
    )
    result = validate_summary(
        text,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        evidence_results={label: "PASS" for label in CHECKLIST_LABELS},
    )
    assert any(item.code == "checklist-evidence" for item in result.errors)

    result = validate_summary(
        _render(), evidence_results={label: "MANUAL" for label in CHECKLIST_LABELS}
    )
    assert len([item for item in result.errors if item.code == "checklist-gate"]) == 6


def test_hand_edited_impact_and_layout_statistics_are_rejected():
    before, after = _surface()
    impact = _render().replace(
        "| A100 | Optimized | 2 | 2.00x | 2.00x | 2.00x |",
        "| A100 | Optimized | 2 | 1.00x | 1.00x | 1.00x |",
        1,
    )
    result = validate_summary(
        impact,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
    )
    assert any(item.code == "impact-statistics" for item in result.errors)

    layout = _render().replace(
        "| A100 | Interleaved / Planar | 2.00x / 2.00x / 2.00x | 2.00x / 2.00x / 2.00x |",
        "| A100 | Interleaved / Planar | 1.00x / 1.00x / 1.00x | 1.00x / 1.00x / 1.00x |",
        1,
    )
    result = validate_summary(
        layout,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
    )
    assert any(item.code == "layout-statistics" for item in result.errors)


def test_non_reference_sku_requires_prominent_warning():
    before, after = _surface()
    for config in (before, after):
        for entry in config["configs"].values():
            payload = next(iter(entry["baselines"].values()))
            payload["RTX_6000_Ada"] = {"gpu_time_us_cpp": 100}
    text = generate_summary(
        _metadata(after),
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        checklist=_checklist(),
        learnings=("Vector loads improved the measured path by 2.00x.",),
    )
    assert "⚠️ **Non-reference local GPU:** Results from RTX 6000 Ada" in text


def test_partial_non_reference_sku_is_not_silently_omitted():
    before, after = _surface()
    first_before = next(iter(before["configs"].values()))
    first_after = next(iter(after["configs"].values()))
    next(iter(first_before["baselines"].values()))["RTX_6000_Ada"] = {
        "gpu_time_us_cpp": 10
    }
    next(iter(first_after["baselines"].values()))["RTX_6000_Ada"] = {
        "gpu_time_us_cpp": 5
    }
    with pytest.raises(SummaryError, match="lacks full before/after"):
        generate_summary(
            SummaryMetadata(
                "Example",
                "provisional",
                BASE,
                CANDIDATE,
                _metadata(after).optimized_cases,
            ),
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        )


def test_conversion_categories_use_logical_input_and_output_types():
    before, after = _surface()
    key = "example_convert_advanced"
    entry, old, new = _entry(
        "advanced",
        "uchar4",
        "NHWC",
        10,
        5,
        extra_axes={"code": ["YUV2RGB_NV12"]},
    )
    case_key = expected_case_keys_for_entry(key, entry)[0]
    for config, value in ((before, old), (after, new)):
        item = copy.deepcopy(entry)
        item["baselines"] = {
            case_key: {
                A100: {"gpu_time_us_cpp": value},
                H100: {"gpu_time_us_cpp": value},
            }
        }
        config["configs"][key] = item
    metadata = SummaryMetadata("Example", "final", BASE, CANDIDATE, (case_key,))
    text = generate_summary(
        metadata,
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Compute-bound",
        profile_evidence="ncu measured 91% Compute SOL and 20% Memory SOL.",
        checklist=_checklist(),
        learnings=("Conversion fusion improved the measured path by 2.00x.",),
    )
    assert "`Tensor Interleaved NV12→RGBA8`" in text


@pytest.mark.parametrize(
    ("axes", "expected"),
    [
        ({"InOutDataType": "uint8", "numChannels": "3"}, "RGB8"),
        (
            {"InOutDataType": "uchar3", "outDataType": "float32"},
            "RGB8→RGBF32",
        ),
        ({"InOutDataType": "uchar3", "outChannels": "4"}, "RGB8→RGBA8"),
        (
            {"InOutDataType": "short2", "outDataType": "float32"},
            "2S16→2F32",
        ),
    ],
)
def test_logical_type_uses_channel_and_output_axes(axes, expected):
    assert _logical_type(axes) == expected


def test_duplicate_workload_identity_is_rejected_before_layout_pairing():
    before, after = _surface()
    source = "example_rgb_interleaved_basic"
    duplicate = "example_rgb_interleaved_duplicate_basic"
    for config in (before, after):
        entry = copy.deepcopy(config["configs"][source])
        old_key = next(iter(entry["baselines"]))
        new_key = duplicate + old_key[slice(old_key.index("["), None)]
        entry["baselines"] = {new_key: entry["baselines"][old_key]}
        config["configs"][duplicate] = entry
    metadata = _metadata(after)
    with pytest.raises(SummaryError, match="duplicate expanded workload identity"):
        generate_summary(
            metadata,
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        )


def test_duplicate_workload_without_layout_counterpart_is_rejected():
    before, after = _surface()
    source = "example_rgb_interleaved_basic"
    duplicate = "example_rgb_interleaved_duplicate_basic"
    for config in (before, after):
        config["configs"].pop("example_rgb_planar_basic")
        entry = copy.deepcopy(config["configs"][source])
        old_key = next(iter(entry["baselines"]))
        new_key = duplicate + old_key[slice(old_key.index("["), None)]
        entry["baselines"] = {new_key: entry["baselines"][old_key]}
        config["configs"][duplicate] = entry

    with pytest.raises(SummaryError, match="duplicate expanded workload identity"):
        generate_summary(
            _metadata(after),
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
            checklist=_checklist(),
            learnings=("Vectorized RGB loads improved the measured path by 2.00x.",),
        )


def test_duplicate_workload_identity_ignores_axis_declaration_order():
    before, after = _surface()
    source = "example_float_interleaved_advanced"
    duplicate = "example_float_interleaved_reordered_advanced"
    for config in (before, after):
        entry = copy.deepcopy(config["configs"][source])
        entry["string_axes"] = dict(reversed(entry["string_axes"].items()))
        old_key = next(iter(entry["baselines"]))
        new_key = expected_case_keys_for_entry(duplicate, entry)[0]
        entry["baselines"] = {new_key: entry["baselines"][old_key]}
        config["configs"][duplicate] = entry

    with pytest.raises(SummaryError, match="complete raw axes"):
        generate_summary(
            _metadata(after),
            baseline_config=before,
            candidate_config=after,
            sku_map=_sku_map(),
            bottleneck="Memory-bound",
            profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        )


def test_legacy_duplicate_alias_canonicalization_ignores_config_order():
    canonical = "brightnesscontrast_float3_varshape_advanced"
    legacy_alias = "brightnesscontrast_rgb_f32_1080p_advanced"

    def config(order):
        entries = {}
        for config_key in order:
            entry, _, _ = _entry("advanced", "float3", "NHWC", 20, 10)
            case_key = expected_case_keys_for_entry(config_key, entry)[0]
            entry["baselines"] = {
                case_key: {
                    A100: {"gpu_time_us_cpp": 10},
                    H100: {"gpu_time_us_cpp": 20},
                }
            }
            entries[config_key] = entry
        return {"benchmark": "brightnesscontrast", "configs": entries}

    canonical_key = expected_case_keys_for_entry(
        canonical,
        config((canonical,))["configs"][canonical],
    )[0]
    baseline = _config_surface(config((canonical, legacy_alias)), "baseline")
    candidate = _config_surface(config((legacy_alias, canonical)), "candidate")

    assert tuple(baseline.cases) == (canonical_key,)
    assert tuple(candidate.cases) == (canonical_key,)


def test_identical_raw_axes_in_different_tiers_are_distinct_workloads():
    before, after = _surface()
    source = "example_float_interleaved_advanced"
    duplicate = "example_float_interleaved_basic_copy"
    for config in (before, after):
        entry = copy.deepcopy(config["configs"][source])
        entry["tier"] = "basic"
        old_key = next(iter(entry["baselines"]))
        new_key = duplicate + old_key[slice(old_key.index("["), None)]
        entry["baselines"] = {new_key: entry["baselines"][old_key]}
        config["configs"][duplicate] = entry

    text = generate_summary(
        _metadata(after),
        baseline_config=before,
        candidate_config=after,
        sku_map=_sku_map(),
        bottleneck="Memory-bound",
        profile_evidence="ncu measured 93% Memory SOL and 21% Compute SOL.",
        checklist=_checklist(),
        learnings=("Vectorized RGB loads improved the measured path by 2.00x.",),
    )

    assert parse_summary(text).metadata.operator == "Example"


def test_checked_in_operator_configs_have_supported_workload_identities():
    failures = []
    for path in sorted((REPO / "bench" / "config" / "operators").glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        try:
            _config_surface(config, path.name)
        except SummaryError as exc:
            if "duplicate expanded workload identity" in str(exc):
                failures.append(f"{path.name}: {exc}")

    assert not failures, "\n".join(failures)
