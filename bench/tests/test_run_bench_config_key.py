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

"""Unit tests for benchmark tier configs and exact --config-key discovery."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import run_bench
from _internal.baselines import (
    BaselineError,
    baseline_payload_from_dataframe,
    expected_case_keys_for_entry,
    fake_planar_pairing_issues,
    load_config_index,
    parse_case_key,
)
from _internal.quality import DEFAULT_BENCHMARK_QUALITY
from config.load_config import (
    generate_axis_args,
    get_configs_for_benchmark,
    load_bench_config,
    load_bench_manifest,
    parse_config_key_arg,
    parse_operator_arg,
    parse_tier_arg,
)


class DummyRunner(run_bench.BenchmarkRunner):
    def __init__(
        self,
        bench_folder,
        config,
        config_keys,
        language,
        operators=None,
        tiers=None,
    ):
        super().__init__(
            "bench_",
            str(bench_folder),
            operators=operators,
            language=language,
            config_keys=config_keys,
            tiers=tiers,
        )
        self.config = config
        self.operator_manifest = {
            entry["benchmark"]: {
                "config": f"operators/{entry['benchmark']}.json",
                "cpp": f"bench_{entry['benchmark']}",
                "python": f"bench_{entry['benchmark']}.py",
            }
            for entry in config.values()
            if isinstance(entry, dict) and "benchmark" in entry
        }

        from config.load_config import (  # noqa: PLC0415
            get_operator_from_benchmark_name,
            get_configs_for_benchmark,
        )

        self.get_operator_from_benchmark_name = get_operator_from_benchmark_name
        self.get_configs_for_benchmark = get_configs_for_benchmark

    def build_command(self, benchmark_path, extra_args, output_file, config_key=None):
        return []


def _touch(path):
    path.write_text("")


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


def _write_baseline_config_tree(root):
    _write_json(
        root / "sku_map.json",
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 250,
                    "locked_sm_clock_mhz": 1095,
                    "stem": "A100_PCIE_40GB_250W_1095MHz",
                }
            ]
        },
    )
    _write_json(
        root / "operators" / "resize.json",
        {
            "benchmark": "resize",
            "configs": {
                "resize_contract_area_tensor_uchar3_basic": {
                    "tier": "basic",
                    "dtypes": ["uchar3"],
                    "string_axes": {
                        "shape": ["32x1080x1920"],
                        "kernelSize": ["5x5"],
                        "resizeType": ["CONTRACT"],
                        "interpolation": ["AREA"],
                        "layout": ["NHWC"],
                        "inputKind": ["Tensor"],
                    },
                }
            },
        },
    )


def _baseline_row(language="cpp", gpu_us=525.7525, noise_us=1.2, bwutil=0.42):
    return {
        "Benchmark": "resize",
        "tier": "basic",
        "config_key": "resize_contract_area_tensor_uchar3_basic",
        "Language": language,
        "GPU Time (µs)": gpu_us,
        "GPU Noise (µs)": noise_us,
        "BWUtil": bwutil,
        "Device Name": "NVIDIA Test GPU",
        "Power Cap (W)": 250,
        "Locked SM Clock (MHz)": 1095,
        "InOutDataType": "uchar3",
        "shape": "32x1080x1920",
        "resizeType": "CONTRACT",
        "interpolation": "AREA",
        "layout": "NHWC",
        "inputKind": "Tensor",
    }


def test_advanced_tier_is_not_discovered_by_basic_tier():
    config = {
        "resize_basic": {
            "benchmark": "resize",
            "tier": "basic",
        },
        "resize_advanced": {
            "benchmark": "resize",
            "tier": "advanced",
        },
    }

    assert get_configs_for_benchmark("resize", config, tiers={"basic"}) == [
        "resize_basic"
    ]
    assert get_configs_for_benchmark("resize", config, tiers={"advanced"}) == [
        "resize_advanced"
    ]
    assert get_configs_for_benchmark("resize", config, tiers={"basic", "advanced"}) == [
        "resize_basic",
        "resize_advanced",
    ]


def test_parse_tier_arg_accepts_basic_and_advanced():
    assert parse_tier_arg("basic") == {"basic"}
    assert parse_tier_arg("advanced") == {"advanced"}
    assert parse_tier_arg("basic, advanced") == {"basic", "advanced"}

    with pytest.raises(ValueError, match="unknown tier"):
        parse_tier_arg("ci")


def test_parse_config_key_arg_single_and_comma_separated():
    assert parse_config_key_arg("resize_advanced") == ["resize_advanced"]
    assert parse_config_key_arg("resize_advanced, gaussian_advanced ") == [
        "resize_advanced",
        "gaussian_advanced",
    ]


def test_parse_config_key_arg_rejects_empty_and_duplicate_values():
    with pytest.raises(ValueError, match="empty"):
        parse_config_key_arg(" , ")

    with pytest.raises(ValueError, match="duplicate"):
        parse_config_key_arg("resize_advanced,resize_advanced")


def test_parse_operator_arg_accepts_ordered_exact_names():
    assert parse_operator_arg("resize") == ["resize"]
    assert parse_operator_arg("resize, gaussian flip") == [
        "resize",
        "gaussian",
        "flip",
    ]


def test_parse_operator_arg_rejects_empty_and_duplicate_values():
    with pytest.raises(ValueError, match="empty"):
        parse_operator_arg(" , ")

    with pytest.raises(ValueError, match="duplicate"):
        parse_operator_arg("resize,resize")


def test_load_bench_config_expands_split_manifest(tmp_path):
    operators = tmp_path / "operators"
    operators.mkdir()
    _write_json(
        operators / "resize.json",
        {
            "resize_basic": {
                "benchmark": "resize",
                "tier": "basic",
            },
        },
    )
    _write_json(
        operators / "gaussian.json",
        {
            "gaussian_advanced": {
                "benchmark": "gaussian",
                "tier": "advanced",
            },
        },
    )
    _write_json(
        tmp_path / "bench_params.json",
        {"include": ["operators/resize.json", "operators/gaussian.json"]},
    )

    assert list(load_bench_config(str(tmp_path / "bench_params.json"))) == [
        "resize_basic",
        "gaussian_advanced",
    ]
    assert list(load_bench_config(str(tmp_path))) == [
        "resize_basic",
        "gaussian_advanced",
    ]


def test_load_bench_config_expands_operator_manifest(tmp_path):
    operators = tmp_path / "operators"
    operators.mkdir()
    _write_json(
        operators / "resize.json",
        {
            "resize_basic": {
                "benchmark": "resize",
                "tier": "basic",
            },
        },
    )
    _write_json(
        operators / "gaussian.json",
        {
            "gaussian_advanced": {
                "benchmark": "gaussian",
                "tier": "advanced",
            },
        },
    )
    _write_json(
        tmp_path / "bench_params.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                },
                "gaussian": {
                    "config": "operators/gaussian.json",
                    "cpp": "bench_gaussian",
                    "python": "bench_gaussian.py",
                },
            }
        },
    )

    assert list(load_bench_config(str(tmp_path / "bench_params.json"))) == [
        "resize_basic",
        "gaussian_advanced",
    ]
    assert list(load_bench_manifest(str(tmp_path / "bench_params.json"))) == [
        "resize",
        "gaussian",
    ]


def test_load_bench_config_rejects_duplicate_split_keys(tmp_path):
    operators = tmp_path / "operators"
    operators.mkdir()
    entry = {
        "resize_basic": {
            "benchmark": "resize",
            "tier": "basic",
        },
    }
    _write_json(operators / "resize.json", entry)
    _write_json(operators / "duplicate.json", entry)
    _write_json(
        tmp_path / "bench_params.json",
        {"include": ["operators/resize.json", "operators/duplicate.json"]},
    )

    with pytest.raises(ValueError, match="Duplicate config key"):
        load_bench_config(str(tmp_path / "bench_params.json"))


def _entry_string_axis_values(entry, axis):
    return entry.get("string_axes", {}).get(axis, [])


def _entry_row_count(entry):
    count = len(entry.get("dtypes", [])) or 1
    for axis_group in ("string_axes", "int64_axes", "float64_axes"):
        for values in entry.get(axis_group, {}).values():
            count *= len(values)
    return count


_RESIZE_ANISOTROPIC_TARGETS = {
    "CONTRACT": "TARGET_480x864",
    "EXPAND": "TARGET_2160x3840",
}
_RESIZE_ANISOTROPIC_INPUT_SPATIAL_SHAPES = {
    "TARGET_480x864": ["1080", "1920"],
    "TARGET_2160x3840": ["480", "864"],
}


def _resize_coverage_signature(operator, axes):
    fields = ["InOutDataType", "interpolation", "layout", "inputKind"]
    if operator == "hqresize":
        fields.extend(("antialias", "numChannels"))
    return tuple(axes[field] for field in fields)


def test_generate_axis_args_preserves_axis_order_and_grouping():
    config = {
        "synthetic_planar_basic": {
            "benchmark": "synthetic",
            "tier": "basic",
            "dtypes": ["uchar3"],
            "string_axes": {
                "shape": ["1x2x3"],
                "layout": ["NCHW"],
                "inputKind": ["Tensor", "VarShape"],
            },
        }
    }

    assert generate_axis_args("synthetic_planar_basic", config) == [
        "--axis",
        "InOutDataType=uchar3",
        "--axis",
        "shape=1x2x3",
        "--axis",
        "layout=NCHW",
        "--axis",
        "inputKind=[Tensor,VarShape]",
    ]


def test_repository_config_separates_basic_and_advanced_tiers():
    config = load_bench_config()
    manifest = load_bench_manifest()

    assert {
        entry.get("tier") for entry in config.values() if isinstance(entry, dict)
    } == {"basic", "advanced"}
    assert manifest["resize"] == {
        "config": "operators/resize.json",
        "cpp": "bench_resize",
        "python": "bench_resize.py",
    }
    removed_field = "manual" + "_only"
    assert [
        key
        for key, entry in config.items()
        if isinstance(entry, dict) and removed_field in entry
    ] == []
    assert [
        key
        for key, entry in config.items()
        if isinstance(entry, dict)
        and entry.get("tier") == "basic"
        and "advanced" in key
    ] == []
    assert [
        key
        for key, entry in config.items()
        if isinstance(entry, dict)
        and entry.get("tier") == "advanced"
        and not key.endswith("_advanced")
    ] == []

    planar_layouts = {"NCHW", "CHW", "NCHW_FAKE", "CHW_FAKE"}
    fake_planar_layouts = {"NCHW_FAKE", "CHW_FAKE"}
    planar_keys = []
    fake_planar_keys = []

    for key, entry in config.items():
        if not isinstance(entry, dict):
            continue

        string_axes = entry.get("string_axes", {})
        layouts = _entry_string_axis_values(entry, "layout")

        if any(token in key.lower() for token in ("planar", "nchw", "chw")):
            assert (
                "layout" in string_axes
            ), f"{key} names a planar layout but has no layout axis"

        if any(layout in planar_layouts for layout in layouts):
            planar_keys.append(key)

        if any(layout in fake_planar_layouts for layout in layouts):
            fake_planar_keys.append(key)
            assert entry["tier"] == "advanced"
            input_kinds = _entry_string_axis_values(entry, "inputKind")
            assert input_kinds == [
                "Tensor"
            ], f"{key} fake-planar row is not Tensor-only"

    assert planar_keys
    assert fake_planar_keys

    # CvtColor keeps one basic case for every reviewed conversion branch.
    assert config["cvtcolor_basic"]["string_axes"]["code"] == [
        "RGB2GRAY",
        "RGB2HSV",
        "YUV2RGB",
        "YUV2RGB_NV12",
    ]
    assert config["cvtcolor_planar_nchw_1080p_basic"]["string_axes"]["code"] == [
        "RGB2GRAY",
        "RGB2HSV",
        "YUV2RGB",
    ]
    assert config["cvtcolor_rgb2rgba_uchar3_basic"]["string_axes"]["inputKind"] == [
        "Tensor",
        "VarShape",
    ]

    # CLAHE keeps the representative clip limit in basic; the alternate limit
    # and fake-planar comparison live in advanced.
    assert config["clahe_basic"]["int64_axes"]["clip10"] == [400]
    assert config["clahe_clip20_advanced"]["int64_axes"]["clip10"] == [20]
    assert config["clahe_clip20_advanced"]["tier"] == "advanced"
    assert config["clahe_fake_planar_advanced"]["tier"] == "advanced"

    # Normalize's RGB8 basic matrix also measures the tensor-free scalar-parameter path.
    assert config["normalize_rgb_u8_1080p_basic"]["string_axes"]["inputKind"] == [
        "Tensor",
        "TensorScalar",
        "VarShape",
    ]
    assert config["normalize_planar_rgb_u8_nchw_1080p_basic"]["string_axes"][
        "inputKind"
    ] == ["Tensor", "TensorScalar", "VarShape"]
    assert config["normalize_rgb_f32_1080p_basic"]["string_axes"]["inputKind"] == [
        "Tensor"
    ]
    assert config["normalize_planar_rgb_f32_nchw_1080p_basic"]["string_axes"][
        "inputKind"
    ] == ["Tensor"]
    assert config["normalize_rgb_f32_1080p_advanced"]["string_axes"]["inputKind"] == [
        "VarShape"
    ]

    # GammaContrast float3 is symmetric across native layouts for Tensor;
    # VarShape float3 remains advanced-only.
    assert config["gammacontrast_float3_lcg_basic"]["string_axes"] == {
        "shape": ["32x1080x1920"],
        "layout": ["NHWC", "NCHW"],
        "inputKind": ["Tensor"],
    }
    assert config["gammacontrast_planar_nchw_float3_varshape_lcg_advanced"][
        "string_axes"
    ]["inputKind"] == ["VarShape"]

    # The real container coverage added for non-standard APIs stays explicit.
    assert config["findhomography_basic"]["string_axes"]["inputKind"] == ["Tensor"]
    assert config["findhomography_varshape_basic"]["string_axes"]["inputKind"] == [
        "VarShape"
    ]
    assert config["stack_uchar3_basic"]["string_axes"] == {
        "shape": ["32x2160x3840"],
        "layout": ["NHWC", "NCHW"],
        "inputKind": ["Tensor", "VarShape"],
    }
    assert config["stack_uchar3_layout_compare_4k_advanced"]["string_axes"] == {
        "shape": ["4x2160x3840"],
        "layout": ["NHWC", "NCHW", "NCHW_FAKE"],
        "inputKind": ["Tensor"],
    }

    # These image-layout-bearing benchmarks now expose their truthful layout;
    # semantic N-A operators intentionally do not receive a dummy axis.
    layout_bearing = {
        "bndbox",
        "channelreorder",
        "composite",
        "cropflipnormalizereformat",
        "histogram",
        "label",
        "minmaxloc",
        "osd",
        "padandstack",
        "resizecropconvertreformat",
        "sift",
        "stack",
    }
    assert all(
        "layout" in entry.get("string_axes", {})
        for entry in config.values()
        if entry.get("benchmark") in layout_bearing
    )
    layout_na = {
        "findhomography",
        "minarearect",
        "nonmaximumsuppression",
        "pairwisematcher",
        "reformat",
    }
    assert all(
        "layout" not in entry.get("string_axes", {})
        for entry in config.values()
        if entry.get("benchmark") in layout_na
    )
    assert config["label_basic"]["string_axes"]["inputKind"] == ["Tensor"]
    assert config["pairwisematcher_uint8_basic"]["string_axes"]["inputKind"] == [
        "Tensor"
    ]

    # AdaptiveThreshold's ImageBatch path rejects planar formats; its NCHW
    # benchmark is therefore truthfully Tensor-only rather than a skipped row.
    assert config["adaptivethreshold_planar_nchw_basic"]["string_axes"][
        "inputKind"
    ] == ["Tensor"]

    assert "distancemap" not in manifest


@pytest.mark.parametrize(
    ("operator", "dtype"),
    [
        ("resize", "uchar3"),
        ("pillowresize", "uchar3"),
        ("hqresize", "uint8"),
    ],
)
def test_resize_family_has_basic_anisotropic_profiles(operator, dtype):
    config = load_bench_config()
    profiles = {
        "contract_480p_tensor": ("TARGET_480x864", "Tensor", ["1080", "1920"]),
        "contract_480p_varshape": (
            "TARGET_480x864",
            "VarShape",
            ["1080", "1920"],
        ),
        "expand_4k_tensor": ("TARGET_2160x3840", "Tensor", ["480", "864"]),
        "expand_4k_varshape": (
            "TARGET_2160x3840",
            "VarShape",
            ["480", "864"],
        ),
    }

    for suffix, (resize_type, input_kind, spatial_shape) in profiles.items():
        entry = config[f"{operator}_anisotropic_{suffix}_rgb8_basic"]
        axes = entry["string_axes"]
        assert entry["tier"] == "basic"
        assert entry["dtypes"] == [dtype]
        assert axes["layout"] == ["NHWC"]
        assert axes["resizeType"] == [resize_type]
        assert axes["inputKind"] == [input_kind]
        assert axes["shape"][0].split("x")[1:] == spatial_shape


_RESIZE_EXISTING_BASIC_ANISOTROPIC_SIGNATURES = {
    "resize": {
        ("TARGET_480x864", ("uchar3", "AREA", "NHWC", "Tensor")),
        ("TARGET_480x864", ("uchar3", "AREA", "NHWC", "VarShape")),
        ("TARGET_2160x3840", ("uchar3", "LINEAR", "NHWC", "Tensor")),
        ("TARGET_2160x3840", ("uchar3", "LINEAR", "NHWC", "VarShape")),
    },
    "pillowresize": {
        ("TARGET_480x864", ("uchar3", "CUBIC", "NHWC", "Tensor")),
        ("TARGET_480x864", ("uchar3", "CUBIC", "NHWC", "VarShape")),
        ("TARGET_2160x3840", ("uchar3", "CUBIC", "NHWC", "Tensor")),
        ("TARGET_2160x3840", ("uchar3", "CUBIC", "NHWC", "VarShape")),
    },
    "hqresize": {
        (
            "TARGET_480x864",
            ("uint8", "CUBIC", "NHWC", "Tensor", "1", "3"),
        ),
        (
            "TARGET_480x864",
            ("uint8", "CUBIC", "NHWC", "VarShape", "1", "3"),
        ),
        (
            "TARGET_2160x3840",
            ("uint8", "CUBIC", "NHWC", "Tensor", "0", "3"),
        ),
        (
            "TARGET_2160x3840",
            ("uint8", "CUBIC", "NHWC", "VarShape", "0", "3"),
        ),
    },
}


@pytest.mark.parametrize(
    "operator,basic_rows,advanced_rows",
    [
        ("resize", 22, 209),
        ("pillowresize", 22, 247),
        ("hqresize", 26, 86),
    ],
)
def test_resize_family_anisotropic_coverage_mirrors_isotropic_matrix(
    operator, basic_rows, advanced_rows
):
    config = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == operator
    }
    tier_rows = {
        tier: sum(
            _entry_row_count(entry)
            for entry in config.values()
            if entry["tier"] == tier
        )
        for tier in ("basic", "advanced")
    }
    assert tier_rows == {"basic": basic_rows, "advanced": advanced_rows}

    isotropic_signatures = set()
    anisotropic_cases = {}
    for key, entry in config.items():
        for case_key in expected_case_keys_for_entry(key, entry):
            axes = dict(parse_case_key(case_key)[1])
            resize_type = axes["resizeType"]
            signature = _resize_coverage_signature(operator, axes)

            if resize_type in _RESIZE_ANISOTROPIC_TARGETS:
                isotropic_signatures.add(
                    (_RESIZE_ANISOTROPIC_TARGETS[resize_type], signature)
                )
            elif resize_type in _RESIZE_ANISOTROPIC_TARGETS.values():
                assert axes["shape"].split("x")[1:] == (
                    _RESIZE_ANISOTROPIC_INPUT_SPATIAL_SHAPES[resize_type]
                )
                semantic_case = (resize_type, signature)
                assert semantic_case not in anisotropic_cases, (
                    f"duplicate {operator} anisotropic semantic case: "
                    f"{semantic_case} in {anisotropic_cases[semantic_case][0]} and {key}"
                )
                anisotropic_cases[semantic_case] = (key, entry["tier"])

    existing_basic = _RESIZE_EXISTING_BASIC_ANISOTROPIC_SIGNATURES[operator]
    assert {
        semantic_case
        for semantic_case, (_, tier) in anisotropic_cases.items()
        if tier == "basic"
    } == existing_basic

    # Some pre-existing basic profiles (for example anisotropic EXPAND
    # VarShape) intentionally have no isotropic source. They remain in the
    # matrix, while every projected case that was added by this extension is
    # advanced-only.
    assert set(anisotropic_cases) == isotropic_signatures | existing_basic
    for semantic_case in isotropic_signatures - existing_basic:
        assert anisotropic_cases[semantic_case][1] == "advanced"

    target_configs = {
        key: entry
        for key, entry in config.items()
        if any(
            resize_type in _RESIZE_ANISOTROPIC_TARGETS.values()
            for resize_type in entry["string_axes"]["resizeType"]
        )
    }
    assert fake_planar_pairing_issues(target_configs) == ()

    layout_batches = {}
    for key, entry in target_configs.items():
        for case_key in expected_case_keys_for_entry(key, entry):
            _, case_axes = parse_case_key(case_key)
            axes = dict(case_axes)
            layout = axes["layout"]
            if layout not in {"NHWC", "NCHW", "NCHW_FAKE"}:
                continue
            shape = axes["shape"].split("x")
            identity = tuple(
                sorted(
                    (name, value)
                    for name, value in case_axes
                    if name not in {"layout", "shape"}
                )
            ) + (("spatialShape", "x".join(shape[1:])),)
            layout_batches.setdefault(identity, {})[layout] = int(shape[0])

    for identity, batches in layout_batches.items():
        if "NHWC" in batches and {"NCHW", "NCHW_FAKE"} & batches.keys():
            assert (
                len(set(batches.values())) == 1
            ), f"{operator} layout batch mismatch for {dict(identity)}: {batches}"


@pytest.mark.parametrize(
    "operator",
    [
        "autocontrast",
        "bndbox",
        "boxblur",
        "brightnesscontrast",
        "centercrop",
        "clahe",
        "convertto",
        "copymakeborder",
        "cvtcolor",
        "erase",
        "flip",
        "gaussiannoise",
        "inpaint",
        "invert",
    ],
)
def test_priority_operator_fake_planar_rows_have_native_twins(operator):
    config = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == operator
    }

    assert any(
        layout in {"NCHW_FAKE", "CHW_FAKE"}
        for entry in config.values()
        for layout in _entry_string_axis_values(entry, "layout")
    )
    assert fake_planar_pairing_issues(config) == ()


def test_reformat_single_call_calibration_case_keys_are_stable():
    config = load_bench_config()

    expected = {
        "reformat_uint8_pitched_fallback_advanced": (
            "reformat_uint8_pitched_fallback_advanced"
            "[InOutDataType=uint8][shape=512x480x640]"
            "[inputKind=Tensor][rowAlignment=256]"
        ),
        "reformat_uint8_pitched_large_image_advanced": (
            "reformat_uint8_pitched_large_image_advanced"
            "[InOutDataType=uint8][shape=128x1080x1920]"
            "[inputKind=Tensor][rowAlignment=256]"
        ),
    }

    for config_key, case_key in expected.items():
        entry = config[config_key]
        assert expected_case_keys_for_entry(config_key, entry) == [case_key]
        assert "timed_repetitions" not in entry.get("metadata", {})

    assert "reformat_uint8_pitched_fullhd_advanced" not in config


def test_brightnesscontrast_benchmark_signatures_are_canonical_and_unique():
    config = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == "brightnesscontrast"
    }
    assert {
        "brightnesscontrast_rgb_f32_1080p_advanced",
        "brightnesscontrast_planar_rgb_f32_nchw_1080p_advanced",
    } <= config.keys()

    signatures = {}
    for key, entry in config.items():
        for case in expected_case_keys_for_entry(key, entry):
            _, axes = parse_case_key(case)
            signature = (entry["tier"], tuple(sorted(axes)))
            assert signature not in signatures, (
                f"{case} duplicates canonical benchmark signature from "
                f"{signatures.get(signature)}"
            )
            signatures[signature] = case


def test_parse_args_rejects_unknown_config_key(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--config-key", "definitely_not_a_config_key"],
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2


def test_parse_args_accepts_exact_operator_names(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--operator", "resize,gaussian"],
    )

    args = run_bench.parse_args()

    assert args.operators == ["resize", "gaussian"]


def test_parse_args_forwards_arguments_after_explicit_bench_folder(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", ".", "--", "--axis", "shape=1x2x3"],
    )

    args = run_bench.parse_args()

    assert args.bench_folder == "."
    assert args.bench_args[-2:] == ["--axis", "shape=1x2x3"]


def test_parse_args_defaults_to_csv_output(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_bench.py"])

    args = run_bench.parse_args()

    assert args.output.endswith("bench_output.csv")


def test_parse_args_uses_shared_quality_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_bench.py"])

    args = run_bench.parse_args()

    assert args.max_noise_pct == DEFAULT_BENCHMARK_QUALITY.max_noise_pct
    assert args.max_perf_diff_pct == DEFAULT_BENCHMARK_QUALITY.max_perf_diff_pct
    assert args.max_perf_diff_us == DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us


def test_parse_args_accepts_warmup_cap(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--warmup-cap", "50"])

    args = run_bench.parse_args()

    assert args.warmup_cap == 50


@pytest.mark.parametrize("value", ["-1", "abc", "1.5"])
def test_parse_args_rejects_invalid_warmup_cap(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--warmup-cap", value])

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2


def test_parse_args_uses_csv_output_for_custom_config_file(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_json(
        config_dir / "bench_params_smoke.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                }
            }
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--config-file", str(config_dir / "bench_params_smoke.json")],
    )

    args = run_bench.parse_args()

    assert args.output.endswith("bench_output_smoke.csv")


def test_parse_args_accepts_keep_outputs_and_json_output(monkeypatch, tmp_path):
    out = tmp_path / "bench_output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bench.py",
            "--operator",
            "resize",
            "--keep-outputs",
            "--output",
            str(out),
        ],
    )

    args = run_bench.parse_args()

    assert args.output == str(out)
    assert args.keep_outputs is True


def test_parse_args_rejects_unknown_operator(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--operator", "res"],
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2


def test_deprecated_benchmarks_arg_uses_exact_operator_names(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--benchmarks", "resize"],
    )

    args = run_bench.parse_args()

    assert args.operators == ["resize"]


def test_baseline_payload_from_dataframe_emits_raw_baselines_json(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    index = load_config_index(config_dir / "operators")
    df = pd.DataFrame(
        [
            _baseline_row(language="cpp"),
            _baseline_row(
                language="python", gpu_us=535.7935, noise_us=3.88, bwutil=0.41
            ),
        ]
    )

    payload = baseline_payload_from_dataframe(
        df,
        index=index,
        sku_map_path=config_dir / "sku_map.json",
        source=tmp_path / "bench_output.json",
    )

    case = (
        "resize_contract_area_tensor_uchar3_basic"
        "[InOutDataType=uchar3]"
        "[shape=32x1080x1920]"
        "[kernelSize=5x5]"
        "[resizeType=CONTRACT]"
        "[interpolation=AREA]"
        "[layout=NHWC]"
        "[inputKind=Tensor]"
    )
    assert payload == {
        case: {
            "A100_PCIE_40GB_250W_1095MHz": {
                "n_runs": 1,
                "gpu_time_us_cpp": 525.7525,
                "gpu_time_us_python": 535.7935,
                "gpu_noise_us_cpp": 1.2,
                "gpu_noise_us_python": 3.88,
                "gpu_bwutil_cpp": 0.42,
                "gpu_bwutil_python": 0.41,
            }
        }
    }


def test_baseline_payload_from_dataframe_allows_single_language(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    index = load_config_index(config_dir / "operators")

    payload = baseline_payload_from_dataframe(
        pd.DataFrame([_baseline_row(language="cpp")]),
        index=index,
        sku_map_path=config_dir / "sku_map.json",
        source=tmp_path / "bench_output.json",
    )

    metrics = next(iter(next(iter(payload.values())).values()))
    assert metrics == {
        "n_runs": 1,
        "gpu_time_us_cpp": 525.7525,
        "gpu_noise_us_cpp": 1.2,
        "gpu_bwutil_cpp": 0.42,
    }


def test_baseline_payload_from_dataframe_rejects_missing_ambiguous_axis(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    config_path = config_dir / "operators" / "resize.json"
    config = json.loads(config_path.read_text())
    config["configs"]["resize_contract_area_tensor_uchar3_basic"]["string_axes"][
        "kernelSize"
    ] = ["3x3", "5x5"]
    _write_json(config_path, config)

    index = load_config_index(config_dir / "operators")

    with pytest.raises(BaselineError, match="missing axis column 'kernelSize'"):
        baseline_payload_from_dataframe(
            pd.DataFrame([_baseline_row(language="cpp")]),
            index=index,
            sku_map_path=config_dir / "sku_map.json",
            source=tmp_path / "bench_output.json",
        )


def test_baseline_config_paths_uses_run_manifest(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    _write_json(
        config_dir / "bench_params.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                }
            }
        },
    )

    paths, sku_map_path = run_bench._baseline_config_paths(
        str(config_dir / "bench_params.json")
    )

    assert paths == [(config_dir / "operators" / "resize.json").resolve()]
    assert sku_map_path == config_dir / "sku_map.json"


def test_write_output_json_writes_raw_payload(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    _write_json(
        config_dir / "bench_params.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                }
            }
        },
    )
    out = tmp_path / "bench_baselines.json"
    args = SimpleNamespace(
        config_file=str(config_dir / "bench_params.json"),
        output=str(out),
    )
    df = pd.DataFrame(
        [
            _baseline_row(language="cpp"),
            _baseline_row(
                language="python", gpu_us=535.7935, noise_us=3.88, bwutil=0.41
            ),
        ]
    )

    run_bench._write_output(args, df)

    case = next(iter(json.loads(out.read_text())))
    assert case == (
        "resize_contract_area_tensor_uchar3_basic"
        "[InOutDataType=uchar3]"
        "[shape=32x1080x1920]"
        "[kernelSize=5x5]"
        "[resizeType=CONTRACT]"
        "[interpolation=AREA]"
        "[layout=NHWC]"
        "[inputKind=Tensor]"
    )


def test_write_output_csv_writes_legacy_combined_csv(tmp_path):
    out = tmp_path / "bench_output.csv"
    args = SimpleNamespace(
        output=str(out),
        max_noise_pct=DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        max_perf_diff_pct=DEFAULT_BENCHMARK_QUALITY.max_perf_diff_pct,
        max_perf_diff_us=DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us,
    )
    df = pd.DataFrame(
        [
            {
                **_baseline_row(gpu_us=525.75251, noise_us=1.234),
                "Device": 0,
                "GPU Noise (%)": 0.23456,
                "CPU Time (µs)": 42.4242,
                "CPU Noise (%)": 0.1234,
                "CPU Noise (µs)": 0.5678,
                "Py overhead (%)": 1.2345,
                "Py overhead (µs)": 2.3456,
            }
        ]
    )

    run_bench._write_output(args, df)

    written = pd.read_csv(out)
    assert "Device" not in written.columns
    assert written.loc[0, "config_key"] == "resize_contract_area_tensor_uchar3_basic"
    assert written.loc[0, "GPU Time (µs)"] == pytest.approx(525.75)
    assert written.loc[0, "BWUtil"] == pytest.approx(0.42)


def test_discover_operator_pairs_uses_manifest_operator_names(tmp_path):
    for name in ("resize", "gaussian"):
        _touch(tmp_path / f"bench_{name}")
        _touch(tmp_path / f"bench_{name}.py")

    config = {
        "resize_basic": {
            "benchmark": "resize",
            "tier": "basic",
        },
        "gaussian_basic": {
            "benchmark": "gaussian",
            "tier": "basic",
        },
    }

    cpp_runner = DummyRunner(
        tmp_path,
        config,
        config_keys=None,
        language="cpp",
        operators=["gaussian", "resize"],
        tiers={"basic"},
    )
    python_runner = DummyRunner(
        tmp_path,
        config,
        config_keys=None,
        language="python",
        operators=["gaussian", "resize"],
        tiers={"basic"},
    )

    pairs = run_bench.discover_operator_pairs(cpp_runner, python_runner)

    assert [
        (key, os.path.basename(cpp), os.path.basename(py)) for key, cpp, py in pairs
    ] == [
        ("gaussian_basic", "bench_gaussian", "bench_gaussian.py"),
        ("resize_basic", "bench_resize", "bench_resize.py"),
    ]


def test_discover_operator_pairs_with_config_keys_supports_mixed_operators(tmp_path):
    for name in ("resize", "gaussian"):
        _touch(tmp_path / f"bench_{name}")
        _touch(tmp_path / f"bench_{name}.py")

    config = {
        "resize_advanced": {
            "benchmark": "resize",
            "tier": "advanced",
        },
        "gaussian_advanced": {
            "benchmark": "gaussian",
            "tier": "advanced",
        },
    }
    config_keys = ["resize_advanced", "gaussian_advanced"]

    cpp_runner = DummyRunner(
        tmp_path,
        config,
        config_keys,
        language="cpp",
    )
    python_runner = DummyRunner(
        tmp_path,
        config,
        config_keys,
        language="python",
    )

    pairs = run_bench.discover_operator_pairs(
        cpp_runner, python_runner, config_keys=config_keys
    )

    assert [
        (key, os.path.basename(cpp), os.path.basename(py)) for key, cpp, py in pairs
    ] == [
        ("resize_advanced", "bench_resize", "bench_resize.py"),
        ("gaussian_advanced", "bench_gaussian", "bench_gaussian.py"),
    ]
