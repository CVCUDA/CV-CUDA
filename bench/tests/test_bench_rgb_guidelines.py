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

"""Static guideline checks for the RGB benchmark-config standardization.

Each test maps to a rule (R1..R11) from the MR guidelines. These scan the
operator JSON configs and the category manifest only (no GPU / build needed);
timing rules (R7, R9) and parity (R12) are verified by local benchmark runs.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_OPS_DIR = os.path.join(_BENCH, "config", "operators")
_MANIFEST = os.path.join(_BENCH, "config", "operator_categories.json")

TWO_GB = 2**31  # TensorWrap32 uses int32 byte offsets

_BASE_BYTES = {
    "uint8": 1,
    "int8": 1,
    "uchar": 1,
    "char": 1,
    "uint16": 2,
    "int16": 2,
    "ushort": 2,
    "short": 2,
    "uint32": 4,
    "int32": 4,
    "uint": 4,
    "int": 4,
    "float32": 4,
    "float": 4,
    "float64": 8,
    "double": 8,
}


# A vector dtype is a base name + a single channel digit (uchar3, float3, short2),
# distinct from a scalar whose trailing digits are a bit width (float32, uint16).
_VEC = re.compile(r"[a-z](\d)$", re.IGNORECASE)


def _channels(dtype: str) -> int:
    m = _VEC.search(dtype)
    if m and int(m.group(1)) in (2, 3, 4):
        return int(m.group(1))
    return 1


def _base(dtype: str) -> str:
    return dtype[:-1] if _channels(dtype) > 1 else dtype


def _dtype_bytes(dtype: str) -> int:
    return _BASE_BYTES[_base(dtype)] * _channels(dtype)


def _is_scalar(dtype: str) -> bool:
    return _channels(dtype) == 1


def _is_rgb(dtype: str) -> bool:
    return _channels(dtype) == 3


def _is_rgba(dtype: str) -> bool:
    return _channels(dtype) == 4


def _load_manifest() -> dict:
    return json.load(open(_MANIFEST))["operators"]


def _op_files() -> dict:
    return {
        os.path.basename(f)[:-5]: f for f in glob.glob(os.path.join(_OPS_DIR, "*.json"))
    }


def _profiles():
    """Yield (op, profile_key, body) for every profile in every operator config."""
    for op, f in sorted(_op_files().items()):
        raw = json.load(open(f))
        profiles = raw["configs"] if "configs" in raw else raw
        for key, body in profiles.items():
            yield op, key, body


def _shapes(body) -> list:
    return body.get("string_axes", {}).get("shape", [])


def _batch(shape: str):
    head = shape.split("x")[0]
    return int(head) if head.lstrip("-").isdigit() else None


def _is_round(n: int) -> bool:
    """Powers of 2, multiples of 4, or small values (<=4)."""
    return n <= 4 or (n & (n - 1)) == 0 or n % 4 == 0


def _max_tensor_elems(body, shape: str):
    """Largest input/output tensor represented by a resize profile."""
    try:
        dims = [int(x) for x in shape.split("x")]
    except ValueError:
        return None

    input_elems = math.prod(dims)
    if len(dims) != 3:
        return input_elems

    batch, height, width = dims
    resize_types = body.get("string_axes", {}).get("resizeType", [])
    output_elems = [input_elems]
    for resize_type in resize_types:
        if resize_type == "EXPAND":
            output_elems.append(batch * (height * 2) * (width * 2))
        elif resize_type.startswith("TARGET_"):
            target_height, target_width = (
                int(dim) for dim in resize_type.removeprefix("TARGET_").split("x")
            )
            output_elems.append(batch * target_height * target_width)
    return max(output_elems)


CATS = _load_manifest()
CAT_A = {op for op, m in CATS.items() if m["category"] == "A"}
CAT_B = {op for op, m in CATS.items() if m["category"] == "B"}


# ---------------------------------------------------------------------------
# R1 — every operator classified A/B/C; manifest matches the config dir.
# ---------------------------------------------------------------------------
def test_r1_every_operator_classified():
    on_disk = set(_op_files())
    classified = set(CATS)
    assert on_disk == classified, (
        f"unclassified ops: {sorted(on_disk - classified)}; "
        f"stale manifest entries: {sorted(classified - on_disk)}"
    )
    assert all(m["category"] in {"A", "B", "C"} for m in CATS.values())


# ---------------------------------------------------------------------------
# R2 / R5 — Category-A basic tier is RGB-only (uchar3/float3); no scalar, no RGBA.
# ---------------------------------------------------------------------------
def test_r2_cat_a_basic_is_rgb_only():
    bad = []
    for op, key, body in _profiles():
        if (
            op in CAT_A
            and not CATS[op].get("rgb_unsupported")
            and body.get("tier") == "basic"
        ):
            non_rgb = [dt for dt in body.get("dtypes", []) if not _is_rgb(dt)]
            if non_rgb:
                bad.append(f"{key}: {non_rgb}")
    assert (
        not bad
    ), "Cat-A basic profiles must be RGB-only (uchar3/float3):\n  " + "\n  ".join(bad)


# ---------------------------------------------------------------------------
# R3 — Category-A single-channel profiles live only in the advanced tier.
# ---------------------------------------------------------------------------
def test_r3_cat_a_single_channel_only_in_advanced():
    bad = []
    for op, key, body in _profiles():
        if (
            op in CAT_A
            and not CATS[op].get("rgb_unsupported")
            and body.get("tier") != "advanced"
        ):
            scalar = [dt for dt in body.get("dtypes", []) if _is_scalar(dt)]
            if scalar:
                bad.append(f"{key} (tier={body.get('tier')}): {scalar}")
    assert (
        not bad
    ), "Cat-A single-channel profiles must be in the advanced tier:\n  " + "\n  ".join(
        bad
    )


# ---------------------------------------------------------------------------
# R4 — Category-A ops with basic RGB coverage carry an RGBA mirror in advanced.
# ---------------------------------------------------------------------------
def test_r4_cat_a_has_advanced_rgba_mirror():
    missing = []
    for op in sorted(CAT_A):
        if CATS[op].get("rgba_unsupported") or CATS[op].get("rgb_unsupported"):
            continue  # op's kernel has no 4-channel (or no multi-channel) instantiation
        if op not in _op_files():
            continue
        raw = json.load(open(_op_files()[op]))
        body_by_key = raw["configs"] if "configs" in raw else raw
        has_basic_rgb = any(
            b.get("tier") == "basic" and any(_is_rgb(d) for d in b.get("dtypes", []))
            for b in body_by_key.values()
        )
        has_adv_rgba = any(
            b.get("tier") == "advanced"
            and any(_is_rgba(d) for d in b.get("dtypes", []))
            for b in body_by_key.values()
        )
        if has_basic_rgb and not has_adv_rgba:
            missing.append(op)
    assert (
        not missing
    ), "Cat-A ops with basic RGB but no advanced RGBA mirror: " + ", ".join(missing)


# ---------------------------------------------------------------------------
# R6 — batch sizes are round numbers / powers of two (all operators).
# ---------------------------------------------------------------------------
def test_r6_batch_sizes_are_round():
    bad = []
    for op, key, body in _profiles():
        if op not in CAT_A:  # only the operators this MR re-tunes
            continue
        for shape in _shapes(body):
            n = _batch(shape)
            if n is not None and not _is_round(n):
                bad.append(f"{key}: N={n} ({shape})")
    assert not bad, (
        "Cat-A batch sizes must be round (pow2 / multiple of 4 / <=4):\n  "
        + "\n  ".join(bad)
    )


# ---------------------------------------------------------------------------
# R8 — no tensor exceeds 2 GB (TensorWrap32), input or EXPAND output.
# ---------------------------------------------------------------------------
def test_r8_no_tensor_exceeds_2gb():
    bad = []
    for op, key, body in _profiles():
        channels = max(body.get("int64_axes", {}).get("numChannels", [1]))
        for dt in body.get("dtypes", []):
            for shape in _shapes(body):
                elems = _max_tensor_elems(body, shape)
                if elems is None:
                    continue
                nbytes = elems * _dtype_bytes(dt) * channels
                if nbytes >= TWO_GB:
                    bad.append(f"{key}: {dt} {shape} = {nbytes/1e9:.2f} GB")
    assert not bad, "Configs exceeding the 2 GB TensorWrap32 limit:\n  " + "\n  ".join(
        bad
    )


# ---------------------------------------------------------------------------
# R10 — inputKind is the string container selector everywhere. Operator-specific
# API variants extend it here: TensorBatch, TensorScalar, and ScalarGamma are
# reserved for operators with distinct execution pathways.
# ---------------------------------------------------------------------------
def test_r10_input_kind_is_valid():
    allowed = (
        "Tensor",
        "VarShape",
        "TensorBatch",
        "TensorScalar",
        "ScalarGamma",
    )
    bad = []
    for op, key, body in _profiles():
        kinds = body.get("string_axes", {}).get("inputKind", [])
        invalid = [v for v in kinds if v not in allowed]
        if op != "hqresize" and "TensorBatch" in kinds:
            invalid.append("TensorBatch (only HQResize supports it)")
        if op != "normalize" and "TensorScalar" in kinds:
            invalid.append("TensorScalar (only Normalize supports it)")
        if op != "gammacontrast" and "ScalarGamma" in kinds:
            invalid.append("ScalarGamma (only GammaContrast supports it)")
        if invalid:
            bad.append(f"{key}: {invalid}")
    assert not bad, "inputKind has unsupported values:\n  " + "\n  ".join(bad)
