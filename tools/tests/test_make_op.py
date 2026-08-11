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
"""Contract/unit tests for the deterministic ``/make-op`` checker (tools/make_op.py).

Two layers, both stdlib-only and decoupled from any single operator's live coverage state:
  1. CLI contract tests (subprocess): phases run, determinism, GAP/exit-code, graceful
     degradation, and the --bare spec-delegation behaviour.
  2. Unit tests of the COV "teeth" helpers (dtype canonicalisation, FMT channel inference,
     dtype-token detection) imported directly.

Wired into CI via the bench Python unit-test step, which runs ``pytest`` over ``tools/tests/``.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "make_op.py"
sys.path.insert(0, str(REPO / "tools"))

import make_op  # noqa: E402

VALID_STATUS = {"PASS", "GAP", "N-A", "MANUAL", "RECOMMENDATION"}


def run(*args):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=180,
    )


# --------------------------------------------------------------------------- CLI contract
def test_scaffold_runs_and_emits_a_report():
    r = run("Flip", "--phase", "scaffold", "--format", "json")
    assert r.stdout.strip(), "expected a non-empty report"
    data = json.loads(r.stdout)
    assert data["operator"] == "Flip" and data["phase"] == "scaffold"
    assert data["findings"], "expected at least one finding"


def test_done_runs_and_emits_a_report():
    r = run("Flip", "--phase", "done", "--format", "json")
    data = json.loads(r.stdout)
    assert data["phase"] == "done"
    ids = {f["id"] for f in data["findings"]}
    # composition: make-op groups + reused review_op domains are all present
    assert {
        "SPEC-BRIEF",
        "SCF-1",
        "COV-GOLD",
        "COV-1",
        "DOC-REL",
        "EXEC-1",
        "RDY-1",
    } <= ids
    assert {"SUP-1", "TST-1", "BEN-1", "DOC-1"} <= ids  # reused review_op findings


def test_bench_driver_stub_is_gated():
    """BEN-DRV: a real operator's bench drivers (no TODO(make-op) stub) must PASS."""
    r = run("Flip", "--phase", "done", "--format", "json")
    bd = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "BEN-DRV"]
    assert bd and bd[0]["status"] == "PASS"


def test_scaffold_report_is_deterministic():
    """Static-core guarantee: same tree -> byte-identical scaffold report."""
    a = run("Flip", "--phase", "scaffold", "--format", "json")
    b = run("Flip", "--phase", "scaffold", "--format", "json")
    assert a.stdout == b.stdout


def test_exit_code_matches_gap_contract():
    r = run("Flip", "--phase", "scaffold", "--format", "json")
    findings = json.loads(r.stdout)["findings"]
    assert all(f["status"] in VALID_STATUS for f in findings)
    has_gap = any(f["status"] == "GAP" for f in findings)
    assert r.returncode in (0, 1)
    assert (r.returncode == 1) == has_gap


def test_findings_carry_evidence_and_guideline():
    for f in json.loads(run("Flip", "--phase", "done", "--format", "json").stdout)[
        "findings"
    ]:
        assert f["id"] and f["domain"] and f["status"]
        if f["status"] != "N-A":
            assert f.get("evidence") or f.get("guideline")


def test_unknown_operator_degrades_gracefully():
    """A bogus operator must not crash; scaffold must GAP on the missing public header."""
    r = run("NotARealMakeOp", "--phase", "scaffold", "--format", "json")
    scf1 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "SCF-1"]
    assert scf1 and scf1[0]["status"] == "GAP"


def test_bare_mode_delegates_spec_as_manual():
    """--bare: the SPEC items must be MANUAL (spec delegated), never GAP."""
    r = run("NotARealMakeOp", "--phase", "scaffold", "--bare", "--format", "json")
    spec = [
        f
        for f in json.loads(r.stdout)["findings"]
        if f["id"].startswith("SPEC-") and f["id"] != "SPEC-CORRECT"
    ]
    assert spec, "expected SPEC findings"
    assert all(f["status"] == "MANUAL" for f in spec)


def test_spec_gated_without_bare():
    """Without --bare, an unauthored stub header GAPs SPEC-BRIEF/ORACLE."""
    r = run("NotARealMakeOp", "--phase", "scaffold", "--format", "json")
    ids = {f["id"]: f["status"] for f in json.loads(r.stdout)["findings"]}
    assert ids.get("SPEC-ORACLE") == "GAP"


# --------------------------------------------------------------------------- COV unit tests
def test_bench_dtype_canon():
    cases = {
        "uchar3": "u8",
        "uchar4": "u8",
        "uint8": "u8",
        "float3": "f32",
        "float4": "f32",
        "float32": "f32",
        "float16": "f16",
        "half": "f16",
        "ushort": "u16",
        "uint16": "u16",
        "short3": "s16",
        "double": "f64",
        "float64": "f64",
    }
    for name, canon in cases.items():
        assert make_op.bench_dtype_canon(name) == canon, name


def test_fmt_channels_inference():
    assert make_op.fmt_channels("FMT_RGB8 FMT_RGBA8 FMT_U8") == {1, 3, 4}
    assert make_op.fmt_channels("nvcv::TYPE_F32") == {1}
    assert 4 in make_op.fmt_channels("FMT_RGBAf32")
    assert 3 in make_op.fmt_channels("FMT_BGRf32")


def test_dtype_token_detection():
    import re

    txt = "ValueList { FMT_RGB8, FMT_RGBAf32, FMT_F16 }"
    assert re.search(make_op.DTYPE_TEST_RE["u8"], txt)
    assert re.search(make_op.DTYPE_TEST_RE["f32"], txt)
    assert re.search(make_op.DTYPE_TEST_RE["f16"], txt)
    # a dtype with no token present is (correctly) not detected
    assert not re.search(make_op.DTYPE_TEST_RE["f64"], txt)
