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
"""Contract + unit tests for tools/refactor_op.py.

Contract tests run the CLI against the live tree (real operators); helper tests load the module
directly to exercise the deterministic primitives. No GPU/build is required.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TOOL = REPO / "tools" / "refactor_op.py"

VALID_STATUS = {"PASS", "GAP", "N-A", "MANUAL", "RECOMMENDATION"}


def run(*args, timeout=60):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _module():
    spec = importlib.util.spec_from_file_location("refactor_op", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- CLI contract
def test_runs_and_emits_a_report():
    r = run("BrightnessContrast", "--format", "json")
    data = json.loads(r.stdout)
    assert data["operator"] == "BrightnessContrast"
    assert data["phase"] == "assess"
    assert data["findings"]


def test_report_is_deterministic():
    for args in (
        ("Flip", "--format", "json"),
        ("Flip", "--phase", "verify", "--format", "json"),
    ):
        a = run(*args)
        b = run(*args)
        assert a.stdout == b.stdout, f"{args} not byte-identical"


def test_assess_default_exit_is_zero():
    # refactoring is advisory: an assess run never fails a gate.
    assert run("Flip").returncode == 0
    assert run("BrightnessContrast").returncode == 0


def test_exit_code_matches_gap_contract():
    for args in (("Flip",), ("Flip", "--phase", "verify")):
        r = run(*args, "--format", "json")
        data = json.loads(json.dumps(json.loads(r.stdout)))  # round-trip = valid json
        has_gap = any(f["status"] == "GAP" for f in data["findings"])
        assert (r.returncode == 1) == has_gap
        assert r.returncode in (0, 1)


def test_verify_emits_refactor_summary():
    data = json.loads(run("Normalize", "--phase", "verify", "--format", "json").stdout)
    s = data["summary"]
    assert {
        "loc_net",
        "loc_insertions",
        "loc_deletions",
        "redundancy_resolved",
        "redundancy_introduced",
        "redundancy_open_now",
    } <= set(s)
    assert isinstance(s["redundancy_resolved"], list)
    assert isinstance(s["loc_net"], int)


def test_assess_has_no_summary():
    data = json.loads(run("Flip", "--format", "json").stdout)
    assert "summary" not in data  # the impact summary is a verify-phase artifact


def test_status_vocabulary_is_valid():
    for phase in ("assess", "verify"):
        data = json.loads(run("Resize", "--phase", phase, "--format", "json").stdout)
        assert all(f["status"] in VALID_STATUS for f in data["findings"])


def test_domain_filter_scopes_output():
    data = json.loads(
        run("BrightnessContrast", "--domain", "impl", "--format", "json").stdout
    )
    assert data["findings"]
    assert all(f["domain"] == "impl" for f in data["findings"])


def test_unknown_operator_degrades_gracefully():
    r = run("NotARealOperator", "--domain", "impl")
    assert r.returncode == 2
    assert "could not be resolved" in r.stderr
    assert "Traceback" not in r.stderr


def test_invalid_domain_is_a_usage_error():
    assert run("Flip", "--domain", "bogus").returncode == 2


def test_invalid_phase_is_a_usage_error():
    assert run("Flip", "--phase", "bogus").returncode == 2


def test_findings_carry_evidence_or_guideline():
    data = json.loads(run("BrightnessContrast", "--format", "json").stdout)
    for f in data["findings"]:
        if f["status"] != "N-A":
            assert f["evidence"] or f["guideline"], f


# ------------------------------------------------------------------------- helper unit tests
def test_stable_hash_is_process_independent():
    m = _module()
    expected = int.from_bytes(hashlib.blake2b(b"abc", digest_size=8).digest(), "big")
    assert m._stable_hash("abc") == expected  # blake2b, not Python's salted hash()


def test_jaccard_identical_near_distinct():
    m = _module()
    # A realistic kernel-sized body; a near-duplicate differs in only the addressing line
    # (the Tensor-vs-VarShape case), which shifts a small number of shingle windows.
    base = [f"acc{i} = src{i} * w{i} + b{i};" for i in range(30)]
    near = base[:-1] + ["acc29 = tex2D(src, x, y);"]  # one differing line near the edge
    distinct = [f"totally_unrelated_token_{i}();" for i in range(30)]

    def sh(lines):
        return m.shingle_hashes([m.normalize_line(x) for x in lines])

    assert m.jaccard(sh(base), sh(base)) == 1.0
    assert m.jaccard(sh(base), sh(near)) >= m.SIM_THRESHOLD
    assert m.jaccard(sh(base), sh(distinct)) < 0.3


def test_extract_blocks_finds_namespaced_function():
    m = _module()
    src = """
namespace cvcuda { namespace priv {
__global__ void Kernel(int *p, int n) {
    int i = threadIdx.x;
    p[i] = i * 2;
    p[i] += 1;
    p[i] *= 3;
    p[i] -= 4;
    p[i] /= 5;
}
}}  // namespace
"""
    blocks = m.extract_blocks(src, "cpp")
    names = [b.name for b in blocks]
    assert "Kernel" in names


def test_extract_blocks_skips_namespace_and_control():
    m = _module()
    src = """
namespace x {
void f(int n) {
    if (n > 0) {
        n = n + 1;
        n = n + 2;
        n = n + 3;
        n = n + 4;
        n = n + 5;
        n = n + 6;
    }
}
}
"""
    names = [b.name for b in m.extract_blocks(src, "cpp")]
    assert "f" in names
    assert "if" not in names and "x" not in names


def test_extract_blocks_skips_if_constexpr():
    """`if constexpr (...)` must not read as a function named "constexpr" — the nested
    block would pair with its enclosing function as a false near-duplicate (regression:
    OpHQResizeKernel.cuh's TryRunDirectLinear)."""
    m = _module()
    src = """
bool g(int n) {
    if constexpr (Supported<int, float>()) {
        n = n + 1;
        n = n + 2;
        n = n + 3;
        n = n + 4;
        n = n + 5;
        n = n + 6;
    }
    return n > 0;
}
"""
    names = [b.name for b in m.extract_blocks(src, "cpp")]
    assert "g" in names
    assert "constexpr" not in names


def test_hqresize_impl_has_no_nested_block_false_duplicates():
    """Endpoint regression: the HQResize kernel headers contain `if constexpr` bodies
    that must not surface as RED-1 near-duplicates of their enclosing functions."""
    r = run("HQResize", "--domain", "impl", "--format", "json")
    data = json.loads(r.stdout)
    red1 = [f for f in data["findings"] if f["id"] == "RED-1"]
    assert red1, "RED-1 must be reported for HQResize"
    assert all("constexpr" not in f["summary"] for f in red1)


def test_near_duplicate_pairs_threshold():
    m = _module()
    body = [f"step{i}();" for i in range(10)]

    def blk(name, body):
        norm = [m.normalize_line(x) for x in body]
        return m.Block(name, 1, 10, norm, m.shingle_hashes(norm))

    a, b = blk("a", body), blk("b", list(body))
    assert m.near_duplicate_pairs([a, b], 0.80)  # identical -> reported
    c = blk("c", [f"other{i}();" for i in range(10)])
    assert not m.near_duplicate_pairs([a, c], 0.80)  # distinct -> not reported


def test_feature_sig_parity():
    m = _module()
    hdr_a = "Limitations:\n * Input:\n * Data Layout: [kNHWC, kHWC]\n * Channels: [3]\n"
    hdr_b = "Limitations:\n * Input:\n * Data Layout: [kNHWC, kHWC]\n * Channels: [3]\n"
    hdr_c = "Limitations:\n * Input:\n * Data Layout: [kNHWC]\n * Channels: [3]\n"
    assert m._feature_sig(hdr_a) == m._feature_sig(hdr_b)
    assert m._feature_sig(hdr_a) != m._feature_sig(hdr_c)

    hdr_out_a = (
        "Limitations:\n"
        " * Input:\n"
        " * Data Layout: [kNHWC]\n"
        " * Channels: [3]\n"
        " * Output:\n"
        " * Data Layout: [kNHWC]\n"
        " * Channels: [3]\n"
    )
    hdr_out_b = (
        "Limitations:\n"
        " * Input:\n"
        " * Data Layout: [kNHWC]\n"
        " * Channels: [3]\n"
        " * Output:\n"
        " * Data Layout: [kNHWC]\n"
        " * Channels: [3]\n"
    )
    hdr_out_c = (
        "Limitations:\n"
        " * Input:\n"
        " * Data Layout: [kNHWC]\n"
        " * Channels: [3]\n"
        " * Output:\n"
        " * Data Layout: [kNHWC]\n"
        " * Channels: [1, 3]\n"
    )
    assert m._feature_sig(hdr_out_a) == m._feature_sig(hdr_out_b)
    assert m._feature_sig(hdr_out_a) != m._feature_sig(hdr_out_c)


def test_coverage_sig_detects_dropped_test():
    m = _module()
    a = "TEST_P(OpFooBar, works) {}\nTEST(OpFooBar, neg) {}\n"
    b = "TEST_P(OpFooBar, works) {}\n"  # a test was removed
    assert m._coverage_sig(a) != m._coverage_sig(b)
    assert m._coverage_sig(a) == m._coverage_sig(a)


def test_resolve_op_attributes_legacy_kernels_to_the_owning_op():
    """Gaussian must resolve its shared filter kernels, not GaussianNoise's legacy files
    (regression: the old prefix glob claimed gaussian_noise*.cu for Gaussian)."""
    m = _module()
    priv = {
        p.relative_to(m.REPO / "src/cvcuda/priv").as_posix()
        for p in m.resolve_op("Gaussian").priv
    }
    assert "legacy/filter.cu" in priv
    assert "legacy/filter_var_shape.cu" in priv
    assert not any("gaussian_noise" in p for p in priv)


def _mock_binding_only_diff(m, monkeypatch, baseline_source, candidate_source):
    paths = m.resolve_op("HQResize")

    def fake_git(*args):
        if m.rel(paths.pybind) in args:
            return "binding implementation diff"
        return ""

    original_read = m.read
    monkeypatch.setattr(m, "git", fake_git)
    monkeypatch.setattr(
        m,
        "git_show",
        lambda _base, path: baseline_source if path == m.rel(paths.pybind) else None,
    )
    monkeypatch.setattr(
        m,
        "read",
        lambda path: candidate_source if path == paths.pybind else original_read(path),
    )
    return m._api_abi(paths, "base", "guide")


def test_verify_api_allows_binding_implementation_only_diff(monkeypatch):
    m = _module()
    source = (REPO / "python/mod_cvcuda/operators/OpHQResize.cpp").read_text()
    baseline = source.replace(
        "class BatchShapesHelper",
        "// baseline-only implementation marker\nclass BatchShapesHelper",
        1,
    )

    finding = _mock_binding_only_diff(m, monkeypatch, baseline, source)

    assert finding.status == m.PASS


def test_verify_api_rejects_reachable_binding_alias_change(monkeypatch):
    m = _module()
    baseline = (REPO / "python/mod_cvcuda/operators/OpHQResize.cpp").read_text()
    candidate = baseline.replace(
        "using Roi  = pybind11::tuple;", "using Roi  = pybind11::list;", 1
    )

    finding = _mock_binding_only_diff(m, monkeypatch, baseline, candidate)

    assert finding.status == m.GAP


def test_verify_api_marks_unparseable_binding_surface_manual(monkeypatch):
    m = _module()
    baseline = (REPO / "python/mod_cvcuda/operators/OpHQResize.cpp").read_text()
    candidate = baseline.replace(
        "using Roi  = pybind11::tuple;",
        "using Roi  = decltype([] { return pybind11::tuple{}; }());",
        1,
    )

    finding = _mock_binding_only_diff(m, monkeypatch, baseline, candidate)

    assert finding.status == m.MANUAL
