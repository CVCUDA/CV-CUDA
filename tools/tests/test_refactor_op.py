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


# ------------------------------------------------------- RED-2 (cross-operator duplication)
def test_red2_is_always_emitted_and_advisory():
    """RED-2 reports on any operator with priv sources, and never fails the gate. Deliberately
    does not assert a particular operator's clone list: those change as shared headers land, and
    pinning them here would make every future hoist look like a test regression."""
    r = run("Invert", "--domain", "impl", "--format", "json")
    red2 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "RED-2"]
    assert red2, "RED-2 must be reported for an operator with priv sources"
    assert all(f["status"] in {"PASS", "RECOMMENDATION"} for f in red2)
    assert all(f["domain"] == "impl" for f in red2)
    assert r.returncode == 0


def test_all_operators_rollup_matches_per_operator_runs():
    """The roll-up must agree with the per-operator reports it replaces, or agents will keep
    hand-rolling shell loops that disagree with the tool."""
    data = json.loads(
        run("--all-operators", "--domain", "impl", "--format", "json").stdout
    )
    rows = {r["Op"]: r for r in data["operators"]}
    assert len(rows) > 40
    for op in ("Invert", "Posterize"):
        single = json.loads(run(op, "--domain", "impl", "--format", "json").stdout)
        by_id = {}
        for f in single["findings"]:
            if f["status"] in {"RECOMMENDATION", "MANUAL"}:
                by_id[f["id"]] = by_id.get(f["id"], 0) + 1
        assert rows[op]["by_id"] == by_id
        assert rows[op]["open"] == sum(by_id.values())


def test_if_conditions_reads_a_wrapped_guard_whole():
    """RED-3 reports what a guard *excludes*, so splitting a wrapped disjunction across lines
    produces a false positive — the second width looks absent. Parse the balanced condition.
    """
    m = _module()
    src = """
void f() {
    if constexpr (sizeof(BT) == 1 ||
                  sizeof(BT) == 4)
    { g(); }
}
"""
    conds = m.if_conditions(m._strip_code(src))
    assert conds, "the guard must be found"
    joined = " ".join(conds[0][1].split())
    assert joined == "sizeof(BT) == 1 || sizeof(BT) == 4"


def test_red12_ignores_commented_out_code():
    """A commented-out helper is not a live reimplementation."""
    m = _module()
    shape = m.REINVENTION_SHAPES[0]["rx"]
    commented = "// struct FakeVec4Type { using type = uchar4; };\n"
    assert shape.findall(commented), "sanity: the shape matches the raw text"
    assert not shape.findall(m._strip_code(commented))


def test_variants_and_hoist_reject_a_positional_operator():
    """They scan the whole corpus, so a positional operator would read as scoping the run."""
    assert run("Invert", "--variants", "Foo").returncode == 2
    assert run("Invert", "--simulate-hoist", "Foo").returncode == 2


def test_all_operators_rejects_incompatible_flags():
    assert run("--all-operators", "Invert").returncode == 2
    assert run("--all-operators", "--phase", "verify").returncode == 2


def test_variants_groups_definitions_into_equivalence_classes():
    """`--variants` answers 'how many genuinely different versions of this helper exist' — the
    question that gates any hoist. Asserted structurally, not against a fixed count, so landing
    a hoist does not fail the test."""
    data = json.loads(
        run("--variants", "ValidateSrcDstTensors", "--format", "json").stdout
    )
    assert data["block"] == "ValidateSrcDstTensors"
    assert len(data["variants"]) >= 2, "this helper is known to have divergent versions"
    members = [m for v in data["variants"] for m in v["members"]]
    assert len(members) == len(
        set(members)
    ), "a definition may only belong to one class"
    assert all(v["lines"] > 0 and v["members"] for v in data["variants"])

    empty = json.loads(run("--variants", "NoSuchBlockName", "--format", "json").stdout)
    assert empty["variants"] == []


def test_simulate_hoist_predicts_the_red2_delta():
    """Hoisting a name must never increase any operator's RED-2 count, and must strictly reduce
    it for at least one — otherwise the simulation is not modelling the change at all.
    """
    data = json.loads(
        run(
            "--simulate-hoist",
            "ValidateSrcDstTensors,ValidateSrcDstVarBatch",
            "--format",
            "json",
        ).stdout
    )
    assert data["hoisted"] == ["ValidateSrcDstTensors", "ValidateSrcDstVarBatch"]
    rows = data["operators"]
    assert rows, "these helpers are duplicated, so some operator must be affected"
    assert all(r["after"] <= r["before"] for r in rows)
    assert any(r["after"] < r["before"] for r in rows)
    assert all(len(r["residual"]) == r["after"] for r in rows)

    none = json.loads(
        run("--simulate-hoist", "NoSuchBlockName", "--format", "json").stdout
    )
    assert all(r["after"] == r["before"] for r in none["operators"])


def test_min_block_lines_sweep_is_monotonic_and_bounded():
    """The floor is swept downwards while planning a refactor, so a smaller floor must only ever
    add leads. It must also stay opt-in: the default run is unchanged."""

    # Key on the *local* block (the `path:line` before " vs "), not the whole evidence string.
    # The rest of that string names the chosen partner and a `(+N more)` count, and both may
    # legitimately move as a lower floor adds blocks to the comparison pool — while the same
    # local block stays exactly as valid a lead, which is what this test is about.
    def leads(data):
        return {
            f["evidence"].split(" vs ", 1)[0]
            for f in data["findings"]
            if f["id"] == "RED-2" and f["status"] == "RECOMMENDATION"
        }

    counts = {}
    for n in (6, 4, 2):
        counts[n] = leads(
            json.loads(
                run(
                    "Invert",
                    "--domain",
                    "impl",
                    "--min-block-lines",
                    str(n),
                    "--format",
                    "json",
                ).stdout
            )
        )
    assert counts[6] <= counts[4] <= counts[2]
    assert len(counts[2]) > len(
        counts[6]
    ), "floor 2 must surface small helpers floor 6 hides"

    default = json.loads(run("Invert", "--domain", "impl", "--format", "json").stdout)
    assert counts[6] == leads(default)


def test_min_block_lines_below_the_shingle_window_still_fingerprints():
    """Sweeping under SHINGLE_K must keep short bodies matchable — shingle_hashes falls back to
    hashing the body whole. Deleting that branch as 'dead code' would silently make every
    sub-window block unmatchable, defeating the sweep."""
    m = _module()
    short = [m.normalize_line(x) for x in ["return a + b;", "// noop"]]
    assert len(short) < m.SHINGLE_K
    assert m.shingle_hashes(short), "a sub-window body must still produce a fingerprint"
    assert m.jaccard(m.shingle_hashes(short), m.shingle_hashes(list(short))) == 1.0
    other = [m.normalize_line("return a - b;")]
    assert m.jaccard(m.shingle_hashes(short), m.shingle_hashes(other)) == 0.0


def test_invalid_min_block_lines_is_a_usage_error():
    assert run("Invert", "--min-block-lines", "0").returncode == 2


def test_cross_op_blocks_excludes_the_operators_own_files():
    """The invariant that keeps RED-2 from matching an operator against itself. Without it a
    shared header registered in SHARED_KERNEL_SOURCES would pair with its own consumers and the
    finding could never be resolved."""
    m = _module()
    P = m.resolve_op("Invert")
    mine = {p.resolve() for p in P.priv}
    assert mine, "Invert must resolve to at least one priv file"
    assert not [p for p, _ in m.cross_op_blocks(P, m.read) if p.resolve() in mine]


def test_cross_op_duplicates_finds_only_cross_file_twins():
    m = _module()

    def blk(name, body):
        norm = [m.normalize_line(x) for x in body]
        return m.Block(name, 1, len(body), norm, m.shingle_hashes(norm))

    body = [f"step{i}();" for i in range(10)]
    mine = [blk("Helper", body)]
    twin = [(Path("src/cvcuda/priv/OpOther.cu"), blk("Helper", list(body)))]
    other = [
        (
            Path("src/cvcuda/priv/OpOther.cu"),
            blk("Other", [f"x{i}();" for i in range(10)]),
        )
    ]

    assert "Helper" in m.cross_op_duplicates(mine, twin, 0.80)
    assert not m.cross_op_duplicates(mine, other, 0.80)
    # allowlisted names are filtered by the caller, exactly as RED-1 does
    assert not m.cross_op_duplicates(
        [b for b in mine if b.name not in {"Helper"}], twin, 0.80
    )


def test_cross_op_scan_honours_the_injected_reader():
    """RED-2 must read the corpus through the reader it is given, not the working tree.

    refactor_summary's base pass supplies a git_show reader. If the corpus were read directly
    instead, a refactor that hoists several operators at once would see the siblings already
    deduplicated while assessing <base>, report nothing there, and so have nothing to resolve —
    a silent failure visible only in the verify summary.
    """
    m = _module()
    P = m.resolve_op("Invert")
    assert m.cross_op_blocks(P, m.read), "sanity: the real reader finds blocks"
    assert m.cross_op_blocks(P, lambda _p: None) == []


def test_open_red_counts_tracks_partial_resolution(monkeypatch):
    """An id can be legitimately partially resolved, which a set difference over ids cannot
    express. This is what `redundancy_reduced` reports."""
    m = _module()
    P = m.resolve_op("Invert")

    def counted(n):
        return [m.Finding("RED-2", "impl", m.REC, f"dup {i}") for i in range(n)]

    monkeypatch.setattr(m, "run_assess", lambda P, d, c, r=None: counted(3))
    base = m._open_red_counts(P, {}, m.read)
    monkeypatch.setattr(m, "run_assess", lambda P, d, c, r=None: counted(1))
    now = m._open_red_counts(P, {}, m.read)

    assert base == {"RED-2": 3} and now == {"RED-2": 1}
    # still open, so `redundancy_resolved` (a set difference) would report nothing
    assert set(base) - set(now) == set()
    assert now["RED-2"] < base["RED-2"]
