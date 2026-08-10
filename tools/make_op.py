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
"""Deterministic checker for adding a NEW CV-CUDA operator (the /make-op skill).

Implements the checklist in .agents/guidance/MAKE_OP_GUIDELINES.md. It does not author the
operator (that is the agent/human + tools/mkop/mkop.sh); it gates completeness, wiring, and
regression rigor against the operator's declared contract (its C-API header). Two phases:

  --phase scaffold : the wired skeleton — SPEC (approved contract) + SCF (files/wiring) + a
                     best-effort IMP stub scan. Run right after mkop.sh.
  --phase done     : the deterministic final regression checklist — composes review_op.py (all
                     four domains) + optimize_op.py preflight (RDY), then adds the make-op
                     teeth: IMP (no stubs), COV (declared-matrix mirror + gold + bit-exact +
                     parity + complement-negatives), DOC-REL (relnote), EXEC (--run tests pass).

Composes, does not duplicate: review_op.py and optimize_op.py are imported / invoked. The static
checks are read-only and deterministic (no network, clocks, randomness); only --run executes.

Usage:
  python3 tools/make_op.py <Operator> --phase scaffold|done
          [--bare] [--format md|json] [--out PATH] [--run]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import review_op  # noqa: E402  (sibling module, same tools/ dir)

MAKE_OP_GUIDE = ".agents/guidance/MAKE_OP_GUIDELINES.md"
PASS, GAP, NA, MANUAL, REC = (
    review_op.PASS,
    review_op.GAP,
    review_op.NA,
    review_op.MANUAL,
    review_op.REC,
)
Finding = review_op.Finding
read, rel, grep, first_evidence = (
    review_op.read,
    review_op.rel,
    review_op.grep,
    review_op.first_evidence,
)

# Phase-group ordering for rendering.
SCAFFOLD_DOMAINS = ("spec", "scaffold")
DONE_DOMAINS = (
    "spec",
    "scaffold",
    "implementation",
    "coverage",
    "support",
    "test",
    "bench",
    "docs",
    "execution",
    "readiness",
)

# canonical dtype -> regex detecting it in a C++ test (FMT_* / nvcv::TYPE_* / type-list tokens).
# Only run for *declared* dtypes, so unrelated tokens (e.g. a TYPE_S32 flip-code) never mislead.
DTYPE_TEST_RE = {
    # The unsigned FMT_* patterns use a negative lookbehind so a signed token (FMT_S8 / FMT_S16)
    # is NOT misread as unsigned coverage; the signed rows match the FMT_*S8 / FMT_*S16 forms too.
    "u8": r"FMT_[A-Za-z]*(?<![Ss])8\b|TYPE_U8\b|\bU8\b|\buchar\b|\buint8\b",
    "s8": r"FMT_[A-Za-z]*S8\b|TYPE_S8\b|\bS8\b|\bschar\b|\bint8\b",
    "u16": r"FMT_[A-Za-z]*(?<![Ss])16\b|TYPE_U16\b|\bU16\b|\bushort\b|\buint16\b",
    "s16": r"FMT_[A-Za-z]*S16\b|TYPE_S16\b|\bS16\b|\bshort\b|\bint16\b",
    "u32": r"TYPE_U32\b|\bU32\b|\buint32\b",
    "s32": r"TYPE_S32\b|\bS32\b|\bint32\b",
    "f16": r"[Ff]16\b|TYPE_F16\b|\bhalf\b",
    "f32": r"[Ff]32\b|TYPE_F32\b|\bfloat\b",
    "f64": r"[Ff]64\b|TYPE_F64\b|\bdouble\b",
}


def bench_dtype_canon(name):
    """Map a bench dtype name (uchar3 / float32 / ushort …) to a canonical token."""
    n = name.lower()
    # Longest / most specific stems first so uint8 != uint, float16 != float, etc.
    table = [
        ("uchar", "u8"),
        ("uint8", "u8"),
        ("schar", "s8"),
        ("int8", "s8"),
        ("ushort", "u16"),
        ("uint16", "u16"),
        ("short", "s16"),
        ("int16", "s16"),
        ("uint32", "u32"),
        ("uint", "u32"),
        ("int32", "s32"),
        ("int", "s32"),
        ("half", "f16"),
        ("float16", "f16"),
        ("double", "f64"),
        ("float64", "f64"),
        ("float", "f32"),
    ]
    for stem, canon in table:
        if n.startswith(stem):
            return canon
    return None


def fmt_channels(text):
    """Channel counts exercised in a C++ test, inferred from FMT_*/TYPE_* tokens."""
    chans = set()
    for tok in re.findall(r"FMT_[A-Za-z0-9]+", text or ""):
        if "RGBA" in tok or "BGRA" in tok:
            chans.add(4)
        elif "RGB" in tok or "BGR" in tok:
            chans.add(3)
        else:
            chans.add(1)
    if re.search(r"nvcv::TYPE_[A-Z0-9]+|\bTYPE_[A-Z0-9]+", text or ""):
        chans.add(1)
    return chans


# --------------------------------------------------------------------- extra operator paths
def extra_paths(P):
    """Scaffold paths not on review_op.OpPaths."""
    return {
        "capi_impl": REPO / f"src/cvcuda/Op{P.Op}.cpp",
        "priv_hpp": REPO / f"src/cvcuda/priv/Op{P.Op}.hpp",
        "priv_cpp": REPO / f"src/cvcuda/priv/Op{P.Op}.cpp",
        "priv_cu": REPO / f"src/cvcuda/priv/Op{P.Op}.cu",
        "cml_lib": REPO / "src/cvcuda/CMakeLists.txt",
        "cml_priv": REPO / "src/cvcuda/priv/CMakeLists.txt",
        "cml_test": REPO / "tests/cvcuda/system/CMakeLists.txt",
        "cml_py": REPO / "python/mod_cvcuda/CMakeLists.txt",
        "py_main": REPO / "python/mod_cvcuda/Main.cpp",
        "py_ops_hpp": REPO / "python/mod_cvcuda/operators/Operators.hpp",
        "cml_bench_cpp": REPO / "bench/cpp/CMakeLists.txt",
        "cml_bench_py": REPO / "bench/python/CMakeLists.txt",
        "bench_params": REPO / "bench/config/bench_params.json",
        "cmakelists": REPO / "CMakeLists.txt",
    }


def latest_relnote():
    """Path to the relnote whose version matches CMakeLists.txt's project VERSION."""
    cml = read(REPO / "CMakeLists.txt") or ""
    m = re.search(r"^\s+VERSION\s+(\d+\.\d+\.\d+)", cml, re.M)
    if not m:
        return None
    ver = m.group(1)
    cands = sorted((REPO / "docs/sphinx/relnotes").glob(f"v{ver}-*.rst"))
    if not cands:
        cands = sorted((REPO / "docs/sphinx/relnotes").glob(f"v{ver}.rst"))
    return cands[0] if cands else None


def exists_f(p):
    return p.exists()


# ============================================================================= SPEC domain
STUB_BRIEF = "Defines types and functions to handle the"


def check_spec(P, bare):
    out = []
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#spec / Part 0"
    htext = read(P.header) or ""

    def spec_finding(fid, ok, summary_ok, summary_bad, evidence, fix):
        if bare:
            return Finding(
                fid,
                "spec",
                MANUAL,
                summary_bad + " (spec delegated; --bare)",
                evidence,
                g,
            )
        return Finding(
            fid,
            "spec",
            PASS if ok else GAP,
            summary_ok if ok else summary_bad,
            evidence,
            g,
            "" if ok else fix,
        )

    # SPEC-BRIEF
    brief_stub = STUB_BRIEF in htext or "TBD args" in htext
    out.append(
        spec_finding(
            "SPEC-BRIEF",
            bool(htext) and not brief_stub,
            "Header @brief authored (real semantics)",
            "Header @brief is still the mkop stub",
            f"{rel(P.header)}",
            "Author the @brief with the operator's real semantics (Part 0).",
        )
    )

    # SPEC-ORACLE
    oracle = bool(
        re.search(
            r"Reference:|\bmatches\b|\bmimics\b|oracle waived|no external reference|custom operator",
            htext,
            re.I,
        )
    )
    out.append(
        spec_finding(
            "SPEC-ORACLE",
            oracle,
            "Reference oracle cited (or explicitly waived)",
            "No reference-oracle citation/waiver in the header",
            f"{rel(P.header)}",
            "Cite a reference oracle (e.g. 'Reference: torchvision invert') or record a waiver.",
        )
    )

    # SPEC-MATRIX
    lim = review_op.parse_limitations(htext)
    # the Limitations region still carrying TODO placeholders means it is not authored
    lim_region = ""
    m = re.search(r"Limitations:(.*?)\*/", htext, re.S)
    if m:
        lim_region = m.group(1)
    matrix_ok = lim is not None and "TODO" not in lim_region
    out.append(
        spec_finding(
            "SPEC-MATRIX",
            matrix_ok,
            "Limitations matrix declared (no TODO rows)",
            "Limitations matrix not declared / still has TODO rows",
            f"{rel(P.header)}",
            "Fill the Doxygen Limitations table with the approved support matrix.",
        )
    )

    # SPEC-CORRECT — always human
    out.append(
        Finding(
            "SPEC-CORRECT",
            "spec",
            MANUAL,
            "Declared semantics match the cited oracle; gold implements it independently",
            "review against the cited Reference oracle",
            g,
        )
    )
    return out


# ============================================================================ SCAFFOLD domain
def check_scaffold(P, X):
    out = []
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#scaffold / make_operator.rst"

    def present(fid, path, desc, fix):
        ok = path.exists()
        return Finding(
            fid,
            "scaffold",
            PASS if ok else GAP,
            desc + (" present" if ok else " MISSING"),
            rel(path) + ("" if ok else " (missing)"),
            g,
            "" if ok else fix,
        )

    out.append(
        present("SCF-1", P.header, "Public C API header", "Run tools/mkop/mkop.sh.")
    )
    out.append(present("SCF-2", P.hpp, "Public C++ header", "Run mkop.sh."))
    out.append(present("SCF-3", X["capi_impl"], "C API implementation", "Run mkop.sh."))
    # SCF-4 priv impl: .cpp or .cu (author's choice)
    priv_ok = X["priv_cpp"].exists() or X["priv_cu"].exists()
    out.append(
        Finding(
            "SCF-4",
            "scaffold",
            PASS if priv_ok else GAP,
            "Private implementation present (.cpp or .cu)"
            if priv_ok
            else "Private implementation MISSING",
            rel(X["priv_cu"]) if X["priv_cu"].exists() else rel(X["priv_cpp"]),
            g,
            "" if priv_ok else "Run mkop.sh.",
        )
    )
    out.append(present("SCF-5", X["priv_hpp"], "Private header", "Run mkop.sh."))
    out.append(present("SCF-6", P.test_cpp, "C++ system test", "Run mkop.sh."))
    out.append(
        present(
            "SCF-7",
            P.pybind,
            "Python binding (under operators/)",
            "Run mkop.sh (binding lives in operators/).",
        )
    )
    out.append(present("SCF-8", P.test_py, "Python test", "Run mkop.sh."))
    # SCF-9 bench files
    bench_ok = P.bench_cpp.exists() and P.bench_py.exists() and P.bench_cfg.exists()
    miss = [rel(p) for p in (P.bench_cpp, P.bench_py, P.bench_cfg) if not p.exists()]
    out.append(
        Finding(
            "SCF-9",
            "scaffold",
            PASS if bench_ok else GAP,
            "Bench C++/Python/config present"
            if bench_ok
            else "Bench files MISSING: " + ", ".join(miss),
            "bench cpp/py/json present" if bench_ok else ", ".join(miss),
            g,
            "" if bench_ok else "Run mkop.sh (emits bench stubs).",
        )
    )

    # wiring
    def wired(fid, path, pattern, desc, fix):
        txt = read(path)
        ok = bool(grep(pattern, txt))
        return Finding(
            fid,
            "scaffold",
            PASS if ok else GAP,
            desc + (" wired" if ok else " NOT wired"),
            f"{rel(path)}: /{pattern}/" + ("" if ok else " (not found)"),
            g,
            "" if ok else fix,
        )

    # SCF-10 lib/priv/test cmake
    lib = bool(grep(rf"Op{re.escape(P.Op)}\.cpp", read(X["cml_lib"])))
    privc = bool(grep(rf"Op{re.escape(P.Op)}\.c(pp|u)", read(X["cml_priv"])))
    testc = bool(grep(rf"TestOp{re.escape(P.Op)}\.cpp", read(X["cml_test"])))
    scf10_ok = lib and privc and testc
    out.append(
        Finding(
            "SCF-10",
            "scaffold",
            PASS if scf10_ok else GAP,
            "Lib/priv/test CMake wiring present"
            if scf10_ok
            else "Lib/priv/test CMake wiring incomplete",
            f"lib={lib} priv={privc} test={testc}",
            g,
            "" if scf10_ok else "Ensure Op/TestOp entries are in the three CMakeLists.",
        )
    )
    # SCF-11 python module wiring
    pm = bool(grep(rf"ExportOp{re.escape(P.Op)}\b", read(X["py_main"])))
    ph = bool(grep(rf"ExportOp{re.escape(P.Op)}\b", read(X["py_ops_hpp"])))
    pc = bool(grep(rf"operators/Op{re.escape(P.Op)}\.cpp", read(X["cml_py"])))
    scf11_ok = pm and ph and pc
    out.append(
        Finding(
            "SCF-11",
            "scaffold",
            PASS if scf11_ok else GAP,
            "Python module wiring present"
            if scf11_ok
            else "Python module wiring incomplete",
            f"Main={pm} Operators.hpp={ph} CMake(operators/)={pc}",
            g,
            ""
            if scf11_ok
            else "Wire ExportOp into Main.cpp + Operators.hpp + operators/ in CMakeLists.",
        )
    )
    # SCF-12 bench wiring
    bc = bool(grep(rf"ops/Bench{re.escape(P.Op)}\.cpp", read(X["cml_bench_cpp"])))
    bp = bool(grep(rf"ops/bench_{re.escape(P.op)}\.py", read(X["cml_bench_py"])))
    manifest = review_op.load_bench_cfg(X["bench_params"]) or {}
    bm = P.op in (manifest.get("operators") or {})
    scf12_ok = bc and bp and bm
    out.append(
        Finding(
            "SCF-12",
            "scaffold",
            PASS if scf12_ok else GAP,
            "Bench wiring present" if scf12_ok else "Bench wiring incomplete",
            f"cpp_cmake={bc} py_cmake={bp} manifest={bm}",
            g,
            ""
            if scf12_ok
            else "Register bench cpp/py + add the bench_params.json manifest entry.",
        )
    )
    # SCF-13 docs rows
    oplist = read(REPO / "docs/sphinx/operator_list.rst") or ""
    ops = read(REPO / "docs/sphinx/modules/python/operators.rst") or ""
    rel_row = bool(grep(rf":py:func:`cvcuda\.{re.escape(P.pyname)}`", oplist))
    af = bool(grep(rf"cvcuda-autofunction::\s*cvcuda\.{re.escape(P.pyname)}\b", ops))
    af_into = bool(
        grep(rf"cvcuda-autofunction::\s*cvcuda\.{re.escape(P.pyname)}_into\b", ops)
    )
    relnote = latest_relnote()
    rn = bool(
        relnote
        and re.search(
            rf"``{re.escape(P.Op)}``|\b{re.escape(P.Op)}\b", read(relnote) or ""
        )
    )
    scf13_ok = rel_row and af and af_into and rn
    out.append(
        Finding(
            "SCF-13",
            "scaffold",
            PASS if scf13_ok else GAP,
            "Docs rows present" if scf13_ok else "Docs rows incomplete",
            f"operator_list={rel_row} autofunction={af} autofunction_into={af_into} relnote={rn}",
            g,
            ""
            if scf13_ok
            else "Add operator_list row + autofunction directives + latest-relnote bullet.",
        )
    )
    # SCF-14 SPDX
    spdx_missing = []
    for p in (
        P.header,
        P.hpp,
        X["capi_impl"],
        X["priv_hpp"],
        P.test_cpp,
        P.pybind,
        P.test_py,
        P.bench_cpp,
        P.bench_py,
    ):
        if p.suffix == ".json":
            continue
        t = read(p)
        if t is not None and "SPDX-License-Identifier" not in t[:600]:
            spdx_missing.append(rel(p))
    out.append(
        Finding(
            "SCF-14",
            "scaffold",
            PASS if not spdx_missing else GAP,
            "SPDX headers present"
            if not spdx_missing
            else "SPDX missing: " + ", ".join(spdx_missing[:4]),
            "all checked files carry SPDX"
            if not spdx_missing
            else ", ".join(spdx_missing),
            "AGENTS.md",
            "" if not spdx_missing else "Add the SPDX 2026 header.",
        )
    )
    return out


# ====================================================================== IMPLEMENTATION domain
STUB_MARKERS = (
    r"\bTODO\b|t\.fail\(|\bno-?op\b|std::generate\(goldVec|Test failed intentionally"
)


def check_impl(P, X):
    out = []
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#implementation"
    priv = X["priv_cu"] if X["priv_cu"].exists() else X["priv_cpp"]
    blobs = {
        "priv": read(priv),
        "capi": read(X["capi_impl"]),
        "test_cpp": read(P.test_cpp),
        "test_py": read(P.test_py),
        "pybind": read(P.pybind),
    }
    stub_hits = []
    for name, txt in blobs.items():
        for ln, line in grep(STUB_MARKERS, txt):
            stub_hits.append(f"{name}:{ln}")
    out.append(
        Finding(
            "IMP-1",
            "implementation",
            PASS if not stub_hits else GAP,
            "No stub markers remain"
            if not stub_hits
            else f"{len(stub_hits)} stub marker(s) remain",
            ", ".join(stub_hits[:6])
            if stub_hits
            else "no TODO/t.fail/noop/placeholder markers",
            g,
            ""
            if not stub_hits
            else "Implement the operator + replace the stub test (remove TODO/t.fail/placeholder gold).",
        )
    )

    # IMP-2 Limitations no TODO (shares SPEC-MATRIX intent)
    htext = read(P.header) or ""
    m = re.search(r"Limitations:(.*?)\*/", htext, re.S)
    has_todo = bool(m and "TODO" in m.group(1))
    out.append(
        Finding(
            "IMP-2",
            "implementation",
            GAP if (has_todo or not m) else PASS,
            "Limitations table filled"
            if (m and not has_todo)
            else "Limitations table missing/has TODO rows",
            rel(P.header),
            g,
            ""
            if (m and not has_todo)
            else "Fill the Limitations table (no TODO rows).",
        )
    )

    # IMP-3 multi-GPU safety if device alloc
    priv_txt = blobs["priv"] or ""
    if re.search(r"\bcudaMalloc\b", priv_txt):
        pdr = bool(re.search(r"PerDeviceResource", priv_txt))
        out.append(
            Finding(
                "IMP-3",
                "implementation",
                PASS if pdr else GAP,
                "Device allocation wrapped in PerDeviceResource"
                if pdr
                else "cudaMalloc without PerDeviceResource (multi-GPU unsafe)",
                rel(priv),
                "make_operator.rst#multi_gpu",
                ""
                if pdr
                else "Wrap device allocations in PerDeviceResource<> (see make_operator.rst / OpCLAHE).",
            )
        )
    else:
        out.append(
            Finding(
                "IMP-3",
                "implementation",
                NA,
                "No device allocation -> multi-GPU wrapper N-A",
                "",
                g,
            )
        )

    # NVTX-1 always-on NVTX submit marker present, per submit entry. The marker test
    # (tests/cvcuda/python/test_nvtx_markers.py) has no registry to edit: it scans every Op*.h
    # submit declaration for a CVCUDA_DEFINE_API definition whose body opens with the matching
    # CVCUDA_NVTX_RANGE (and every Python binding for its NvtxTrace). Mirror the per-submit
    # source check here so a GAP surfaces before that test fails.
    htext_nvtx = read(P.header) or ""
    capi_txt = blobs["capi"] or ""
    declared_submits = re.findall(
        rf"CVCUDA_PUBLIC\s+NVCVStatus\s+(cvcuda{re.escape(P.Op)}\w*Submit)\s*\(",
        htext_nvtx,
    )

    def _submit_definition_marked(name):
        # Definition-scoped, like the test: every CVCUDA_DEFINE_API definition of this
        # submit must open its body with the matching range as the first statement. A
        # marker elsewhere in the file (or a missing definition) does not count.
        definitions = list(
            re.finditer(
                rf"CVCUDA_DEFINE_API\(\s*[^,]+,\s*[^,]+,\s*NVCVStatus\s*,\s*{re.escape(name)}\s*,",
                capi_txt,
            )
        )
        entry_range = re.compile(
            rf'\s*CVCUDA_NVTX_RANGE\(\s*"{re.escape(name)}"\s*\)\s*;'
        )
        if not definitions:
            return False
        for match in definitions:
            body_start = capi_txt.find("{", match.end())
            if body_start < 0 or not entry_range.match(capi_txt, body_start + 1):
                return False
        return True

    unmarked_submits = [
        name for name in declared_submits if not _submit_definition_marked(name)
    ]
    marker_hits = grep(
        rf'CVCUDA_NVTX_RANGE\("cvcuda{re.escape(P.Op)}\w*Submit"', blobs["capi"]
    )
    nvtx_ok = bool(declared_submits) and not unmarked_submits
    if nvtx_ok:
        nvtx_msg = f"all {len(declared_submits)} submit entries carry their matching NVTX range"
        nvtx_ev = first_evidence(X["capi_impl"], marker_hits)
    elif not declared_submits:
        nvtx_msg = f"no cvcuda{P.Op}...Submit declaration found in the public header"
        nvtx_ev = rel(P.header)
    else:
        nvtx_msg = (
            "submit entries missing their matching CVCUDA_NVTX_RANGE: "
            + ", ".join(unmarked_submits)
        )
        nvtx_ev = rel(X["capi_impl"])
    out.append(
        Finding(
            "NVTX-1",
            "implementation",
            PASS if nvtx_ok else GAP,
            nvtx_msg,
            nvtx_ev,
            g,
            ""
            if nvtx_ok
            else (
                f'Open every cvcuda{P.Op}*Submit definition body with CVCUDA_NVTX_RANGE("<exact '
                'submit name>") (the scaffold template emits it); '
                "tests/cvcuda/python/test_nvtx_markers.py enforces the same by scanning sources."
            ),
        )
    )
    return out


# ============================================================================ COVERAGE domain
def check_coverage(P, support_info):
    out = []
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#coverage"
    t = read(P.test_cpp) or ""
    lim = support_info.get("limitations") or {}
    declared_dtypes = lim.get("dtypes") or set()
    declared_channels = lim.get("channels") or set()
    has_vs = support_info.get("has_vs")
    planar_declared = support_info.get("planar_declared")
    planar_not_applicable = support_info.get("planar_not_applicable", False)
    planar_reason = support_info.get("planar_reason", "")

    # COV-GOLD
    gold = grep(r"Gold|gold|[Rr]eference|naive|CPU[Rr]ef|RefImpl", t)
    out.append(
        Finding(
            "COV-GOLD",
            "coverage",
            PASS if gold else GAP,
            "Independent CPU gold reference present"
            if gold
            else "No CPU gold reference found",
            first_evidence(P.test_cpp, gold) or "no Gold/Reference/naive symbol",
            g,
            ""
            if gold
            else "Add an independent CPU gold reference implementing the cited oracle.",
        )
    )

    # COV-BITEXACT
    near = grep(r"EXPECT_NEAR|ASSERT_NEAR", t)
    eq = grep(r"EXPECT_EQ|ASSERT_EQ", t)
    if near:
        out.append(
            Finding(
                "COV-BITEXACT",
                "coverage",
                GAP,
                f"{len(near)} EXPECT_NEAR/ASSERT_NEAR site(s) - bit-exact is the default",
                first_evidence(P.test_cpp, near),
                g,
                "Make the gold bit-exact and use EXPECT_EQ/ASSERT_EQ; never loosen tolerance silently.",
            )
        )
    else:
        out.append(
            Finding(
                "COV-BITEXACT",
                "coverage",
                PASS if eq else GAP,
                "Bit-exact comparison (EXPECT_EQ, no NEAR)"
                if eq
                else "No bit-exact comparison found",
                first_evidence(P.test_cpp, eq) or "no EXPECT_EQ/ASSERT_EQ",
                g,
                "" if eq else "Compare gold vs result with EXPECT_EQ/ASSERT_EQ.",
            )
        )

    # COV-1 declared dtypes tested
    if not declared_dtypes:
        out.append(
            Finding(
                "COV-1",
                "coverage",
                MANUAL,
                "No declared dtype matrix parsed -> verify dtype coverage",
                rel(P.header),
                g,
            )
        )
    else:
        missing = [
            d
            for d in sorted(declared_dtypes)
            if not re.search(DTYPE_TEST_RE.get(d, d), t)
        ]
        out.append(
            Finding(
                "COV-1",
                "coverage",
                PASS if not missing else GAP,
                "Every declared dtype tested"
                if not missing
                else "Declared dtype(s) not tested: " + ", ".join(missing),
                f"declared={sorted(declared_dtypes)}",
                g,
                ""
                if not missing
                else "Add a positive interleaved test case for each missing dtype.",
            )
        )

    # COV-CHAN declared channels tested
    if not declared_channels:
        out.append(
            Finding(
                "COV-CHAN",
                "coverage",
                MANUAL,
                "No declared channel set parsed -> verify channel coverage",
                rel(P.header),
                g,
            )
        )
    else:
        tested_ch = fmt_channels(t)
        missing_ch = sorted(c for c in declared_channels if c not in tested_ch)
        out.append(
            Finding(
                "COV-CHAN",
                "coverage",
                PASS if not missing_ch else GAP,
                "Every declared channel-count tested"
                if not missing_ch
                else "Declared channel(s) not tested: "
                + ", ".join(map(str, missing_ch)),
                f"declared={sorted(declared_channels)} tested={sorted(tested_ch)}",
                g,
                ""
                if not missing_ch
                else "Add a positive test case covering each missing channel count.",
            )
        )

    # COV-2 declared dtypes benched
    cfg = review_op.load_bench_cfg(P.bench_cfg) or {}
    benched = set()
    for c in (cfg.get("configs") or {}).values():
        for d in c.get("dtypes", []):
            canon = bench_dtype_canon(d)
            if canon:
                benched.add(canon)
    if not declared_dtypes:
        out.append(
            Finding(
                "COV-2",
                "coverage",
                MANUAL,
                "No declared dtype matrix -> verify bench dtype coverage",
                rel(P.bench_cfg),
                g,
            )
        )
    else:
        missing_b = [d for d in sorted(declared_dtypes) if d not in benched]
        out.append(
            Finding(
                "COV-2",
                "coverage",
                PASS if not missing_b else GAP,
                "Every declared dtype benched"
                if not missing_b
                else "Declared dtype(s) not benched: " + ", ".join(missing_b),
                f"benched={sorted(benched)}",
                g,
                ""
                if not missing_b
                else "Add bench config dtypes covering each missing dtype.",
            )
        )

    # BEN-DRV: the bench *drivers* must be implemented, not the mkop NHWC-only stub. The structural
    # BEN-* checks (files/config/baselines) pass on a skip-everything stub, so this guards that the
    # C++ and Python drivers actually exercise the declared layouts/containers (no TODO(make-op) /
    # planar+varshape state.skip stubs).
    stub_drivers = []
    for label, path in (("C++", P.bench_cpp), ("Python", P.bench_py)):
        txt = read(path)
        if txt is None:
            continue  # absence is already a GAP in BEN-1/2
        if re.search(r"TODO\(make-op\)", txt):
            stub_drivers.append(label)
    out.append(
        Finding(
            "BEN-DRV",
            "coverage",
            PASS if not stub_drivers else GAP,
            "Bench drivers implemented (not the NHWC-only stub)"
            if not stub_drivers
            else "Bench driver is still a stub: " + ", ".join(stub_drivers),
            "no TODO(make-op) markers in bench drivers"
            if not stub_drivers
            else f"TODO(make-op) stub in {', '.join(stub_drivers)} driver",
            g,
            ""
            if not stub_drivers
            else "Implement the bench driver(s) to exercise the declared layouts (NHWC/NCHW/"
            "NCHW_FAKE) and containers (Tensor/VarShape); see BenchFlip.cpp / bench_flip.py.",
        )
    )

    # COV-3 containers tested + benched
    tensor_test = bool(grep(rf"tensor_correct|TEST(_P)?\(\s*Op{re.escape(P.Op)}\b", t))
    benched_kinds = set()
    for c in (cfg.get("configs") or {}).values():
        benched_kinds.update(c.get("string_axes", {}).get("inputKind", []))
    missing_c = []
    if not tensor_test:
        missing_c.append("Tensor positive test")
    if "Tensor" not in benched_kinds:
        missing_c.append("Tensor bench")
    if has_vs:
        if not grep(r"[Vv]ar[Ss]hape.*correct|varshape_correct|ImageBatchVarShape", t):
            missing_c.append("VarShape positive test")
        if "VarShape" not in benched_kinds:
            missing_c.append("VarShape bench")
    out.append(
        Finding(
            "COV-3",
            "coverage",
            PASS if not missing_c else GAP,
            "Declared containers tested + benched"
            if not missing_c
            else "Container coverage gaps: " + ", ".join(missing_c),
            f"tensor_test={tensor_test} has_vs={has_vs} benched_kinds={sorted(benched_kinds)}",
            g,
            "" if not missing_c else "Add the missing container test/bench coverage.",
        )
    )

    # COV-PARITY + COV-5 (image layouts are complete by default)
    parity = bool(grep(r"PlanarParityUtils|matches_interleaved", t)) or bool(
        grep(r"[Rr]eformat", t)
        and grep(r"NCHW|planar|Planar", t)
        and grep(r"EXPECT_EQ|ASSERT_EQ", t)
    )
    bench_layouts = set()
    for c in (cfg.get("configs") or {}).values():
        bench_layouts.update(c.get("string_axes", {}).get("layout", []))
    if planar_not_applicable:
        out.append(
            Finding(
                "COV-PARITY",
                "coverage",
                NA,
                "Planar image layouts are not applicable -> parity N-A",
                planar_reason,
                MAKE_OP_GUIDE,
            )
        )
        out.append(
            Finding(
                "COV-5",
                "coverage",
                NA,
                "Planar image layouts are not applicable",
                planar_reason,
                MAKE_OP_GUIDE,
            )
        )
    else:
        out.append(
            Finding(
                "COV-PARITY",
                "coverage",
                PASS if parity else GAP,
                "Equivalent image-layout parity test present"
                if parity
                else "Equivalent image-layout parity test MISSING",
                first_evidence(
                    P.test_cpp, grep(r"PlanarParityUtils|matches_interleaved", t)
                )
                or "no parity helper / reformat+EQ",
                MAKE_OP_GUIDE,
                ""
                if parity
                else "Add bit-exact planar == reformat->interleaved-op->reformat parity.",
            )
        )
        miss5 = []
        if not planar_declared:
            miss5.append("planar not declared in header")
        if not parity:
            miss5.append("no equivalent image-layout parity test")
        if "NCHW" not in bench_layouts:
            miss5.append("no NCHW bench")
        if "NCHW_FAKE" not in bench_layouts:
            miss5.append("no NCHW_FAKE bench")
        out.append(
            Finding(
                "COV-5",
                "coverage",
                PASS if not miss5 else GAP,
                "Image-layout coverage complete (declared+parity+bench)"
                if not miss5
                else "Image-layout coverage incomplete: " + "; ".join(miss5),
                f"declared={planar_declared} parity={parity} layouts={sorted(bench_layouts)}",
                MAKE_OP_GUIDE,
                ""
                if not miss5
                else (
                    "Add complete planar layout coverage or declare in the C-API header why "
                    "planar image layouts are not applicable."
                ),
            )
        )

    # COV-NEG complement
    neg_suite = bool(grep(rf"Op{re.escape(P.Op)}_Negative|_Negative", t))
    neg_err = bool(grep(r"ERROR_INVALID_ARGUMENT", t))
    neg_ok = neg_suite and neg_err
    out.append(
        Finding(
            "COV-NEG",
            "coverage",
            PASS if neg_ok else GAP,
            "Complement negative suite present (rejects unsupported inputs)"
            if neg_ok
            else "Negative/complement coverage MISSING",
            f"negative_suite={neg_suite} error_invalid_argument={neg_err}",
            g,
            ""
            if neg_ok
            else "Add a _Negative suite asserting NVCV_ERROR_INVALID_ARGUMENT for "
            "unsupported dtype/layout/channel + in/out mismatch.",
        )
    )
    out.append(
        Finding(
            "COV-MATRIX",
            "coverage",
            MANUAL,
            "Per-dtype x per-container exhaustiveness (each declared variant has its own bit-exact case)",
            f"inspect {rel(P.test_cpp)}",
            g,
        )
    )

    # DOC-REL relnote
    out.append(check_doc_rel(P))
    return out


def check_doc_rel(P):
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#docs (DOC-REL)"
    relnote = latest_relnote()
    if relnote is None:
        return Finding(
            "DOC-REL",
            "docs",
            MANUAL,
            "Could not resolve the latest relnote from CMakeLists VERSION",
            "",
            g,
        )
    txt = read(relnote) or ""
    hit = re.search(rf"``{re.escape(P.Op)}``|\b{re.escape(P.Op)}\b", txt)
    return Finding(
        "DOC-REL",
        "docs",
        PASS if hit else GAP,
        f"Operator in the latest relnote ({rel(relnote)})"
        if hit
        else f"Operator MISSING from the latest relnote ({rel(relnote)})",
        rel(relnote),
        g,
        "" if hit else f"Add a New-Features bullet for {P.Op} to {rel(relnote)}.",
    )


# =========================================================================== EXECUTION domain
def find_build_dir():
    for name in ("build-rel", "build", "build-debug"):
        d = REPO / name
        if (d / "CMakeCache.txt").exists():
            return d
    for d in sorted(REPO.glob("build*")):
        if (d / "CMakeCache.txt").exists():
            return d
    return None


def check_exec(P, do_run):
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#execution"
    if not do_run:
        return [
            Finding(
                "EXEC-1",
                "execution",
                MANUAL,
                "C++ tests not executed (pass --run to build+run+assert pass)",
                "",
                g,
            ),
            Finding(
                "EXEC-2",
                "execution",
                MANUAL,
                "Python tests not executed (pass --run)",
                "",
                g,
            ),
        ]
    out = []
    bd = find_build_dir()
    if bd is None:
        out.append(
            Finding(
                "EXEC-1",
                "execution",
                GAP,
                "No configured build dir found (build-rel/build); cannot run tests",
                "configure the project first",
                g,
                "Configure & build, then re-run --run; else defer to CI.",
            )
        )
        out.append(
            Finding(
                "EXEC-2",
                "execution",
                GAP,
                "No configured build dir found; cannot run Python tests",
                "",
                g,
                "Build the Python module, then re-run --run; else defer to CI.",
            )
        )
        return out
    # C++: build the system test target and run this op's filter
    try:
        b = subprocess.run(
            ["cmake", "--build", str(bd), "--target", "cvcuda_test_system"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        if b.returncode != 0:
            out.append(
                Finding(
                    "EXEC-1",
                    "execution",
                    GAP,
                    "C++ system-test build failed",
                    (b.stderr or b.stdout)[-300:],
                    g,
                    "Fix the build.",
                )
            )
        else:
            binp = bd / "bin" / "cvcuda_test_system"
            r = subprocess.run(
                [str(binp), f"--gtest_filter=Op{P.Op}*:*{P.Op}*"],
                capture_output=True,
                text=True,
                cwd=str(REPO),
            )
            ok = r.returncode == 0 and "FAILED" not in r.stdout
            out.append(
                Finding(
                    "EXEC-1",
                    "execution",
                    PASS if ok else GAP,
                    "C++ tests pass" if ok else "C++ tests failed",
                    (r.stdout or "")[-300:],
                    g,
                    "" if ok else "Fix failing C++ tests.",
                )
            )
    except (OSError, subprocess.SubprocessError) as e:
        out.append(
            Finding(
                "EXEC-1", "execution", MANUAL, f"Could not run C++ tests ({e})", "", g
            )
        )
    # Python
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(P.test_py), "-q"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        ok = r.returncode == 0
        out.append(
            Finding(
                "EXEC-2",
                "execution",
                PASS if ok else GAP,
                "Python tests pass" if ok else "Python tests failed",
                (r.stdout or "")[-300:],
                g,
                ""
                if ok
                else "Fix failing Python tests (needs the built module on PYTHONPATH).",
            )
        )
    except (OSError, subprocess.SubprocessError) as e:
        out.append(
            Finding(
                "EXEC-2",
                "execution",
                MANUAL,
                f"Could not run Python tests ({e})",
                "",
                g,
            )
        )
    return out


# =========================================================================== READINESS domain
def check_readiness(P, do_run):
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#readiness / .agents/guidance/OPTIMIZATION_GUIDELINES.md"
    opt = Path(__file__).resolve().parent / "optimize_op.py"
    if not opt.exists():
        return [
            Finding(
                "RDY-1",
                "readiness",
                MANUAL,
                "optimize_op.py not found; run preflight manually",
                "",
                g,
            )
        ]
    try:
        r = subprocess.run(
            [
                sys.executable,
                str(opt),
                P.Op,
                "--phase",
                "preflight",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
    except (OSError, subprocess.SubprocessError) as e:
        return [
            Finding(
                "RDY-1",
                "readiness",
                MANUAL,
                f"Could not run optimize_op preflight ({e})",
                "",
                g,
            )
        ]
    gaps = None
    try:
        data = json.loads(r.stdout)
        gaps = data.get("counts", {}).get(GAP, data.get("exit_gap"))
    except (json.JSONDecodeError, AttributeError):
        pass
    ok = r.returncode == 0
    summary = (
        "Optimization-readiness (optimize_op preflight) green"
        if ok
        else "Optimization-readiness not met (optimize_op preflight)"
    )
    evidence = f"preflight exit={r.returncode}" + (
        f", GAPs={gaps}" if gaps is not None else ""
    )
    return [
        Finding(
            "RDY-1",
            "readiness",
            PASS if ok else GAP,
            summary,
            evidence,
            g,
            ""
            if ok
            else "Resolve preflight GAPs (bench coverage, captured baseline [requires CI], profiling).",
        )
    ]


# ================================================================================= driver
def check_bench_guidelines(P):
    """BEN-GUIDE: run the static RGB-benchmark-guideline tests the CI burn-in legs run.

    These (bench/tests/test_bench_rgb_guidelines.py + test_run_bench_config_key.py) are
    static (no GPU/build) and gate the burn-in *before* any benchmark runs, so a config
    that is unclassified in operator_categories.json, mis-tiered (Cat-A scalar in basic),
    or that drifts the tripwire row/key counts fails the baseline regen silently. Running
    them here catches it locally instead of after a ~1 h CI fan-out.
    """
    g = ".agents/guidance/MAKE_OP_GUIDELINES.md#coverage (BEN-GUIDE)"
    tests = [
        REPO / "bench" / "tests" / "test_bench_rgb_guidelines.py",
        REPO / "bench" / "tests" / "test_run_bench_config_key.py",
    ]
    present = [t for t in tests if t.exists()]
    missing = [t for t in tests if not t.exists()]
    if missing:
        return [
            Finding(
                "BEN-GUIDE",
                "coverage",
                GAP,
                "Bench guideline tests missing",
                ", ".join(rel(t) for t in missing),
                g,
                "Restore the missing bench guideline test file(s).",
            )
        ]
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", *[str(t) for t in present], "-q"],
            capture_output=True,
            text=True,
            cwd=str(REPO / "bench"),
        )
    except (OSError, subprocess.SubprocessError) as e:
        return [
            Finding(
                "BEN-GUIDE",
                "coverage",
                MANUAL,
                f"Could not run bench guideline tests ({e})",
                "",
                g,
            )
        ]
    ok = r.returncode == 0
    # Keep evidence deterministic (no pytest timing line) so the static report is byte-identical
    # on re-run: only the "FAILED <nodeid> - <reason>" lines, which carry no clocks.
    failed = "; ".join(
        ln.strip() for ln in (r.stdout or "").splitlines() if ln.startswith("FAILED ")
    )
    return [
        Finding(
            "BEN-GUIDE",
            "coverage",
            PASS if ok else GAP,
            "Bench RGB-guideline tests pass"
            if ok
            else "Bench RGB-guideline tests fail (would fail the CI baseline burn-in)",
            "" if ok else (failed or "see `pytest bench/tests/` output"),
            g,
            ""
            if ok
            else "Classify the op in bench/config/operator_categories.json (A/B/C), make a Cat-A "
            "config RGB-only in basic with single-channel/RGBA in advanced (R2/R3/R4), and bump the "
            "tripwire key/row counts in bench/tests/test_run_bench_config_key.py.",
        )
    ]


def run_scaffold(P, X, bare):
    # Scaffold phase is structural-only: it gates wiring (SCF) + the approved contract (SPEC).
    # Implementation/coverage/bench-guideline checks belong to --phase done, so a freshly
    # scaffolded (not-yet-implemented) op stays scaffold-green with IMP/COV/EXEC outstanding.
    return check_spec(P, bare) + check_scaffold(P, X)


def run_done(P, X, bare, do_run):
    findings = []
    curated = review_op.load_curated()
    sup, support_info = review_op.check_support(P, curated)
    # make-op-specific groups
    findings += check_spec(P, bare)
    findings += check_scaffold(P, X)
    findings += check_impl(P, X)
    findings += check_coverage(P, support_info)
    findings += check_bench_guidelines(P)
    # reuse review_op for the four standard domains (support already computed)
    findings += sup
    findings += review_op.check_test(P, support_info)
    findings += review_op.check_bench(P, support_info, do_run)
    findings += review_op.check_docs(P, support_info)
    # execution + readiness
    findings += check_exec(P, do_run)
    findings += check_readiness(P, do_run)
    return findings


def render_md(P, findings, domains, phase):
    icon = {PASS: "✅", GAP: "❌", NA: "➖", MANUAL: "🔍", REC: "💡"}
    lines = [f"# make-op ({phase}): {P.Op}  (op={P.op}, py=cvcuda.{P.pyname})", ""]
    counts = {}
    seen = set()
    for dom in domains:
        df = [f for f in findings if f.domain == dom]
        if not df:
            continue
        lines.append(f"## {dom}")
        for f in df:
            counts[f.status] = counts.get(f.status, 0) + 1
            seen.add(id(f))
            lines.append(
                f"- {icon.get(f.status, '?')} **{f.status}** `{f.id}` — {f.summary}"
            )
            if f.evidence:
                lines.append(f"    - evidence: {f.evidence}")
            if f.status == GAP and f.fix:
                lines.append(f"    - fix: {f.fix}")
            if f.guideline:
                lines.append(f"    - ref: {f.guideline}")
        gaps = sum(1 for f in df if f.status == GAP)
        man = sum(1 for f in df if f.status == MANUAL)
        verdict = (
            "PASS" if gaps == 0 and man == 0 else ("GAPS" if gaps else "NEEDS-REVIEW")
        )
        lines.append(f"  → **{dom} verdict: {verdict}** ({gaps} GAP, {man} MANUAL)")
        lines.append("")
    total_gap = counts.get(GAP, 0)
    total_man = counts.get(MANUAL, 0)
    overall = (
        "PASS"
        if total_gap == 0 and total_man == 0
        else ("GAPS" if total_gap else "NEEDS-REVIEW")
    )
    lines.append("## verdict")
    lines.append(
        f"**{overall}** — " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    if phase == "scaffold":
        lines.append("")
        lines.append(
            "Scaffold-only: implementation outstanding (IMP/COV/EXEC). Run `--phase done` after implementing."
        )
    else:
        lines.append("")
        lines.append(
            "Done = a re-run (with --run + CI baselines) shows zero GAP and zero unresolved MANUAL."
        )
    return "\n".join(lines)


def render_json(P, findings, domains, phase):
    counts = {}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
    return json.dumps(
        {
            "operator": P.Op,
            "op": P.op,
            "pyname": P.pyname,
            "phase": phase,
            "findings": [vars(f) for f in findings],
            "counts": counts,
            "exit_gap": counts.get(GAP, 0),
        },
        indent=2,
        sort_keys=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic new-operator checker (CV-CUDA /make-op)."
    )
    ap.add_argument("operator", help="Operator name (PascalCase, e.g. Invert)")
    ap.add_argument("--phase", required=True, choices=["scaffold", "done"])
    ap.add_argument(
        "--bare",
        action="store_true",
        help="scaffold-only with no authored spec (SPEC items -> MANUAL)",
    )
    ap.add_argument("--format", default="md", choices=["md", "json"])
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--run",
        action="store_true",
        help="execute the operator's tests (EXEC group); needs a built GPU env",
    )
    args = ap.parse_args(argv)

    P = review_op.resolve_op(args.operator)
    X = extra_paths(P)

    if args.phase == "scaffold":
        findings = run_scaffold(P, X, args.bare)
        domains = SCAFFOLD_DOMAINS
    else:
        findings = run_done(P, X, args.bare, args.run)
        domains = DONE_DOMAINS

    report = (
        render_md(P, findings, domains, args.phase)
        if args.format == "md"
        else render_json(P, findings, domains, args.phase)
    )
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    return 1 if any(f.status == GAP for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
