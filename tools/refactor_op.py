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
"""Deterministic per-operator refactoring / redundancy checker for CV-CUDA.

Implements the catalog defined in .agents/guidance/REFACTOR_OP_GUIDELINES.md. Two phases:

  (default) assess : read-only candidate report of operator-scoped *semantic* redundancy
                     SonarQube CPD cannot express (near-duplicate Tensor/VarShape kernels,
                     reinvented shared utilities, dead code, ...). Findings are advisory
                     (RECOMMENDATION/MANUAL); the assess report is exit 0.
  --phase verify   : the strict, agent-independent PARITY gate for an applied refactor.
                     Deterministic, local, artifact-derived (git + parsed matrices): the
                     refactor must change nothing observable — frozen test surface, frozen
                     bench surface, public API/ABI, the declared feature matrix and the test
                     coverage matrix must all be identical base-vs-working. Bit-exactness is
                     proved by running the frozen tests locally when the MR scope includes a
                     refactor (flagged here as a manual proof).

The checker is read-only, deterministic and idempotent: no clocks, no randomness, no network;
the changed-set comes from `git diff <base>`. The same tree yields a byte-identical report.
Applying a refactor is the agent's job; this tool only assesses and verifies (see the skill).

Usage:
    python3 tools/refactor_op.py <Operator> [--phase assess|verify]
            [--domain impl|api|xcut|all] [--base <ref>] [--format md|json] [--out PATH] [--apply]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUIDE = ".agents/guidance/REFACTOR_OP_GUIDELINES.md"
AGENT_TOOLS = REPO / ".agents" / "tools"
if str(AGENT_TOOLS) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS))

from binding_api import binding_api_snapshot  # noqa: E402
from operator_source_map import (  # noqa: E402
    SHARED_KERNEL_SOURCES,
    all_op_names,
    legacy_belongs,
)

PASS, GAP, NA, MANUAL, REC = "PASS", "GAP", "N-A", "MANUAL", "RECOMMENDATION"
DOMAINS = ("impl", "api", "xcut")

# --- assess tunables (overridable from the guidelines doc's Curated data) -------------------
SHINGLE_K = 5  # normalized-line k-gram window for the similarity fingerprint
SIM_THRESHOLD = 0.80  # Jaccard at/above which two blocks are "near-duplicate"
MIN_BLOCK_LINES = 6  # ignore blocks smaller than this (too small to be worth unifying)

# The canonical "should-reuse" helpers. A locally-defined function whose name collides with one
# of these is a reinvented-wheel candidate (RED-10). Header paths are cited in the guidelines.
SHARED_UTILS = {
    "SaturateCast",
    "StaticCast",
    "ConvertBaseTypeTo",
    "TensorWrap",
    "CreateSameShapeImageBatch",
}

# Reusable Doxygen Limitations dtype row -> canonical token (feature-matrix parity, VER-4).
DTYPE_ROW = {
    tuple(name.split()): token
    for name, token in (
        ("8bit Unsigned", "u8"),
        ("8bit Signed", "s8"),
        ("16bit Unsigned", "u16"),
        ("16bit Signed", "s16"),
        ("32bit Unsigned", "u32"),
        ("32bit Signed", "s32"),
        ("16bit Float", "f16"),
        ("32bit Float", "f32"),
        ("64bit Float", "f64"),
    )
}


@dataclass
class Finding:
    """One deterministic rule result in the emitted report."""

    id: str
    domain: str
    status: str
    summary: str
    evidence: str = ""
    guideline: str = ""
    fix: str = ""


@dataclass
class OpPaths:
    """Resolved file locations for one operator's refactor surface."""

    op: str
    Op: str
    pyname: str
    priv: list
    header: Path
    pybind: Path
    hpp: Path
    test_cpp: Path
    bench_cpp: Path
    test_py: Path
    bench_py: Path
    bench_cfg: Path


@dataclass
class Block:
    """Normalized code block fingerprint used by the redundancy checks."""

    name: str
    start: int
    end: int
    norm: list  # normalized non-empty body lines
    shingles: list  # stable shingle hashes of `norm`


# --------------------------------------------------------------------------- io / git helpers
def read(p: Path):
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except (OSError, AttributeError):
        return None


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def git(*args):
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(REPO), capture_output=True, text=True
        )
        return proc.stdout if proc.returncode == 0 else ""
    except OSError:
        return ""


def git_show(ref: str, relpath: str):
    """File content at <ref>:<relpath>, or None if absent there."""
    try:
        r = subprocess.run(
            ["git", "show", f"{ref}:{relpath}"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        return r.stdout if r.returncode == 0 else None
    except OSError:
        return None


def grep(pattern, text, flags=0):
    if not text:
        return []
    rx = re.compile(pattern, flags)
    return [(i + 1, ln) for i, ln in enumerate(text.splitlines()) if rx.search(ln)]


def first_evidence(path: Path, hits):
    if not hits:
        return ""
    ln, line = hits[0]
    return f"{rel(path)}:{ln}: {line.strip()[:120]}"


# --------------------------------------------------------------- deterministic text analysis
def _stable_hash(s: str) -> int:
    """Process-independent 64-bit hash. Uses hashlib (NOT Python's salted hash()) so the
    report is byte-identical across runs/processes — the determinism guarantee."""
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _strip_code(text: str) -> str:
    """Remove // and /* */ comments and the *contents* of string/char literals, preserving
    newlines (so line numbers are stable) and brace structure. Deterministic, no state leak.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and nxt == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            continue
        if c in "\"'":
            quote = c
            out.append(quote)
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            out.append(quote)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def normalize_line(line: str) -> str:
    """Whitespace-normalized code content of a line (comments/strings already stripped by
    _strip_code upstream). Empty for blank/structure-only lines we don't fingerprint on.
    """
    s = re.sub(r"\s+", " ", line.strip())
    return s


def shingle_hashes(norm_lines, k=SHINGLE_K):
    if len(norm_lines) < k:
        return [_stable_hash("\n".join(norm_lines))] if norm_lines else []
    return [
        _stable_hash("\n".join(norm_lines[i + offset] for offset in range(k)))
        for i in range(len(norm_lines) - k + 1)
    ]


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_SIG_NAME = re.compile(r"([A-Za-z_]\w*)\s*$")


# `constexpr`: an `if constexpr (...)` head leaves `constexpr` as the token before the
# parens, which must not read as a function named "constexpr" (nested control flow would
# then pair with its enclosing function as a near-duplicate).
_CTRL_KW = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "else",
    "do",
    "return",
    "constexpr",
}


def extract_blocks(text, kind="cpp"):
    """Brace/def-balanced function-or-kernel body extraction with stable (file) ordering,
    at any nesting depth (functions live inside namespaces; pybind bodies are lambdas).
    Returns Block(name, start_line, end_line, norm_lines, shingles). Tolerant: on any
    structural oddity it yields fewer blocks rather than raising."""
    if not text:
        return []
    if kind == "py":
        return _extract_py_blocks(text)
    stripped = _strip_code(text)
    blocks = []
    stack = []  # one entry per open brace: (name_or_None, start_line, body_start_char)
    paren = 0
    seg_start = 0  # char index just after the last statement boundary at paren depth 0
    line = 1
    i, n = 0, len(stripped)
    while i < n:
        c = stripped[i]
        if c == "\n":
            line += 1
        elif c == "(":
            paren += 1
        elif c == ")":
            paren = max(0, paren - 1)
        elif paren == 0 and c == ";":
            seg_start = i + 1
        elif c == "{":
            name = _block_open(stripped[seg_start:i]) if paren == 0 else None
            stack.append((name, line, i + 1))
            if paren == 0:
                seg_start = i + 1
        elif c == "}":
            name, start_line, body_start = stack.pop() if stack else (None, line, i)
            if name:
                norm = [
                    normalize_line(ln) for ln in stripped[body_start:i].splitlines()
                ]
                norm = [ln for ln in norm if ln]
                if len(norm) >= MIN_BLOCK_LINES:
                    blocks.append(
                        Block(name, start_line, line, norm, shingle_hashes(norm))
                    )
            if paren == 0:
                seg_start = i + 1
        i += 1
    return blocks


def _block_open(sig):
    """If `sig` precedes a function/kernel/lambda body, return a name; else None (skips
    namespace/class/struct/control-flow braces and brace-initializers)."""
    paren = sig.rfind(")")
    if paren < 0:
        return None
    open_paren = _match_open_paren(sig, paren)
    if open_paren < 0:
        return None
    head = sig[:open_paren].rstrip()
    if head.endswith("]"):  # lambda capture list `[](...)`
        return "lambda"
    m = _SIG_NAME.search(head)
    if not m:
        return None
    if m.group(1) in _CTRL_KW:
        return None
    return m.group(1)


def _match_open_paren(s, close_idx):
    depth = 0
    i = close_idx
    while i >= 0:
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    return -1


def _extract_py_blocks(text):
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        m = re.match(r"(\s*)def\s+(\w+)\s*\(", lines[i])
        if not m:
            i += 1
            continue
        indent, name, start = len(m.group(1)), m.group(2), i + 1
        body = []
        j = i + 1
        while j < len(lines):
            ln = lines[j]
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
                break
            body.append(ln)
            j += 1
        norm = [normalize_line(b) for b in body]
        norm = [x for x in norm if x and not x.startswith("#")]
        if len(norm) >= MIN_BLOCK_LINES:
            blocks.append(Block(name, start, j, norm, shingle_hashes(norm)))
        i = j
    return blocks


def near_duplicate_pairs(blocks, threshold=SIM_THRESHOLD):
    """All block pairs with jaccard >= threshold, sorted by (-ratio, a.start, b.start)
    for byte-stable output."""
    pairs = []
    for x in range(len(blocks)):
        for y in range(x + 1, len(blocks)):
            r = jaccard(blocks[x].shingles, blocks[y].shingles)
            if r >= threshold:
                pairs.append((round(r, 4), blocks[x], blocks[y]))
    pairs.sort(key=lambda t: (-t[0], t[1].start, t[2].start))
    return pairs


def duplicate_pair_findings(blocks, threshold, allow, make_finding):
    """Build findings for near-duplicate block pairs while honoring the per-op allowlist."""
    findings = []
    for ratio, a, b in near_duplicate_pairs(blocks, threshold):
        if a.name in allow or b.name in allow:
            continue
        findings.append(make_finding(ratio, a, b))
    return findings


# ----------------------------------------------------------------------- operator resolution
def resolve_op(arg: str):
    stem = arg.strip()
    op = stem.lower()
    hdr_dir = REPO / "src/cvcuda/include/cvcuda"
    header = next(
        (h for h in sorted(hdr_dir.glob("Op*.h")) if h.stem[2:].lower() == op),
        None,
    )
    Op = header.stem[2:] if header else stem
    header = header or hdr_dir / f"Op{stem}.h"
    pybind = REPO / f"python/mod_cvcuda/operators/Op{Op}.cpp"
    priv_dir = REPO / "src/cvcuda/priv"
    priv = [
        cand
        for cand in (priv_dir / f"Op{Op}.cu", priv_dir / f"Op{Op}.cpp")
        if cand.exists()
    ]
    legacy = priv_dir / "legacy"
    if legacy.is_dir():
        all_ops = all_op_names()
        for g in sorted(legacy.glob("*.c*")):
            if legacy_belongs(g.stem, op, all_ops):
                priv.append(g)
    # Shared/legacy kernel sources not matched by the op-name globs (e.g. filter.cu for
    # Gaussian, the HQResize kernel headers) — see SHARED_KERNEL_SOURCES.
    for extra in SHARED_KERNEL_SOURCES.get(op, []):
        cand = priv_dir / extra
        if cand.exists() and cand not in priv:
            priv.append(cand)
    paths = {
        "header": header,
        "hpp": hdr_dir / f"Op{Op}.hpp",
        "pybind": pybind,
        "test_cpp": REPO / f"tests/cvcuda/system/TestOp{Op}.cpp",
        "test_py": REPO / f"tests/cvcuda/python/test_op{op}.py",
        "bench_cpp": REPO / f"bench/cpp/ops/Bench{Op}.cpp",
        "bench_py": REPO / f"bench/python/ops/bench_{op}.py",
        "bench_cfg": REPO / f"bench/config/operators/{op}.json",
    }
    return OpPaths(op, Op, resolve_pyname(pybind, op), priv=priv, **paths)


def resolve_pyname(pybind: Path, op: str) -> str:
    text = read(pybind)
    if text:
        names = re.findall(r'm\.def\(\s*"([a-z0-9_]+)"', text)
        base = [n for n in names if not n.endswith("_into")]
        if base:
            return sorted(base, key=len)[0]
    return op


def load_curated():
    """Override defaults (clone threshold, shared-util set, per-op duplicate allowlist) from
    the guidelines doc. Falls back to module defaults when the doc is absent."""
    text = read(REPO / GUIDE) or ""
    sim = SIM_THRESHOLD
    utils = set(SHARED_UTILS)
    allowlist = {}
    m = re.search(r"similarity-threshold\s*[:=]\s*(0?\.\d+)", text)
    if m:
        sim = float(m.group(1))
    section, in_block = None, False
    section_markers = {
        "### Shared-util reference set": "utils",
        "### Duplicate allowlist": "allow",
    }
    for line in text.splitlines():
        matched_section = section_markers.get(line)
        if matched_section:
            section, in_block = matched_section, False
            continue
        if line.startswith("###") or line.startswith("## "):
            section = None
        stripped = line.strip()
        if section and stripped.startswith("```"):
            in_block = not in_block
            continue
        if section and in_block:
            if not stripped or stripped.startswith("#"):
                continue
            if section == "utils":
                utils.add(stripped)
            elif section == "allow" and ":" in stripped:
                k, v = stripped.split(":", 1)
                allowlist[k.strip().lower()] = [
                    t.strip() for t in v.split(",") if t.strip()
                ]
    return {"sim": sim, "utils": utils, "allowlist": allowlist}


# ================================================================================ assess: impl
def check_impl(P: OpPaths, curated, reader=read):
    out = []
    g = ".agents/guidance/REFACTOR_OP_GUIDELINES.md#impl"
    priv_texts = [(p, reader(p)) for p in P.priv]
    if not P.priv:
        out.append(
            Finding("RED-1", "impl", NA, "No priv implementation files located", "", g)
        )
        return out

    allow = set(curated["allowlist"].get(P.op, []))

    # RED-1: near-duplicate kernels/functions within the operator's priv (Tensor vs VarShape).
    blocks = []
    for p, t in priv_texts:
        for b in extract_blocks(t, "cpp"):
            blocks.append((p, b))
    flat = [b for _, b in blocks]
    owner = {id(b): p for p, b in blocks}
    red1 = duplicate_pair_findings(
        flat,
        curated["sim"],
        allow,
        lambda ratio, a, b: (
            Finding(
                "RED-1",
                "impl",
                REC,
                f"Near-duplicate bodies '{a.name}' / '{b.name}' (jaccard {ratio:.2f}) — unify",
                f"{rel(owner[id(a)])}:{a.start} vs {rel(owner[id(b)])}:{b.start}",
                g,
                "Unify behind a templated kernel; put the addressing difference in the accessor "
                "(cf. OpBrightnessContrast.cu -> DoBrightnessContrast<isPlanar>).",
            )
        ),
    )
    out += red1
    if not red1:
        out.append(
            Finding(
                "RED-1",
                "impl",
                PASS,
                "No near-duplicate kernel/function bodies in priv",
                f"{len(flat)} block(s) compared at jaccard>={curated['sim']:.2f}",
                g,
            )
        )

    # RED-4: re-implemented layout validation that should use nvcv TensorDataAccess helpers.
    red4 = []
    for p, t in priv_texts:
        red4 += [
            (p, h)
            for h in grep(
                r"TENSOR_NCHW\b.*\bTENSOR_NHWC\b|TENSOR_NHWC\b.*\bTENSOR_NCHW\b", t
            )
        ]
    if red4:
        p, (ln, line) = red4[0]
        out.append(
            Finding(
                "RED-4",
                "impl",
                MANUAL,
                f"Manual layout validation in priv ({len(red4)} site[s]) — prefer shared helpers",
                f"{rel(p)}:{ln}: {line.strip()[:100]}",
                g,
                "Replace hand-rolled layout checks with nvcv TensorDataAccess helpers (cf. OpStack.cpp).",
            )
        )
    else:
        out.append(
            Finding(
                "RED-4",
                "impl",
                PASS,
                "No hand-rolled layout-validation chains found",
                "",
                g,
            )
        )

    # RED-5: manual index/stride arithmetic in a file that does not use a *Wrap accessor.
    red5 = []
    for p, t in priv_texts:
        if not t:
            continue
        uses_wrap = re.search(r"TensorWrap|ImageBatchVarShapeWrap|TensorDataAccess", t)
        stride_hits = grep(r"\*\s*\w*[Ss]tride\w*\b", t)
        if stride_hits and not uses_wrap:
            red5.append((p, stride_hits[0]))
    if red5:
        p, (ln, line) = red5[0]
        out.append(
            Finding(
                "RED-5",
                "impl",
                MANUAL,
                f"Hand-rolled stride arithmetic without an accessor wrapper ({len(red5)} file[s])",
                f"{rel(p)}:{ln}: {line.strip()[:100]}",
                g,
                "Use TensorWrap / TensorDataAccessStridedImagePlanar accessors instead of manual "
                "stride math.",
            )
        )
    else:
        out.append(
            Finding(
                "RED-5",
                "impl",
                PASS,
                "No accessor-free manual stride math detected",
                "",
                g,
            )
        )
    return out


# ================================================================================= assess: api
def check_api(P: OpPaths, curated, reader=read):
    out = []
    g = ".agents/guidance/REFACTOR_OP_GUIDELINES.md#api"
    t = reader(P.pybind)
    if t is None:
        out.append(
            Finding("RED-6", "api", NA, "Python binding not found", rel(P.pybind), g)
        )
        return out
    allow = set(curated["allowlist"].get(P.op, []))
    blocks = [b for b in extract_blocks(t, "cpp")]
    red6 = duplicate_pair_findings(
        blocks,
        curated["sim"],
        allow,
        lambda ratio, a, b: (
            Finding(
                "RED-6",
                "api",
                REC,
                f"Duplicated binding bodies '{a.name}' / '{b.name}' (jaccard {ratio:.2f})",
                f"{rel(P.pybind)}:{a.start} vs {rel(P.pybind)}:{b.start}",
                g,
                "Extract a shared submit helper for the Tensor/VarShape & allocating/_into paths; "
                "reuse VarShapeUtils.hpp (CreateSameShapeImageBatch).",
            )
        ),
    )
    out += red6
    if not red6:
        out.append(
            Finding(
                "RED-6",
                "api",
                PASS,
                "No duplicated binding bodies above threshold",
                f"{len(blocks)} binding block(s) compared",
                g,
            )
        )
    return out


# ================================================================================ assess: xcut
def check_xcut(P: OpPaths, curated, reader=read):
    out = []
    g = ".agents/guidance/REFACTOR_OP_GUIDELINES.md#xcut"
    surface = [(p, reader(p)) for p in (P.priv + [P.pybind])]
    surface = [(p, t) for p, t in surface if t]
    full_text = "\n".join(t for _, t in surface)
    utils = curated["utils"]

    # RED-10: a locally-defined function shadowing a canonical shared util.
    shadows = []
    for p, t in surface:
        for u in sorted(utils):
            for ln, line in grep(rf"\b{re.escape(u)}\b\s*\([^)]*\)\s*(\{{|$)", t):
                # a *definition* (sig followed by a body), not a call site
                if re.search(rf"\b{re.escape(u)}\s*\(", line) and re.search(
                    rf"(template|__device__|__host__|inline|static).*\b{re.escape(u)}\b",
                    line,
                ):
                    shadows.append((u, p, ln, line))
    if shadows:
        u, p, ln, line = shadows[0]
        out.append(
            Finding(
                "RED-10",
                "xcut",
                MANUAL,
                f"Local helper shadows shared util '{u}' ({len(shadows)} site[s])",
                f"{rel(p)}:{ln}: {line.strip()[:100]}",
                g,
                "Replace the reinvented helper with cuda_tools/{SaturateCast,StaticCast,TypeTraits}.hpp.",
            )
        )
    else:
        out.append(
            Finding(
                "RED-10",
                "xcut",
                PASS,
                "No local helper shadows a canonical shared util",
                "",
                g,
            )
        )

    # RED-11: dead code — a static / anonymous-namespace function referenced only by its
    # definition (name occurs exactly once across the operator surface).
    dead = []
    for p, t in surface:
        for b in extract_blocks(t, "cpp"):
            decl = grep(rf"\bstatic\b.*\b{re.escape(b.name)}\s*\(", t)
            if not decl:
                continue
            occurrences = len(re.findall(rf"\b{re.escape(b.name)}\b", full_text))
            if occurrences <= 1:
                dead.append((p, b))
    if dead:
        p, b = dead[0]
        out.append(
            Finding(
                "RED-11",
                "xcut",
                REC,
                f"Dead code: static '{b.name}' referenced only at its definition ({len(dead)} fn[s])",
                f"{rel(p)}:{b.start}",
                g,
                "Remove the unreferenced function.",
            )
        )
    else:
        out.append(
            Finding(
                "RED-11",
                "xcut",
                PASS,
                "No obviously-dead static functions found",
                "",
                g,
            )
        )
    return out


# =============================================================================== verify: parity
def parse_limitations(header_text):
    """Declared feature matrix from the Doxygen Limitations table, split by Input/Output."""
    if not header_text:
        return {}
    m = re.search(
        r"Limitations:(.*?)(?:\n\s*\*\s*Input/Output dependency|\*/)", header_text, re.S
    )
    blob = m.group(1) if m else header_text

    def section(label):
        found = re.search(rf"{label}:(.*?)(?:Input:|Output:|$)", blob, re.S)
        region = found.group(1) if found else ""
        layouts = set()
        for lay in re.finditer(r"Data Layout:\s*\[([^\]]*)\]", region):
            layouts.update(re.findall(r"k([A-Z]+)", lay.group(1)))
        channels = set()
        for ch in re.finditer(r"Channels:\s*\[([^\]]*)\]", region):
            channels.update(int(x) for x in re.findall(r"\d+", ch.group(1)))
        dtypes = set()
        for bits, sign, allowed in re.findall(
            r"(\d+bit)\s+(Unsigned|Signed|Float)\s*\|\s*(Yes|No)", region
        ):
            if allowed == "Yes" and (bits, sign) in DTYPE_ROW:
                dtypes.add(DTYPE_ROW[(bits, sign)])
        return {"layouts": layouts, "channels": channels, "dtypes": dtypes}

    if "Input:" not in blob and "Output:" not in blob:
        blob = f"Input:{blob}"
    return {"input": section("Input"), "output": section("Output")}


def _feature_sig(header_text):
    lim = parse_limitations(header_text)
    return tuple(
        (
            side,
            tuple(sorted(lim.get(side, {}).get("layouts", set()))),
            tuple(sorted(lim.get(side, {}).get("channels", set()))),
            tuple(sorted(lim.get(side, {}).get("dtypes", set()))),
        )
        for side in ("input", "output")
    )


_TEST_MACRO = re.compile(
    r"\b(TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P|NVCV_TYPED_TEST_SUITE|"
    r"NVCV_INSTANTIATE_TEST_SUITE_P|INSTANTIATE_TEST_SUITE_P|NVCV_TEST_SUITE_P)\s*\(\s*([A-Za-z_]\w*)"
)


def _coverage_sig(test_text):
    """A stable signature of the test surface: the multiset of test/instantiation macros, plus a
    count of parametrized value rows (test_case_t/Param/ValuesIn entries). Used for VER-5.
    """
    if not test_text:
        return ()
    macros = sorted(
        f"{m.group(1)}:{m.group(2)}" for m in _TEST_MACRO.finditer(test_text)
    )
    value_rows = len(re.findall(r"test::Param|test_case_t\{|\{\s*test::", test_text))
    return tuple(macros) + (f"value_rows={value_rows}",)


def check_verify(P: OpPaths, base):
    g = ".agents/guidance/REFACTOR_OP_GUIDELINES.md#verify"
    out = []

    test_rels = [rel(P.test_cpp), rel(P.test_py)]
    bench_rels = sorted({rel(P.bench_cpp), rel(P.bench_py), rel(P.bench_cfg)})

    # VER-1: frozen test surface (no diff base->working under the op's test files).
    test_diff = git("diff", "--name-only", base, "--", *test_rels)
    changed_tests = [ln.strip() for ln in test_diff.splitlines() if ln.strip()]

    # VER-4: declared feature matrix identical base-vs-working.
    base_hdr = git_show(base, rel(P.header))
    cur_hdr = read(P.header)
    feat_base, feat_cur = _feature_sig(base_hdr), _feature_sig(cur_hdr)
    if feat_base == feat_cur:
        out.append(
            Finding(
                "VER-4",
                "verify",
                PASS,
                "Declared feature matrix unchanged (layouts/channels/dtypes)",
                f"layouts/ch/dtypes identical vs {base}",
                g,
            )
        )
    else:
        out.append(
            Finding(
                "VER-4",
                "verify",
                GAP,
                "Declared feature matrix changed — refactoring must not change supported features",
                f"base={feat_base} now={feat_cur}",
                g,
                "Revert the Limitations/feature change, or move it to a separate feature MR.",
            )
        )

    # VER-5: test coverage matrix identical (only matters if the test files changed).
    if not changed_tests:
        out.append(
            Finding(
                "VER-1",
                "verify",
                PASS,
                "Frozen test surface: no diff under the operator's test files",
                "; ".join(test_rels),
                g,
            )
        )
        out.append(
            Finding(
                "VER-5",
                "verify",
                PASS,
                "Test coverage matrix unchanged (test files untouched)",
                "",
                g,
            )
        )
    else:
        cov_base = _coverage_sig(git_show(base, rel(P.test_cpp))) + _coverage_sig(
            git_show(base, rel(P.test_py))
        )
        cov_cur = _coverage_sig(read(P.test_cpp)) + _coverage_sig(read(P.test_py))
        same = cov_base == cov_cur
        out.append(
            Finding(
                "VER-1",
                "verify",
                GAP,
                "Test files changed; refactoring must not change frozen operator tests",
                "changed: " + ", ".join(changed_tests),
                g,
                "Move shared helpers outside the operator test files or split the test change "
                "into a separate MR.",
            )
        )
        out.append(
            Finding(
                "VER-5",
                "verify",
                PASS if same else GAP,
                "Test coverage matrix " + ("unchanged" if same else "changed"),
                f"macros/value-rows {'identical' if same else 'differ'} vs {base}",
                g,
                (
                    ""
                    if same
                    else "Restore the original test coverage; refactoring is feature-neutral."
                ),
            )
        )

    # VER-2: frozen bench surface (no diff under the op's bench files incl. config/baselines).
    bench_diff = git("diff", "--name-only", base, "--", *bench_rels)
    changed_bench = sorted({ln.strip() for ln in bench_diff.splitlines() if ln.strip()})
    out.append(
        Finding(
            "VER-2",
            "verify",
            PASS if not changed_bench else GAP,
            (
                "Frozen bench surface (sources/config/baselines untouched)"
                if not changed_bench
                else "Bench surface changed — benchmarks are the frozen measurement instrument"
            ),
            (
                "; ".join(bench_rels)
                if not changed_bench
                else "changed: " + ", ".join(changed_bench)
            ),
            g,
            (
                ""
                if not changed_bench
                else "Revert all bench/baseline edits; a refactor must not touch the measurement."
            ),
        )
    )

    # VER-3: public API/ABI unchanged (header + .hpp + binding signatures).
    out.append(_api_abi(P, base, g))

    # VER-6: bit-exactness — proved by the frozen tests passing on the refactored build. The
    # build+run is a local proof for refactor-scoped work; flagged here so the gate is explicit.
    out.append(
        Finding(
            "VER-6",
            "verify",
            MANUAL,
            "Bit-exact: build the refactored tree and run the frozen Op%s tests (EXPECT_EQ goldens)"
            % P.Op,
            "run: build-rel/bin/cvcuda_test_system --gtest_filter='Op%s*' (must be green)"
            % P.Op,
            g,
        )
    )

    # VER-7: the applied redundancy is actually gone (re-run assess).
    out.append(
        Finding(
            "VER-7",
            "verify",
            MANUAL,
            "Confirm the applied redundancy is resolved: re-run assess and check the finding is gone",
            "run: python3 tools/refactor_op.py %s" % P.Op,
            g,
        )
    )
    return out


def _api_abi(P, base, g):
    header_paths = [rel(P.header), rel(P.hpp)]
    header_diff = git("diff", base, "--", *header_paths)
    sig = []
    signature_rx = re.compile(r"CVCUDA_PUBLIC|operator\(\)|Submit\s*\(")
    for ln in header_diff.splitlines():
        is_changed_line = ln[:1] in "+-" and ln[1:2] != ln[:1]
        if is_changed_line and signature_rx.search(ln):
            sig.append(ln)
    if sig:
        return Finding(
            "VER-3",
            "verify",
            GAP,
            "Public API/ABI signature(s) changed — that is feature work, not a refactor",
            "; ".join(s.strip()[:80] for s in sig[:3]),
            g,
            "Revert the signature change or split it into a separate feature MR.",
        )
    if header_diff.strip():
        return Finding(
            "VER-3",
            "verify",
            MANUAL,
            "Public headers changed outside a recognized signature",
            "diff in " + ", ".join(Path(p).name for p in header_paths),
            g,
            "Inspect the public-header diff and prove that API and ABI are unchanged.",
        )

    binding_path = rel(P.pybind)
    binding_diff = git("diff", base, "--", binding_path)
    if not binding_diff.strip():
        return Finding(
            "VER-3",
            "verify",
            PASS,
            "Public API/ABI surface unchanged",
            "no changes in "
            + ", ".join(Path(p).name for p in (*header_paths, binding_path)),
            g,
        )

    baseline_source = git_show(base, binding_path)
    candidate_source = read(P.pybind)
    baseline_surface = binding_api_snapshot(baseline_source, P.Op)
    candidate_surface = binding_api_snapshot(candidate_source, P.Op)
    if baseline_surface is None or candidate_surface is None:
        return Finding(
            "VER-3",
            "verify",
            MANUAL,
            "Binding API snapshot could not be resolved conservatively",
            f"baseline parsed={baseline_surface is not None}; "
            f"candidate parsed={candidate_surface is not None}",
            g,
            "Inspect registrations, callable signatures, and reachable type aliases manually.",
        )
    if baseline_surface == candidate_surface:
        return Finding(
            "VER-3",
            "verify",
            PASS,
            "Public API/ABI surface unchanged",
            "public headers unchanged; binding registration/signatures/type aliases identical",
            g,
        )

    changed_parts = []
    if baseline_surface[0] != candidate_surface[0]:
        changed_parts.append("m.def registration/arguments/defaults")
    if baseline_surface[1] != candidate_surface[1]:
        changed_parts.append("bound callable signature(s)")
    if baseline_surface[2] != candidate_surface[2]:
        changed_parts.append("bound callable type alias(es)")
    return Finding(
        "VER-3",
        "verify",
        GAP,
        "Python binding API changed — that is feature work, not a refactor",
        "; ".join(changed_parts),
        g,
        "Revert the API change or split it into a separate feature MR.",
    )


# ================================================================================== driver
def run_assess(P, domains, curated, reader=read):
    findings = []
    if "impl" in domains:
        findings += check_impl(P, curated, reader)
    if "api" in domains:
        findings += check_api(P, curated, reader)
    if "xcut" in domains:
        findings += check_xcut(P, curated, reader)
    return findings


def _open_red_ids(P, curated, reader):
    """The set of RED-* ids with a non-PASS, non-N-A finding under `reader` — i.e. open
    redundancy. Used to quantify before-vs-after when verifying an applied refactor."""
    return {
        f.id
        for f in run_assess(P, DOMAINS, curated, reader)
        if f.status in (REC, MANUAL)
    }


def refactor_summary(P, base, curated):
    """Deterministic impact summary for an applied refactor: LOC delta on the impl/binding
    files vs `base`, and which RED-* findings the change resolved (base -> working)."""
    paths = [rel(p) for p in P.priv] + [rel(P.pybind)]
    numstat = git("diff", "--numstat", base, "--", *paths)
    ins = dels = 0
    for ln in numstat.splitlines():
        cols = ln.split("\t")
        if len(cols) >= 2 and cols[0].isdigit() and cols[1].isdigit():
            ins += int(cols[0])
            dels += int(cols[1])
    base_open = _open_red_ids(P, curated, lambda p: git_show(base, rel(p)))
    now_open = _open_red_ids(P, curated, read)
    return {
        "loc_insertions": ins,
        "loc_deletions": dels,
        "loc_net": ins - dels,
        "redundancy_resolved": sorted(base_open - now_open),
        "redundancy_introduced": sorted(now_open - base_open),
        "redundancy_open_now": sorted(now_open),
    }


def render_md(P, phase, findings, domains, summary=None):
    icon = {PASS: "✅", GAP: "❌", NA: "➖", MANUAL: "🔍", REC: "💡"}
    title = f"# refactor-op: {P.Op}  (op={P.op})  phase={phase}"
    lines = [title, ""]
    counts = {}
    groups = domains if phase == "assess" else ["verify"]
    for grp in groups:
        df = [f for f in findings if f.domain == grp]
        if not df:
            continue
        lines.append(f"## {grp}")
        for f in df:
            counts[f.status] = counts.get(f.status, 0) + 1
            lines.append(
                f"- {icon.get(f.status, '?')} **{f.status}** `{f.id}` — {f.summary}"
            )
            if f.evidence:
                lines.append(f"    - evidence: {f.evidence}")
            if f.status in (GAP, REC, MANUAL) and f.fix:
                lines.append(f"    - fix: {f.fix}")
            if f.guideline:
                lines.append(f"    - ref: {f.guideline}")
        lines.append("")
    total_gap = counts.get(GAP, 0)
    total_man = counts.get(MANUAL, 0)
    if phase == "verify":
        verdict = (
            "PARITY-OK"
            if total_gap == 0 and total_man == 0
            else ("GATE-FAIL" if total_gap else "NEEDS-LOCAL-PROOF")
        )
        tail = (
            "Gate passes when zero GAP and every MANUAL is resolved green "
            "(bit-exact/redundancy proofs, plus any VER-3 header/binding inspection)."
        )
    else:
        verdict = "ADVISORY"
        tail = "Assess is advisory (RECOMMENDATION/MANUAL). Apply via the skill, then run --phase verify."
    lines.append("## verdict")
    lines.append(
        f"**{verdict}** — " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    lines.append("")
    lines.append(tail)
    if summary is not None:
        net = summary["loc_net"]
        resolved = ", ".join(summary["redundancy_resolved"]) or "none"
        intro = ", ".join(summary["redundancy_introduced"]) or "none"
        lines += [
            "",
            "## refactor summary",
            f"- LOC delta (impl/binding): +{summary['loc_insertions']} / "
            f"-{summary['loc_deletions']} (net {net:+d})",
            f"- redundancy resolved: {resolved}",
            f"- redundancy introduced: {intro}",
            f"- redundancy still open: {', '.join(summary['redundancy_open_now']) or 'none'}",
        ]
    return "\n".join(lines)


def render_json(P, phase, findings, domains, summary=None):
    counts = {}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
    payload = {
        "operator": P.Op,
        "op": P.op,
        "phase": phase,
        "domains": list(domains) if phase == "assess" else ["verify"],
        "findings": [vars(f) for f in findings],
        "counts": counts,
        "exit_gap": counts.get(GAP, 0),
    }
    if summary is not None:
        payload["summary"] = summary
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic per-operator refactoring/redundancy checker (CV-CUDA)."
    )
    ap.add_argument(
        "operator", help="Operator name (PascalCase, e.g. BrightnessContrast)"
    )
    ap.add_argument("--phase", default="assess", choices=["assess", "verify"])
    ap.add_argument(
        "--domain", default="all", help="impl|api|xcut|all (comma-separated; assess)"
    )
    ap.add_argument(
        "--base",
        default="main",
        help="git base ref for the verify diff (default: main)",
    )
    ap.add_argument("--format", default="md", choices=["md", "json"])
    ap.add_argument("--out", default=None, help="also write the report to this path")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="(wrapper-level) emphasize fixes; the checker stays read-only — the agent applies",
    )
    args = ap.parse_args(argv)

    domains = (
        DOMAINS
        if args.domain == "all"
        else tuple(d.strip() for d in args.domain.split(",") if d.strip() in DOMAINS)
    )
    if not domains:
        ap.error("no valid --domain selected (choose from impl,api,xcut,all)")

    if args.apply:
        print(
            "note: refactor_op.py is read-only; --apply is performed by the /refactor-op "
            "agent per .agents/guidance/REFACTOR_OP_GUIDELINES.md, then proven by `--phase verify`.",
            file=sys.stderr,
        )

    P = resolve_op(args.operator)
    resolved_surface = [
        P.header,
        P.hpp,
        P.pybind,
        *P.priv,
        P.test_cpp,
        P.test_py,
        P.bench_cpp,
        P.bench_py,
        P.bench_cfg,
    ]
    if not any(p.exists() for p in resolved_surface):
        ap.error(f"operator '{args.operator}' could not be resolved to CV-CUDA files")
    curated = load_curated()
    summary = None
    if args.phase == "verify":
        findings = check_verify(P, args.base)
        summary = refactor_summary(P, args.base, curated)
    else:
        findings = run_assess(P, domains, curated)

    report = (
        render_md(P, args.phase, findings, domains, summary)
        if args.format == "md"
        else render_json(P, args.phase, findings, domains, summary)
    )
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")

    return 1 if any(f.status == GAP for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
