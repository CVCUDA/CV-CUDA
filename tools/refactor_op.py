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
                     generic copy-paste detection cannot express (near-duplicate Tensor/VarShape
                     kernels, reinvented shared utilities, dead code, ...). Findings are advisory
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

# RED-2 scans every operator's priv sources, not just this operator's, so the corpus is the
# whole private implementation tree. Shared .cuh/.hpp helpers are included on purpose: a body
# cloned *from* one of them (e.g. AdjustColorCommon.cuh) is exactly the reinvented-wheel case
# RED-10's name-only match cannot see.
PRIV_DIR = REPO / "src/cvcuda/priv"
PRIV_SUFFIXES = (".cu", ".cpp", ".cuh", ".hpp")

# The canonical "should-reuse" helpers. A locally-defined function whose name collides with one
# of these is a reinvented-wheel candidate (RED-10). Header paths are cited in the guidelines.
SHARED_UTILS = {
    "SaturateCast",
    "StaticCast",
    "ConvertBaseTypeTo",
    "TensorWrap",
    "CreateSameShapeImageBatch",
}

# Shape-matched reimplementations of canonical cuda_tools utilities (RED-12). RED-10 matches a
# local definition by *name*, so it is blind whenever the local spelling differs — which is the
# usual case. These match on structure instead.
#
# Entries are added only with measured precision on the live tree. Two candidates were probed and
# rejected: nested `min(max(...))` (32 sites, but every one clamps a *coordinate*, not a value, so
# it is not SaturateCast) and bare `255`/`65535` literals (195 sites, dominated by legitimate grid
# limits and histogram bin counts).
REINVENTION_SHAPES = [
    {
        "name": "type -> vectorN alias trait",
        "canonical": "nvcv::cuda::MakeType (cuda_tools/TypeTraits.hpp)",
        # A struct whose entire body is `using type = <cuda vector type>;` is the MakeType
        # mapping rewritten by hand. Keying it on sizeof rather than the type is also a latent
        # bug: a 2-byte __half maps to ushort4 and its bit pattern is then read as integers.
        "rx": re.compile(
            r"struct\s+\w+[^{;]*\{\s*using\s+type\s*=\s*"
            r"(?:u?char|u?short|u?int|float|double)[234]\s*;\s*\}",
            re.S,
        ),
        "fix": "Use `cuda::MakeType<BT, N>` (an alias template — no `typename`) and delete the "
        "trait. Keyed on the type rather than sizeof, so half stays half.",
    },
]

# Byte width of each canonical dtype token, so a `sizeof(BT) == N` dispatch guard can be compared
# against the dtype table the operator's public header declares (RED-3).
DTYPE_BYTES = {
    "u8": 1,
    "s8": 1,
    "u16": 2,
    "s16": 2,
    "f16": 2,
    "u32": 4,
    "s32": 4,
    "f32": 4,
    "f64": 8,
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
    # A body shorter than the window has no k-gram, so it is fingerprinted whole: such blocks
    # then match only on exact equality (jaccard 1.0 or 0.0, nothing between), which is the
    # behaviour you want for one- and two-line helpers. This branch is dead while
    # MIN_BLOCK_LINES >= SHINGLE_K and becomes live as soon as the floor is swept below it.
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


def extract_blocks(text, kind="cpp", min_lines=None):
    """Brace/def-balanced function-or-kernel body extraction with stable (file) ordering,
    at any nesting depth (functions live inside namespaces; pybind bodies are lambdas).
    Returns Block(name, start_line, end_line, norm_lines, shingles). Tolerant: on any
    structural oddity it yields fewer blocks rather than raising.

    `min_lines` overrides MIN_BLOCK_LINES so a caller can sweep the floor downwards to surface
    small shared helpers (see the RED-2 sweep in the refactor-op skill)."""
    floor = MIN_BLOCK_LINES if min_lines is None else min_lines
    if not text:
        return []
    if kind == "py":
        return _extract_py_blocks(text, floor)
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
                if len(norm) >= floor:
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


def _extract_py_blocks(text, floor=None):
    floor = MIN_BLOCK_LINES if floor is None else floor
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
        if len(norm) >= floor:
            blocks.append(Block(name, start, j, norm, shingle_hashes(norm)))
        i = j
    return blocks


_BLOCK_CACHE = {}


def extract_blocks_cached(path: Path, text, kind="cpp", min_lines=None):
    """extract_blocks memoized per (path, content). A verify run walks the priv corpus twice —
    once at <base> and once on the working tree — and the two agree on every file the refactor
    did not touch, so the second walk is otherwise pure re-parsing.

    The path is part of the key deliberately. RED-1 identifies a block's owning file by
    `id(block)`; if two distinct paths with identical content shared cached Block objects those
    ids would collide and the evidence string would name the wrong file.
    """
    floor = MIN_BLOCK_LINES if min_lines is None else min_lines
    key = (rel(path), kind, floor, _stable_hash(text or ""))
    if key not in _BLOCK_CACHE:
        _BLOCK_CACHE[key] = extract_blocks(text, kind, floor)
    return _BLOCK_CACHE[key]


_IF_HEAD = re.compile(r"\bif\s*(?:constexpr\s*)?\(")


def if_conditions(stripped):
    """(offset, condition_text) for every `if (...)` / `if constexpr (...)` in comment-stripped
    source, with the parentheses balanced so the condition is complete however it is wrapped.

    A line-based probe splits a wrapped disjunction across two matches and sees each half as the
    whole guard — which, for a check that reports what a guard *excludes*, produces false
    positives rather than misses.
    """
    out = []
    for m in _IF_HEAD.finditer(stripped):
        i = m.end() - 1  # at the opening paren
        body_start = i + 1
        depth = 0
        for j in range(i, len(stripped)):
            c = stripped[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    out.append((m.start(), stripped[body_start:j]))
                    break
    return out


def priv_corpus_paths():
    """Every private implementation/kernel source, in deterministic order (RED-2's corpus).

    Recursive: the corpus is documented as `src/cvcuda/priv/**`, and enumerating only the top
    level plus `legacy/` silently dropped everything under `legacy/textbackend/`.
    """
    if not PRIV_DIR.is_dir():
        return []
    return sorted(
        p for p in PRIV_DIR.rglob("*") if p.is_file() and p.suffix in PRIV_SUFFIXES
    )


def cross_op_blocks(P, reader, min_lines=None):
    """(path, Block) for every priv file that is NOT part of this operator's own surface.

    Read through `reader`, never `read()`: refactor_summary's base pass must see the corpus as
    it was at <base>. A refactor that hoists several operators at once would otherwise find the
    siblings already deduplicated in the base pass, report nothing there, and so have nothing to
    resolve.
    """
    mine = {p.resolve() for p in P.priv}
    out = []
    for p in priv_corpus_paths():
        if p.resolve() in mine:
            continue
        for b in extract_blocks_cached(p, reader(p), "cpp", min_lines):
            out.append((p, b))
    return out


def cross_op_duplicates(mine, others, threshold):
    """name -> (Block, [(ratio, path, start), ...]) for this operator's blocks that have a
    near-duplicate in another operator's priv.

    Candidates come from a shingle inverted index. Two blocks with no shingle in common have
    jaccard 0, so the index cannot miss a pair at any positive threshold — the result equals the
    full cross product, without the quadratic comparison count.
    """
    index = {}
    for i, (_, b) in enumerate(others):
        for s in set(b.shingles):
            index.setdefault(s, []).append(i)
    hits = {}
    for a in mine:
        candidates = set()
        for s in set(a.shingles):
            candidates.update(index.get(s, ()))
        for i in sorted(candidates):
            other_path, b = others[i]
            ratio = jaccard(a.shingles, b.shingles)
            if ratio >= threshold:
                hits.setdefault(a.name, (a, []))[1].append(
                    (round(ratio, 4), other_path, b.start)
                )
    for _, partners in hits.values():
        partners.sort(key=lambda t: (-t[0], rel(t[1]), t[2]))
    return hits


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
    min_block_lines = MIN_BLOCK_LINES
    utils = set(SHARED_UTILS)
    allowlist = {}
    m = re.search(r"similarity-threshold\s*[:=]\s*(0?\.\d+)", text)
    if m:
        sim = float(m.group(1))
    m = re.search(r"min-block-lines\s*[:=]\s*(\d+)", text)
    if m:
        min_block_lines = max(1, int(m.group(1)))
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
    return {
        "sim": sim,
        "min_block_lines": min_block_lines,
        "utils": utils,
        "allowlist": allowlist,
    }


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
    floor = curated.get("min_block_lines", MIN_BLOCK_LINES)

    # RED-1: near-duplicate kernels/functions within the operator's priv (Tensor vs VarShape).
    blocks = []
    for p, t in priv_texts:
        for b in extract_blocks_cached(p, t, "cpp", floor):
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

    # RED-2: a body in this operator's priv that also exists in another operator's priv. RED-1
    # only ever compares an operator against itself, so a helper copy-pasted across operators is
    # invisible to it; this is the check that names the shared-header candidates.
    others = cross_op_blocks(P, reader, floor)
    # `lambda` is _block_open's placeholder for an unnamed body, not an identifier, so a match on
    # it names nothing anyone can hoist. It stays below the default floor but surfaces once the
    # floor is swept to 1, which is exactly when the sweep should be getting *more* signal.
    hits = cross_op_duplicates(
        [b for b in flat if b.name not in allow and b.name != "lambda"],
        others,
        curated["sim"],
    )
    # Uncapped on purpose: every cross-operator twin is a lead worth surfacing, and the count is
    # already bounded by this operator's own block count (RED-1 is likewise uncapped).
    ranked = sorted(hits, key=lambda n: (-hits[n][1][0][0], hits[n][0].start, n))
    red2 = []
    for name in ranked:
        a, partners = hits[name]
        ratio, other_path, other_start = partners[0]
        more = f" (+{len(partners) - 1} more)" if len(partners) > 1 else ""
        red2.append(
            Finding(
                "RED-2",
                "impl",
                REC,
                f"'{a.name}' is duplicated in {len(partners)} other operator file(s) "
                f"(jaccard {ratio:.2f}) — hoist to a shared priv header",
                f"{rel(owner[id(a)])}:{a.start} vs {rel(other_path)}:{other_start}{more}",
                g,
                "Hoist the shared body into src/cvcuda/priv/<Name>.cuh (namespace "
                "cvcuda::priv::<area>) and include it from every consumer; priv headers need no "
                "CMake edit (cf. AdjustColorCommon.cuh, OpHQResizePlanar.cuh).",
            )
        )
    out += red2
    if not red2:
        out.append(
            Finding(
                "RED-2",
                "impl",
                PASS,
                "No priv block duplicated in another operator's priv",
                f"{len(flat)} block(s) vs {len(others)} cross-operator block(s) "
                f"at jaccard>={curated['sim']:.2f}",
                g,
            )
        )

    # RED-3: a dispatch guard narrower than the dtypes the operator declares. A vectorized path
    # gated on `sizeof(BT) == 1 || sizeof(BT) == 4` silently excludes every 2-byte type, which
    # then falls back to the scalar kernel — a redundant slow path kept alive by the gate, not by
    # any property of the code behind it. Reported as MANUAL: the tool proves the guard is
    # narrower than the declared support, but only a human/agent can confirm the body would in
    # fact compile and stay bit-exact for the excluded widths.
    declared = parse_limitations(reader(P.header) or "").get("input", {}).get("dtypes")
    red3 = []
    if declared:
        want = {DTYPE_BYTES[d] for d in declared if d in DTYPE_BYTES}
        for p, t in priv_texts:
            # Read the whole parenthesised condition, not one line: a guard wrapped as
            # `if (sizeof(BT) == 1 ||` / `sizeof(BT) == 4)` would otherwise be seen as admitting
            # only the first width, and reported as excluding a width it actually admits.
            for off, cond in if_conditions(_strip_code(t or "")):
                admitted = {
                    int(n)
                    for n in re.findall(r"sizeof\s*\(\s*\w+\s*\)\s*==\s*(\d)", cond)
                }
                if not admitted:
                    continue
                missing = want - admitted
                if missing:
                    ln = (t or "")[:off].count("\n") + 1
                    widths = ", ".join(f"{w}-byte" for w in sorted(missing))
                    names = ", ".join(
                        sorted(d for d in declared if DTYPE_BYTES.get(d) in missing)
                    )
                    red3.append((p, ln, " ".join(cond.split()), widths, names))
    if red3:
        p, ln, line, widths, names = red3[0]
        out.append(
            Finding(
                "RED-3",
                "impl",
                MANUAL,
                f"Dispatch guard admits fewer widths than the operator declares — "
                f"{widths} ({names}) excluded ({len(red3)} site[s])",
                f"{rel(p)}:{ln}: {line.strip()[:100]}",
                g,
                "Check whether the excluded widths work behind this gate; if so widen it, which "
                "removes the fallback path rather than adding code. If they genuinely cannot, "
                "say why in a comment so the next reader does not re-test it.",
            )
        )
    else:
        out.append(
            Finding(
                "RED-3",
                "impl",
                PASS,
                "No dispatch guard narrower than the declared dtype support",
                (
                    f"declared widths {sorted({DTYPE_BYTES[d] for d in declared if d in DTYPE_BYTES})}"
                    if declared
                    else "no declared dtype table to compare against"
                ),
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

    # RED-12: a shared util reinvented under a *different* name. RED-10 matches by name, so it
    # sees nothing when the local spelling differs — which is the common case, since a helper that
    # collided by name would have been caught in review. Matching is by shape instead.
    for shape in REINVENTION_SHAPES:
        sites = []
        for p, t in surface:
            # Match comment- and string-stripped source: a commented-out helper or a snippet in a
            # doc comment is not a live reimplementation, and _strip_code preserves newlines so
            # the line number is still the real one.
            code = _strip_code(t or "")
            for m in shape["rx"].finditer(code):
                sites.append((p, code[: m.start()].count("\n") + 1, m.group(0)))
        if sites:
            p, ln, snippet = sites[0]
            out.append(
                Finding(
                    "RED-12",
                    "xcut",
                    MANUAL,
                    f"'{shape['name']}' reimplements {shape['canonical']} ({len(sites)} site[s])",
                    f"{rel(p)}:{ln}: " + " ".join(snippet.split())[:100],
                    g,
                    shape["fix"],
                )
            )
    if not any(f.id == "RED-12" for f in out):
        out.append(
            Finding(
                "RED-12",
                "xcut",
                PASS,
                "No shape-matched reimplementation of a canonical shared util",
                f"{len(REINVENTION_SHAPES)} shape(s) checked",
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
            "With no build tree or GPU, gather device-code evidence first: "
            "python3 tools/device_code_proof.py %s --base %s. It is evidence for VER-6 on the "
            "arch(es) it compiles, not a replacement for this test run." % (P.Op, base),
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


def _open_red_counts(P, curated, reader):
    """RED-* id -> how many open (RECOMMENDATION/MANUAL) findings it has under `reader`.

    Counted rather than merely collected because an id can be *partially* resolved: RED-2 reports
    one finding per duplicated body, so a refactor that hoists four of five leaves the id open
    while still being the whole point of the change. Presence alone cannot express that.
    """
    counts = {}
    for f in run_assess(P, DOMAINS, curated, reader):
        if f.status in (REC, MANUAL):
            counts[f.id] = counts.get(f.id, 0) + 1
    return counts


def _open_red_ids(P, curated, reader):
    """The set of RED-* ids with a non-PASS, non-N-A finding under `reader` — i.e. open
    redundancy. Used to quantify before-vs-after when verifying an applied refactor."""
    return set(_open_red_counts(P, curated, reader))


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
    base_counts = _open_red_counts(P, curated, lambda p: git_show(base, rel(p)))
    now_counts = _open_red_counts(P, curated, read)
    base_open, now_open = set(base_counts), set(now_counts)
    return {
        "loc_insertions": ins,
        "loc_deletions": dels,
        "loc_net": ins - dels,
        "redundancy_resolved": sorted(base_open - now_open),
        "redundancy_introduced": sorted(now_open - base_open),
        "redundancy_open_now": sorted(now_open),
        # Still open, but with strictly fewer findings than at <base>: partial progress that
        # `redundancy_resolved` (a set difference over ids) cannot represent.
        "redundancy_reduced": {
            i: [base_counts[i], now_counts[i]]
            for i in sorted(base_counts)
            if i in now_open and now_counts[i] < base_counts[i]
        },
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
        reduced = (
            ", ".join(
                f"{i} {a}->{b}" for i, (a, b) in summary["redundancy_reduced"].items()
            )
            or "none"
        )
        lines += [
            "",
            "## refactor summary",
            f"- LOC delta (impl/binding): +{summary['loc_insertions']} / "
            f"-{summary['loc_deletions']} (net {net:+d})",
            f"- redundancy resolved: {resolved}",
            f"- redundancy reduced (still open): {reduced}",
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


def block_variants(name, curated, reader=read, floor=None):
    """Every definition of `name` across the priv corpus, grouped into equivalence classes.

    Answers "how many genuinely different versions of this helper exist, and who shares each" —
    the question that has to be settled before hoisting anything, and that is otherwise done by
    reading N copies side by side. Classes are formed greedily against a representative, which is
    stable because the members of a real clone family sit at ~1.00 to each other.

    The floor defaults to 1 here, not to MIN_BLOCK_LINES. The floor exists to keep *detection*
    quiet, and this is a lookup: the caller already named the block, so a short body is the answer
    to their question rather than noise. Defaulting to the detection floor made a five-line helper
    report as "not found" moments after a sweep had reported it.
    """
    found = []
    for p in priv_corpus_paths():
        for b in extract_blocks_cached(
            p, reader(p), "cpp", 1 if floor is None else floor
        ):
            if b.name == name:
                found.append((p, b))
    classes = []
    for p, b in sorted(found, key=lambda t: (rel(t[0]), t[1].start)):
        for c in classes:
            if jaccard(c["rep"].shingles, b.shingles) >= curated["sim"]:
                c["members"].append((p, b))
                break
        else:
            classes.append({"rep": b, "members": [(p, b)]})
    return classes


def render_variants_md(name, classes, sim):
    lines = [f"# refactor-op: variants of `{name}`", ""]
    total = sum(len(c["members"]) for c in classes)
    if not classes:
        return "\n".join(
            lines + [f"No block named `{name}` found in src/cvcuda/priv/**."]
        )
    lines.append(
        f"{total} definition(s) in {len({rel(p) for c in classes for p, _ in c['members']})} "
        f"file(s), forming **{len(classes)} distinct variant(s)** at jaccard>={sim:.2f}."
    )
    lines.append("")
    for i, c in enumerate(sorted(classes, key=lambda c: -len(c["members"])), 1):
        rep = c["rep"]
        lines.append(
            f"## variant {i} — {len(c['members'])} copy(ies), {len(rep.norm)} lines"
        )
        for p, b in sorted(c["members"], key=lambda t: rel(t[0])):
            lines.append(f"- `{rel(p)}:{b.start}`")
        lines.append("")
    if len(classes) > 1:
        lines.append(
            "More than one variant: they are **not** interchangeable. Hoist the largest class "
            "first and leave the outliers, or reconcile them deliberately — a single helper "
            "covering every variant usually needs a flag per difference, which is worse than "
            "the duplication."
        )
    return "\n".join(lines)


def simulate_hoist(names, curated, reader=read):
    """Predict the RED-2 delta if `names` were hoisted out of every operator that defines them.

    This is the question a refactor plan turns on — "after this change, does the finding actually
    resolve?" — and answering it by hand means hand-simulating the post-change tree, which is
    exactly where a plan silently goes wrong.
    """
    floor = curated.get("min_block_lines", MIN_BLOCK_LINES)
    corpus = {
        p: extract_blocks_cached(p, reader(p), "cpp", floor)
        for p in priv_corpus_paths()
    }
    after = {p: [b for b in bs if b.name not in names] for p, bs in corpus.items()}
    rows = []
    for op in sorted(all_op_names()):
        P = resolve_op(op)
        if not P.priv:
            continue
        mine_paths = {p.resolve() for p in P.priv}
        allow = set(curated["allowlist"].get(P.op, []))

        def count(state):
            mine = [
                b
                for p, bs in state.items()
                if p.resolve() in mine_paths
                for b in bs
                if b.name not in allow
            ]
            others = [
                (p, b)
                for p, bs in state.items()
                if p.resolve() not in mine_paths
                for b in bs
            ]
            return cross_op_duplicates(mine, others, curated["sim"])

        before, now = count(corpus), count(after)
        if before or now:
            rows.append(
                {
                    "Op": P.Op,
                    "before": len(before),
                    "after": len(now),
                    "residual": sorted(now),
                }
            )
    # The simulation is keyed on the block *name*, so it models "every definition of this name
    # goes away". That is only true if every definition is the same code. Where a name has more
    # than one variant the prediction is optimistic — one shared function cannot replace two
    # different bodies without a parameter for the difference — so surface that here rather than
    # letting a plan be written against a number that assumed otherwise.
    split = {}
    for n in sorted(names):
        classes = block_variants(n, curated, reader)
        if len(classes) > 1:
            split[n] = [
                sorted(f"{rel(p)}:{b.start}" for p, b in c["members"]) for c in classes
            ]
    return rows, split


def render_hoist_md(names, rows, split):
    lines = [f"# refactor-op: simulated hoist of {', '.join(sorted(names))}", ""]
    if split:
        lines.append(
            "> **These counts are optimistic.** The simulation removes every definition"
        )
        lines.append(
            "> of each name, but these names have more than one version, so no single"
        )
        lines.append("> hoisted function replaces all their copies as written:")
        for n, classes in split.items():
            joined = "; ".join("{" + ", ".join(c) + "}" for c in classes)
            lines.append(f"> - `{n}` — {len(classes)} variants: {joined}")
        lines.append("> Reconcile or exclude the outliers first (`--variants <name>`).")
        lines.append("")
    changed = [r for r in rows if r["after"] != r["before"]]
    if not changed:
        return "\n".join(
            lines + ["No operator's RED-2 count changes — nothing to hoist."]
        )
    lines += ["| operator | RED-2 before | after | residual |", "|---|---:|---:|---|"]
    for r in sorted(changed, key=lambda r: (-(r["before"] - r["after"]), r["Op"])):
        lines.append(
            f"| {r['Op']} | {r['before']} | {r['after']} | "
            f"{', '.join(r['residual']) or '—'} |"
        )
    cleared = [r["Op"] for r in changed if r["after"] == 0]
    lines += [
        "",
        f"{len(changed)} operator(s) improve; {len(cleared)} reach zero"
        + (f" ({', '.join(cleared)})" if cleared else "")
        + ".",
    ]
    stuck = sorted({n for r in rows for n in r["residual"]})
    if stuck:
        lines.append(
            f"Residual names after the hoist: {', '.join(stuck)} — an operator whose count drops "
            "but stays above zero reports `redundancy reduced`, not `resolved`."
        )
    return "\n".join(lines)


def render_all_md(rows, domains):
    """Ranked cross-operator roll-up. Operators with no open finding are summarised as a count
    rather than listed, so the report leads with where the work is."""
    lines = ["# refactor-op: all operators  phase=assess", ""]
    ranked = sorted(rows, key=lambda r: (-r["open"], r["Op"]))
    busy = [r for r in ranked if r["open"]]
    lines.append(
        f"{len(rows)} operator(s), {sum(r['open'] for r in rows)} open finding(s) in "
        f"{len(busy)}; {len(rows) - len(busy)} clean. domains={','.join(domains)}"
    )
    lines.append("")
    if busy:
        lines += ["| operator | open | by id |", "|---|---:|---|"]
        for r in busy:
            by = ", ".join(f"{i} x{n}" for i, n in sorted(r["by_id"].items()))
            lines.append(f"| {r['Op']} | {r['open']} | {by} |")
        lines.append("")
    totals = {}
    for r in rows:
        for i, n in r["by_id"].items():
            totals[i] = totals.get(i, 0) + n
    lines.append("## totals by finding id")
    for i, n in sorted(totals.items()):
        ops = sum(1 for r in rows if r["by_id"].get(i))
        lines.append(f"- `{i}` — {n} finding(s) across {ops} operator(s)")
    if not totals:
        lines.append("- none")
    lines += [
        "",
        "Assess is advisory. Re-run per operator for evidence lines and fixes.",
    ]
    return "\n".join(lines)


def run_all_operators(domains, curated, fmt):
    """Assess every operator in one process. The per-operator tools are single-operator by
    design, which pushes any cross-operator question into throwaway shell loops; this keeps the
    corpus parse and the block cache warm across the sweep instead."""
    rows = []
    for name in sorted(all_op_names()):
        P = resolve_op(name)
        # No `if not P.priv: continue` — the roll-up must mirror the per-operator reports it
        # replaces, and an operator without priv sources can still carry api (RED-6) and xcut
        # (RED-10/11/12) findings. check_impl already degrades to N-A on its own.
        by_id = {}
        for f in run_assess(P, domains, curated):
            if f.status in (REC, MANUAL):
                by_id[f.id] = by_id.get(f.id, 0) + 1
        rows.append(
            {"Op": P.Op, "op": P.op, "open": sum(by_id.values()), "by_id": by_id}
        )
    if fmt == "json":
        return json.dumps(
            {"phase": "assess", "domains": list(domains), "operators": rows},
            indent=2,
            sort_keys=True,
        )
    return render_all_md(rows, domains)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic per-operator refactoring/redundancy checker (CV-CUDA)."
    )
    ap.add_argument(
        "operator",
        nargs="?",
        help="Operator name (PascalCase, e.g. BrightnessContrast); omit with --all-operators",
    )
    ap.add_argument(
        "--all-operators",
        action="store_true",
        help="assess every operator and emit a ranked roll-up instead of one report",
    )
    ap.add_argument(
        "--variants",
        metavar="BLOCK",
        help="group every definition of BLOCK across priv into equivalence classes — how many "
        "genuinely different versions exist, and which files share each",
    )
    ap.add_argument(
        "--simulate-hoist",
        metavar="BLOCK[,BLOCK...]",
        help="predict the RED-2 delta if these blocks were hoisted to a shared header",
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
    ap.add_argument(
        "--min-block-lines",
        type=int,
        default=None,
        metavar="N",
        help=f"smallest block body to fingerprint (default {MIN_BLOCK_LINES}); sweep downwards "
        "to surface small shared helpers — one-liners are matched whole, on exact equality",
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

    if args.variants or args.simulate_hoist:
        # These scan the whole corpus, so a positional operator would read as scoping the run
        # while silently doing nothing. Reject it rather than ignore it.
        if args.operator:
            ap.error(
                "--variants/--simulate-hoist are whole-corpus; drop the operator argument"
            )
        if args.phase != "assess":
            ap.error("--variants/--simulate-hoist support --phase assess only")
        curated = load_curated()
        if args.min_block_lines is not None:
            if args.min_block_lines < 1:
                ap.error("--min-block-lines must be >= 1")
            curated["min_block_lines"] = args.min_block_lines
        if args.variants:
            classes = block_variants(args.variants, curated)
            report = (
                json.dumps(
                    {
                        "block": args.variants,
                        "variants": [
                            {
                                "lines": len(c["rep"].norm),
                                "members": sorted(
                                    f"{rel(p)}:{b.start}" for p, b in c["members"]
                                ),
                            }
                            for c in classes
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                if args.format == "json"
                else render_variants_md(args.variants, classes, curated["sim"])
            )
        else:
            names = {n.strip() for n in args.simulate_hoist.split(",") if n.strip()}
            if not names:
                ap.error("--simulate-hoist needs at least one block name")
            rows, split = simulate_hoist(names, curated)
            report = (
                json.dumps(
                    {
                        "hoisted": sorted(names),
                        "operators": rows,
                        "multi_variant": split,
                    },
                    indent=2,
                    sort_keys=True,
                )
                if args.format == "json"
                else render_hoist_md(names, rows, split)
            )
        print(report)
        if args.out:
            Path(args.out).write_text(report + "\n", encoding="utf-8")
        return 0

    if args.all_operators:
        if args.operator:
            ap.error("--all-operators takes no operator argument")
        if args.phase != "assess":
            ap.error(
                "--all-operators supports --phase assess only (verify needs one operator)"
            )
        curated = load_curated()
        if args.min_block_lines is not None:
            if args.min_block_lines < 1:
                ap.error("--min-block-lines must be >= 1")
            curated["min_block_lines"] = args.min_block_lines
        report = run_all_operators(domains, curated, args.format)
        print(report)
        if args.out:
            Path(args.out).write_text(report + "\n", encoding="utf-8")
        return 0

    if not args.operator:
        ap.error("an operator name is required (or use --all-operators)")

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
    if args.min_block_lines is not None:
        if args.min_block_lines < 1:
            ap.error("--min-block-lines must be >= 1")
        curated["min_block_lines"] = args.min_block_lines
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
