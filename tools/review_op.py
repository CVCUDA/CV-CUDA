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
"""Deterministic per-operator review checker for CV-CUDA.

Implements the checklist defined in .agents/guidance/REVIEW_OP_GUIDELINES.md across four domains
(support / test / bench / docs). Emits a structured report (markdown or json) with a
status + evidence + guideline id per item, and exits non-zero on any GAP.

The checker is read-only, deterministic and idempotent: no network, no clocks, no
randomness, so the same tree yields a byte-identical report. Corrective actions are named
per finding (the `fix` field); applying them is the wrapper/agent's job (see the skills).

Usage:
    python3 tools/review_op.py <Operator> [--domain support|test|bench|docs|all]
                               [--format md|json] [--out PATH] [--run] [--fix]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REVIEW_OP_GUIDE = ".agents/guidance/REVIEW_OP_GUIDELINES.md"
CURATED_DATA = ".agents/tools/review_op_data.json"
AGENT_TOOLS = REPO / ".agents" / "tools"
sys.path.insert(0, str(AGENT_TOOLS))
sys.path.insert(0, str(REPO / "bench"))
from _internal.baselines import (  # noqa: E402
    BaselineError,
    expected_case_keys_for_entry,
    fake_planar_pairing_issues,
)
from operator_source_map import (  # noqa: E402
    SHARED_KERNEL_SOURCES,
    all_op_names,
    legacy_belongs,
)

PASS, GAP, NA, MANUAL, REC = "PASS", "GAP", "N-A", "MANUAL", "RECOMMENDATION"
DOMAINS = ("support", "test", "bench", "docs")

# dtype table rows in the Doxygen Limitations table -> canonical dtype tokens
DTYPE_ROW = {
    ("8bit", "Unsigned"): "u8",
    ("8bit", "Signed"): "s8",
    ("16bit", "Unsigned"): "u16",
    ("16bit", "Signed"): "s16",
    ("32bit", "Unsigned"): "u32",
    ("32bit", "Signed"): "s32",
    ("16bit", "Float"): "f16",
    ("32bit", "Float"): "f32",
    ("64bit", "Float"): "f64",
}


@dataclass
class Finding:
    id: str
    domain: str
    status: str
    summary: str
    evidence: str = ""
    guideline: str = ""
    fix: str = ""


@dataclass
class OpPaths:
    op: str
    Op: str
    pyname: str
    header: Path
    hpp: Path
    pybind: Path
    test_cpp: Path
    test_py: Path
    bench_cpp: Path
    bench_py: Path
    bench_cfg: Path
    priv: list = field(default_factory=list)


@dataclass(frozen=True)
class SubmitSignature:
    """Container-relevant facts parsed from one public C Submit declaration."""

    name: str
    primary_container: str | None
    line: int
    declaration: str


@dataclass
class CAPIContainers:
    """C-API input-container support derived from Submit signatures."""

    tensor: list[SubmitSignature] = field(default_factory=list)
    varshape: list[SubmitSignature] = field(default_factory=list)
    generic_varshape: list[SubmitSignature] = field(default_factory=list)


# --------------------------------------------------------------------------- io helpers
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


def grep(pattern: str, text, flags=0):
    """Return list of (lineno, line) matching pattern. Empty if text is None."""
    if not text:
        return []
    rx = re.compile(pattern, flags)
    return [(i + 1, ln) for i, ln in enumerate(text.splitlines()) if rx.search(ln)]


def first_evidence(path: Path, hits):
    if not hits:
        return ""
    ln, line = hits[0]
    return f"{rel(path)}:{ln}: {line.strip()[:120]}"


def parse_submit_signatures(Op: str, header_text: str | None) -> list[SubmitSignature]:
    """Parse this operator's public Submit declarations and their primary data input.

    The primary container is the first Tensor/TensorBatch/ImageBatch handle after the
    operator and stream parameters. This deliberately ignores tensor outputs and auxiliary
    tensors: legacy APIs such as CropFlipNormalizeReformat and PadAndStack have an image-batch
    input, a tensor output, and no Tensor-input variant.
    """
    if not header_text:
        return []

    pattern = re.compile(
        rf"\bCVCUDA_PUBLIC\s+NVCVStatus\s+"
        rf"(cvcuda{re.escape(Op)}(?:VarShape|ImageBatch|TensorBatch)?Submit)\s*\((.*?)\)\s*;",
        re.I | re.S,
    )
    handle_pattern = re.compile(r"\b(NVCV(?:ImageBatch|TensorBatch|Tensor)Handle)\b")
    signatures = []
    for match in pattern.finditer(header_text):
        handles = handle_pattern.findall(match.group(2))
        signatures.append(
            SubmitSignature(
                name=match.group(1),
                primary_container=handles[0] if handles else None,
                line=header_text.count("\n", 0, match.start(1)) + 1,
                declaration=" ".join(match.group(0).split()),
            )
        )
    return signatures


def detect_c_api_containers(Op: str, header_text: str | None) -> CAPIContainers:
    """Classify Tensor vs VarShape support from public C Submit signatures.

    A generic ``Submit`` is classified by its primary input handle. A named
    ``VarShapeSubmit`` and ``ImageBatchSubmit`` remain explicit VarShape forms,
    including non-image TensorBatch APIs whose established public name carries
    that distinction.
    """
    result = CAPIContainers()
    generic_name = f"cvcuda{Op}Submit".lower()
    for signature in parse_submit_signatures(Op, header_text):
        lower_name = signature.name.lower()
        is_named_varshape = lower_name.endswith(("varshapesubmit", "imagebatchsubmit"))
        if is_named_varshape or signature.primary_container == "NVCVImageBatchHandle":
            result.varshape.append(signature)
            if signature.name.lower() == generic_name:
                result.generic_varshape.append(signature)
        elif signature.primary_container in {
            "NVCVTensorHandle",
            "NVCVTensorBatchHandle",
        }:
            result.tensor.append(signature)
    return result


def submit_evidence(path: Path, signature: SubmitSignature) -> str:
    return f"{rel(path)}:{signature.line}: {signature.declaration[:180]}"


# --------------------------------------------------------------------- operator resolution
def resolve_op(arg: str):
    """Resolve <Operator> to op/Op/pyname + file paths (case-insensitive on the header)."""
    stem = arg.strip()
    op = stem.lower()
    hdr_dir = REPO / "src/cvcuda/include/cvcuda"
    header = None
    Op = stem
    if hdr_dir.is_dir():
        for h in sorted(hdr_dir.glob("Op*.h")):
            if h.stem[2:].lower() == op:  # "Op<Name>" -> <Name>
                header = h
                Op = h.stem[2:]
                break
    if header is None:
        header = hdr_dir / f"Op{stem}.h"
        Op = stem

    pybind = REPO / f"python/mod_cvcuda/operators/Op{Op}.cpp"
    pyname = resolve_pyname(pybind, op)

    priv = []
    priv_dir = REPO / "src/cvcuda/priv"
    for cand in [priv_dir / f"Op{Op}.cu", priv_dir / f"Op{Op}.cpp"]:
        if cand.exists():
            priv.append(cand)
    legacy = priv_dir / "legacy"
    if legacy.is_dir():
        all_ops = all_op_names()
        for g in sorted(legacy.glob("*.c*")):
            if legacy_belongs(g.stem, op, all_ops):
                priv.append(g)
    # Shared/legacy kernel sources not matched by the op-name globs (e.g. warp.cu for the
    # Warp* ops, the HQResize kernel headers) — see SHARED_KERNEL_SOURCES.
    for extra in SHARED_KERNEL_SOURCES.get(op, []):
        cand = priv_dir / extra
        if cand.exists() and cand not in priv:
            priv.append(cand)

    # Python tests are named after the flattened op name (mkop.sh scaffolds
    # test_op$namelower.py); fall back to the pybind API name only for the few
    # legacy files that use it (e.g. test_opnms.py, test_opmatch.py).
    test_py = REPO / f"tests/cvcuda/python/test_op{op}.py"
    if not test_py.exists():
        alt = REPO / f"tests/cvcuda/python/test_op{pyname}.py"
        if alt.exists():
            test_py = alt

    return OpPaths(
        op=op,
        Op=Op,
        pyname=pyname,
        header=header,
        hpp=hdr_dir / f"Op{Op}.hpp",
        pybind=pybind,
        test_cpp=REPO / f"tests/cvcuda/system/TestOp{Op}.cpp",
        test_py=test_py,
        bench_cpp=REPO / f"bench/cpp/ops/Bench{Op}.cpp",
        bench_py=REPO / f"bench/python/ops/bench_{op}.py",
        bench_cfg=REPO / f"bench/config/operators/{op}.json",
        priv=priv,
    )


def resolve_pyname(pybind: Path, op: str) -> str:
    text = read(pybind)
    if text:
        names = re.findall(r'm\.def\(\s*"([a-z0-9_]+)"', text)
        base = [n for n in names if not n.endswith("_into")]
        if base:
            # the shortest base name is the operator's primary function
            return sorted(base, key=len)[0]
    return op


# --------------------------------------------------------------- deterministic curated data
_MISSING = object()


def _curated_schema_error(field: str, expected: str) -> None:
    raise ValueError(f"{CURATED_DATA}: `{field}` must be {expected}")


def _curated_string_list(data: dict, field: str) -> list[str]:
    value = data.get(field, _MISSING)
    if value is _MISSING:
        return []
    if not isinstance(value, list):
        _curated_schema_error(field, "a list of strings")
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            _curated_schema_error(f"{field}[{idx}]", "a string")
    return value


def _curated_basic_expected(data: dict) -> dict[str, list[str]]:
    value = data.get("basic_expected", _MISSING)
    if value is _MISSING:
        return {}
    if not isinstance(value, dict):
        _curated_schema_error(
            "basic_expected", "an object mapping operator names to string lists"
        )
    normalized = {}
    for op, expected in value.items():
        if not isinstance(op, str):
            _curated_schema_error("basic_expected key", "a string")
        if not isinstance(expected, list):
            _curated_schema_error(f"basic_expected.{op}", "a list of strings")
        for idx, item in enumerate(expected):
            if not isinstance(item, str):
                _curated_schema_error(f"basic_expected.{op}[{idx}]", "a string")
        normalized[op.lower()] = expected
    return normalized


def load_curated():
    """Load non-derivable review facts without putting operator names in guidance."""
    data = load_bench_cfg(REPO / CURATED_DATA)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        _curated_schema_error("root", "an object")
    tensor_only = {op.lower() for op in _curated_string_list(data, "tensor_only")}
    varshape_only = {op.lower() for op in _curated_string_list(data, "varshape_only")}
    bench_layout_na = {
        op.lower() for op in _curated_string_list(data, "bench_layout_na")
    }
    bench_rgb_na = {op.lower() for op in _curated_string_list(data, "bench_rgb_na")}
    basic_expected = _curated_basic_expected(data)
    return tensor_only, varshape_only, basic_expected, bench_layout_na, bench_rgb_na


# --------------------------------------------------------------- Limitations table parsing
def parse_limitations(header_text):
    """Parse the Input Limitations block -> {layouts:set, channels:set, dtypes:set} or None."""
    if not header_text:
        return None
    # isolate the Input: ... up to Output: (the first Limitations block)
    m = re.search(
        r"Limitations:(.*?)(?:\n\s*\*\s*Input/Output dependency|\*/)", header_text, re.S
    )
    blob = m.group(1) if m else header_text
    inp = re.search(r"Input:(.*?)(?:Output:|$)", blob, re.S)
    region = inp.group(1) if inp else blob

    lay = re.search(r"Data Layout:\s*\[([^\]]*)\]", region)
    layouts = set()
    if lay:
        for tok in re.findall(r"(?:k|NVCV_TENSOR_)?([A-Z]+)", lay.group(1)):
            if tok in {"HW", "NHW", "HWC", "NHWC", "CHW", "NCHW"}:
                layouts.add(tok)
    # Planar layouts may be declared in prose/macro form (e.g. HQResize's
    # "NVCV_TENSOR_[N]CHW (planar, 2D only)") that the [kNCHW,...] bracket parse above misses.
    if re.search(r"NVCV_TENSOR_[\[\]A-Z]*CHW", region):
        layouts.update({"NCHW", "CHW"})
    ch = re.search(r"Channels:\s*\[([^\]]*)\]", region)
    channels = set()
    if ch:
        for tok in re.findall(r"\d+", ch.group(1)):
            channels.add(int(tok))
    dtypes = set()
    for bits, sign, allowed in re.findall(
        r"(\d+bit)\s+(Unsigned|Signed|Float)\s*\|\s*(Yes|No)", region
    ):
        if allowed == "Yes":
            key = DTYPE_ROW.get((bits, sign))
            if key:
                dtypes.add(key)
    if not layouts and not channels and not dtypes:
        return None
    return {"layouts": layouts, "channels": channels, "dtypes": dtypes}


def parse_planar_policy(header_text):
    """Parse an operator-local planar inapplicability declaration from C-API Doxygen."""
    policy = {
        "not_applicable": False,
        "reason": "",
        "error": "",
        "line": None,
    }
    if not header_text:
        return policy

    lines = header_text.splitlines()
    declarations = []
    for idx, line in enumerate(lines):
        clean = re.sub(r"^\s*(?:/\*+|\*+)?\s?", "", line).strip()
        if clean.startswith("Planar image layouts:"):
            declarations.append((idx, clean.split(":", 1)[1].strip()))

    if not declarations:
        return policy
    if len(declarations) != 1:
        policy["error"] = "declare 'Planar image layouts' exactly once"
        return policy

    idx, value = declarations[0]
    policy["line"] = idx + 1
    if value != "Not applicable":
        policy["error"] = (
            "the only supported declaration is "
            "'Planar image layouts: Not applicable'"
        )
        return policy

    for following_idx, following in enumerate(lines):
        if following_idx <= idx:
            continue
        clean = re.sub(r"^\s*(?:/\*+|\*+)?\s?", "", following).strip()
        if not clean:
            continue
        if clean.startswith("Reason:"):
            policy["reason"] = clean.split(":", 1)[1].strip()
        break

    if not policy["reason"]:
        policy["error"] = "a planar inapplicability declaration requires a Reason"
        return policy

    policy["not_applicable"] = True
    return policy


def planar_policy_verdict(P: OpPaths, header_text, declared):
    """Require channel-first image layouts unless the operator declares why they do not apply."""
    policy = parse_planar_policy(header_text)
    location = rel(P.header)
    if policy["line"]:
        location += f":{policy['line']}"

    if policy["error"]:
        return Finding(
            "SUP-10",
            "support",
            GAP,
            "Invalid planar layout policy declaration",
            f"{location}: {policy['error']}",
            REVIEW_OP_GUIDE,
            "Use the exact operator-local declaration and provide a non-empty Reason.",
        )
    if policy["not_applicable"] and declared:
        return Finding(
            "SUP-10",
            "support",
            GAP,
            "Planar layouts are both declared and marked not applicable",
            f"{location}: {policy['reason']}",
            REVIEW_OP_GUIDE,
            "Remove the inapplicability declaration or remove NCHW/CHW from the support contract.",
        )
    if policy["not_applicable"]:
        return Finding(
            "SUP-10",
            "support",
            NA,
            "Planar image layouts are not applicable",
            f"{location}: {policy['reason']}",
            REVIEW_OP_GUIDE,
        )
    if declared:
        return Finding(
            "SUP-10",
            "support",
            PASS,
            "Planar image layouts are declared",
            "header Limitations include NCHW/CHW",
            REVIEW_OP_GUIDE,
        )
    return Finding(
        "SUP-10",
        "support",
        GAP,
        "Planar image layouts are required by default",
        f"{location}: no NCHW/CHW layout or inapplicability declaration",
        REVIEW_OP_GUIDE,
        "Add NCHW/CHW support or declare 'Planar image layouts: Not applicable' with a Reason.",
    )


# --------------------------------------------------------------------- bench config helpers
def load_bench_cfg(path: Path):
    text = read(path)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def rows_for_entry(entry: dict) -> int:
    count = len(entry.get("dtypes", [])) or 1
    for grp in ("string_axes", "int64_axes", "float64_axes"):
        for vals in entry.get(grp, {}).values():
            count *= max(len(vals), 1)
    return count


# =========================================================================== SUPPORT domain
def check_support(P: OpPaths, curated):
    tensor_only, varshape_only, _, bench_layout_na, bench_rgb_na = curated
    out = []
    htext = read(P.header)
    g = ".agents/guidance/REVIEW_OP_GUIDELINES.md#support / make_operator.rst"

    # Case-insensitive: C symbol casing can differ from the header stem (HQResize ->
    # cvcudaHqResizeSubmit). Classify by the primary data-input handle rather than the symbol
    # suffix because a few legacy generic Submit APIs take ImageBatch input and Tensor output.
    capi = detect_c_api_containers(P.Op, htext)
    has_tensor = bool(capi.tensor)
    has_vs = bool(capi.varshape)
    generic_varshape = bool(capi.generic_varshape)
    tensor_absence_is_declared = generic_varshape or P.op in varshape_only
    if has_tensor:
        tensor_status = PASS
        tensor_summary = "Tensor container declared (cvcuda%sSubmit)" % P.Op
        tensor_evidence = first_evidence(
            P.header, grep(rf"cvcuda{P.Op}Submit", htext, re.I)
        )
    elif generic_varshape:
        tensor_status = NA
        tensor_summary = (
            "Tensor absent; generic Submit declares a VarShape primary input"
        )
        tensor_evidence = submit_evidence(P.header, capi.generic_varshape[0])
    elif P.op in varshape_only:
        tensor_status = NA
        tensor_summary = (
            "Tensor absent; op is marked var-shape-only in deterministic review data"
        )
        tensor_evidence = f"{rel(P.header)} (not found)"
    else:
        tensor_status = GAP
        tensor_summary = "Tensor container missing"
        tensor_evidence = f"{rel(P.header)} (not found)"
    out.append(
        Finding(
            "SUP-1",
            "support",
            tensor_status,
            tensor_summary,
            tensor_evidence,
            g,
            ""
            if has_tensor or tensor_absence_is_declared
            else "Declare the Tensor Submit entry point.",
        )
    )

    if has_vs:
        out.append(
            Finding(
                "SUP-2",
                "support",
                PASS,
                "VarShape container declared",
                submit_evidence(P.header, capi.varshape[0]),
                g,
            )
        )
    elif P.op in tensor_only:
        out.append(
            Finding(
                "SUP-2",
                "support",
                NA,
                "VarShape absent; op is marked tensor-only in deterministic review data",
                "",
                g,
            )
        )
    else:
        out.append(
            Finding(
                "SUP-2",
                "support",
                GAP,
                "VarShape overload absent and op not marked tensor-only in deterministic review data",
                f"{rel(P.header)}: no cvcuda{P.Op}VarShapeSubmit",
                g,
                "Add VarShape support, or record deterministic/operator-local evidence "
                "that the op is tensor-only.",
            )
        )

    # SUP-3 C++ .hpp overloads
    hpptext = read(P.hpp)
    if hpptext is None:
        out.append(
            Finding(
                "SUP-3",
                "support",
                GAP,
                "C++ .hpp not found",
                f"{rel(P.hpp)} (missing)",
                g,
                "Add the C++ operator header.",
            )
        )
    else:
        hpp_tensor = bool(
            re.search(
                r"operator\([^;{]*\bnvcv::Tensor\s*&\s*(?:in|src|input)[^;{]*"
                r"\bnvcv::Tensor\s*&\s*(?:out|dst|output)",
                hpptext,
                re.S,
            )
        )
        # C++ var-shape overloads take nvcv::ImageBatch& or nvcv::ImageBatchVarShape& (param, not the include)
        hpp_vs = bool(grep(r"ImageBatch\w*\s*&", hpptext))
        ok = (hpp_tensor == has_tensor) and (hpp_vs == has_vs)
        out.append(
            Finding(
                "SUP-3",
                "support",
                PASS if ok else MANUAL,
                "C++ .hpp operator() overloads match C-API containers",
                f"{rel(P.hpp)}: hpp(T={hpp_tensor},VS={hpp_vs}) capi(T={has_tensor},VS={has_vs})",
                g,
                "" if ok else "Reconcile C++ overloads with the C-API container set.",
            )
        )

    # SUP-4 Python allocating + _into
    pytext = read(P.pybind)
    if pytext is None:
        out.append(
            Finding(
                "SUP-4",
                "support",
                GAP,
                "Python binding not found",
                f"{rel(P.pybind)} (missing)",
                g,
                "Add the Python binding.",
            )
        )
    else:
        defs = set(re.findall(r'm\.def\(\s*"([a-z0-9_]+)"', pytext))
        has_alloc = any(not d.endswith("_into") for d in defs)
        has_into = any(d.endswith("_into") for d in defs)
        ok = has_alloc and has_into
        out.append(
            Finding(
                "SUP-4",
                "support",
                PASS if ok else GAP,
                "Python binds allocating + _into variants",
                f"{rel(P.pybind)}: defs={sorted(defs)}",
                g,
                "" if ok else "Bind the missing allocating/_into variant.",
            )
        )

    # SUP-5..8 Limitations table parse
    lim = parse_limitations(htext)
    if lim is None:
        out.append(
            Finding(
                "SUP-5",
                "support",
                GAP if htext else GAP,
                "Limitations tables not present/parseable",
                f"{rel(P.header)}: no parseable Data Layout/Channels/Data Type table",
                g,
                "Add the Doxygen Limitations table (layout/channels/dtype, in+out).",
            )
        )
        out.append(
            Finding(
                "SUP-6/7/8",
                "support",
                MANUAL,
                "Declared layout/dtype/channel matrix not auto-parsed",
                f"Inspect the Limitations table in {rel(P.header)}",
                g,
            )
        )
        lim = {"layouts": set(), "channels": set(), "dtypes": set()}
    else:
        out.append(
            Finding(
                "SUP-5",
                "support",
                PASS,
                "Limitations tables present",
                f"lay={sorted(lim['layouts'])} dt={sorted(lim['dtypes'])} ch={sorted(lim['channels'])}",
                g,
            )
        )
        out.append(
            Finding(
                "SUP-6/7/8",
                "support",
                PASS,
                "Declared matrix parsed",
                f"lay={sorted(lim['layouts'])} dt={sorted(lim['dtypes'])} ch={sorted(lim['channels'])}",
                g,
            )
        )

    # SUP-9 enforcement (presence of input validation in priv)
    enforce_hits = []
    for pv in P.priv:
        enforce_hits += [
            (pv, h)
            for h in grep(
                r"ERROR_INVALID_ARGUMENT|Invalid (DataFormat|channel|data type|format)",
                read(pv),
            )
        ]
    if enforce_hits:
        pv, (ln, line) = enforce_hits[0]
        out.append(
            Finding(
                "SUP-9",
                "support",
                MANUAL,
                "Enforcement present; declared-vs-enforced match needs human read",
                f"{rel(pv)}:{ln}: {line.strip()[:100]}",
                g,
                "Confirm the runtime guards match the declared Limitations matrix.",
            )
        )
    else:
        out.append(
            Finding(
                "SUP-9",
                "support",
                MANUAL,
                "No obvious input-validation guard found in priv; verify manually",
                f"priv files: {[rel(p) for p in P.priv] or 'none located'}",
                g,
                "Ensure unsupported layout/dtype/channel are rejected with ERROR_INVALID_ARGUMENT.",
            )
        )

    # Planar layouts are part of the default support contract. Only operators whose tensors do
    # not represent images may opt out, and that decision lives beside the operator Limitations.
    planar_declared = bool(lim["layouts"] & {"NCHW", "CHW"})
    planar_policy = parse_planar_policy(htext)
    out.append(planar_policy_verdict(P, htext, planar_declared))

    # SUP-11 cross-surface consistency (container sets)
    note = []
    if hpptext is not None and pytext is not None:
        note.append("containers checked across header/.hpp/python (see SUP-3/4)")
        out.append(
            Finding(
                "SUP-11",
                "support",
                PASS,
                "Cross-surface container consistency (see SUP-3/SUP-4)",
                "; ".join(note),
                g,
            )
        )
    else:
        out.append(
            Finding(
                "SUP-11",
                "support",
                MANUAL,
                "Cross-surface consistency: a surface file is missing",
                f".hpp={'ok' if hpptext else 'missing'} python={'ok' if pytext else 'missing'}",
                g,
            )
        )
    return out, {
        "planar": planar_declared and not planar_policy["not_applicable"],
        "planar_declared": planar_declared,
        "planar_not_applicable": planar_policy["not_applicable"],
        "planar_reason": planar_policy["reason"],
        "limitations": lim,
        "has_tensor": has_tensor,
        "has_vs": has_vs,
        "generic_varshape": generic_varshape,
        "bench_layout_na": P.op in bench_layout_na,
        "bench_rgb_na": P.op in bench_rgb_na,
    }


# ============================================================================== TEST domain
def _tensor_positive_test_pattern(Op: str) -> str:
    """Return the candidate Tensor-test macro pattern for one operator.

    Verdicts use :func:`_tensor_positive_test_hits`, which parses both the suite
    and test name. This pattern remains the narrow candidate extractor for
    callers that only need to locate relevant macros.
    """
    test_macro = r"(?:TEST|TEST_P|TYPED_TEST)"
    suite_prefix = rf"Op{re.escape(Op)}"
    return (
        rf"{test_macro}\(\s*{suite_prefix}(?:\s*,|"
        rf"Tensor(?![A-Za-z0-9_]*(?:[Nn]egative|[Ii]nvalid|[Rr]eject|[Ss]moke))"
        rf"[A-Za-z0-9_]*\s*,)"
    )


_NONPOSITIVE_TENSOR_TEST_TOKENS = (
    "negative",
    "incorrect",
    "invalid",
    "reject",
    "smoke",
    "unsupported",
    "mismatch",
    "null",
    "error",
    "failure",
    "overflow",
    "exceed",
    "out_of_range",
    "outofrange",
)


def _tensor_positive_test_hits(Op: str, text: str | None):
    """Return parsed positive Tensor test macros with source-line evidence.

    The generic ``Op<Op>`` suite and Tensor-specialized suites are candidates,
    but their test names still have to describe a non-negative execution case.
    This prevents a smoke or rejection-only suite from satisfying TST-2 while
    preserving established positive names such as ``CustomCrop_packed``.
    """
    if not text:
        return []

    def blank_comment(match):
        return re.sub(r"[^\n]", " ", match.group(0))

    parsed_text = re.sub(r"//[^\n]*|/\*.*?\*/", blank_comment, text, flags=re.S)
    macro = re.compile(
        r"\b(?:TEST|TEST_P|TYPED_TEST)\s*\(\s*"
        r"(?P<suite>[A-Za-z_]\w*)\s*,\s*(?P<name>[A-Za-z_]\w*)\s*\)"
    )
    prefix = f"Op{Op}"
    lines = text.splitlines()
    hits = []
    for match in macro.finditer(parsed_text):
        suite = match.group("suite")
        name = match.group("name")
        if not suite.startswith(prefix):
            continue
        suffix = suite.removeprefix(prefix)
        if suffix and not suffix.lower().startswith("tensor"):
            continue

        suite_and_name = f"{suite}_{name}".lower()
        if any(token in suite_and_name for token in _NONPOSITIVE_TENSOR_TEST_TOKENS):
            continue
        if "varshape" in suite_and_name or "imagebatch" in suite_and_name:
            continue
        if name.lower() in {"op", "operator_creation", "creation"}:
            continue

        line_number = text.count("\n", 0, match.start()) + 1
        hits.append((line_number, lines[line_number - 1]))
    return hits


def _varshape_positive_test_pattern(Op: str, generic_varshape: bool) -> str:
    """Return the positive VarShape-test heuristic for one operator.

    A generic suite is accepted only when its test name is correctness-bearing. This keeps
    unrelated smoke or negative tests from satisfying TST-3 for legacy VarShape-only APIs.
    """
    correctness_token = r"(?<![A-Za-z])[Cc]orrect(?![A-Za-z])"
    generic_suite = (
        rf"|(?:TEST|TEST_P|TYPED_TEST)\(\s*Op{re.escape(Op)}\s*,\s*[^)]*{correctness_token}"
        if generic_varshape
        else ""
    )
    return (
        rf"varshape_correct_output|TEST(_P)?\(\s*Op{re.escape(Op)}Var[Ss]hape\s*,"
        rf"|[Vv]ar[Ss]hape.*{correctness_token}{generic_suite}"
    )


def python_image_layout_na(support_info: dict) -> bool:
    """Return whether image-layout Python coverage is semantically inapplicable."""
    return bool(
        support_info.get("bench_layout_na") and support_info.get("bench_rgb_na")
    )


def check_test(P: OpPaths, support_info):
    out = []
    t = read(P.test_cpp)
    g = (
        ".agents/guidance/REVIEW_OP_GUIDELINES.md#test / make_operator.rst / "
        ".agents/guidance/OPTIMIZATION_GUIDELINES.md"
    )
    planar = support_info.get("planar")
    planar_not_applicable = support_info.get("planar_not_applicable", False)
    has_tensor = support_info.get("has_tensor")
    has_vs = support_info.get("has_vs")
    generic_varshape = support_info.get("generic_varshape", False)

    if t is None:
        out.append(
            Finding(
                "TST-1",
                "test",
                GAP,
                "C++ system test file not found",
                f"{rel(P.test_cpp)} (missing)",
                g,
                "Add the C++ system test.",
            )
        )
        t = ""

    # TST-1 reference fn (heuristic: a Gold/Reference/Ref helper or a goldX)
    ref_hits = grep(r"Gold|gold|[Rr]eference|setGold|MatchesGold", t)
    out.append(
        Finding(
            "TST-1",
            "test",
            PASS if ref_hits else MANUAL,
            "Independent CPU reference present"
            if ref_hits
            else "No obvious CPU reference; verify",
            first_evidence(P.test_cpp, ref_hits) or "no Gold/Reference symbol found",
            g,
            "" if ref_hits else "Add/confirm an independent CPU reference.",
        )
    )

    # TST-2 / TST-3 — a positive test on the op's (non-Negative) suite, any naming convention
    if generic_varshape and not has_tensor:
        out.append(
            Finding("TST-2", "test", NA, "VarShape-only op -> tensor test N-A", "", g)
        )
    else:
        tensor_hits = _tensor_positive_test_hits(P.Op, t)
        out.append(
            Finding(
                "TST-2",
                "test",
                PASS if tensor_hits else GAP,
                (
                    "tensor positive/correctness test present"
                    if tensor_hits
                    else "tensor positive/correctness test MISSING"
                ),
                first_evidence(P.test_cpp, tensor_hits)
                or f"{rel(P.test_cpp)}: no positive Tensor TEST macro found",
                g,
                "" if tensor_hits else "Add tensor positive/correctness test.",
            )
        )
    if has_vs:
        out.append(
            _present(
                P.test_cpp,
                t,
                "TST-3",
                "test",
                _varshape_positive_test_pattern(P.Op, generic_varshape),
                "varshape positive/correctness test",
                g,
            )
        )
    else:
        out.append(
            Finding("TST-3", "test", NA, "Tensor-only op -> varshape test N-A", "", g)
        )

    # TST-4 parametrized or typed suite
    out.append(
        _present(
            P.test_cpp,
            t,
            "TST-4",
            "test",
            r"NVCV_(?:TEST_SUITE_P|TYPED_TEST_SUITE)",
            "parametrized or typed test suite",
            g,
        )
    )

    # TST-5 matrix-mirror (axis coverage) — best-effort; MANUAL residual for modes
    out.append(matrix_mirror(P, t, support_info, g))

    # TST-6 negative suite
    out.append(
        _present(
            P.test_cpp,
            t,
            "TST-6",
            "test",
            rf"Op{P.Op}.*_Negative|_Negative",
            "negative test suite",
            g,
            fix="Add negative tests asserting NVCV_ERROR_INVALID_ARGUMENT.",
        )
    )

    # TST-7 equivalent image-layout parity
    if planar:
        fp = grep(r"PlanarParityUtils|matches_interleaved", t)
        out.append(
            Finding(
                "TST-7",
                "test",
                PASS if fp else GAP,
                "Equivalent image-layout parity test present"
                if fp
                else "Equivalent image-layout parity test MISSING",
                first_evidence(P.test_cpp, fp)
                or "no PlanarParityUtils/matches_interleaved",
                REVIEW_OP_GUIDE,
                "" if fp else "Add Op%sPlanar.*_matches_interleaved." % P.Op,
            )
        )
    elif planar_not_applicable:
        out.append(
            Finding(
                "TST-7",
                "test",
                NA,
                "Image layouts are not applicable -> parity N-A",
                support_info.get("planar_reason", ""),
                g,
            )
        )
    else:
        out.append(
            Finding(
                "TST-7",
                "test",
                GAP,
                "Layout policy is incomplete; parity cannot be verified",
                "SUP-10 requires NCHW/CHW or an operator-local inapplicability declaration",
                g,
                "Complete the operator's image-layout support contract.",
            )
        )

    # TST-8 tolerance discipline
    out.append(tolerance_discipline(P, t, g))

    # TST-9 reference independence + edge adequacy
    out.append(
        Finding(
            "TST-9",
            "test",
            MANUAL,
            "Reference independence + edge-case adequacy need human read",
            f"Inspect {rel(P.test_cpp)}",
            g,
        )
    )

    # TST-10 deterministic inputs
    rng = grep(
        r"\brand\(\)|std::random_device|std::mt19937(?!.*\()|setSeed|RandomValues|\bsrand\b",
        t,
    )
    seeded = grep(r"mt19937[^;]*\(\s*\d|seed|Seed|deterministic|FillDeterministic", t)
    if grep(r"random_device", t):
        out.append(
            Finding(
                "TST-10",
                "test",
                MANUAL,
                "Possible unseeded RNG (random_device) — verify determinism",
                first_evidence(P.test_cpp, grep(r"random_device", t)),
                g,
                "Use a fixed seed / deterministic fill.",
            )
        )
    else:
        out.append(
            Finding(
                "TST-10",
                "test",
                PASS if (seeded or not rng) else MANUAL,
                "Inputs appear deterministic"
                if (seeded or not rng)
                else "RNG without obvious seed — verify",
                first_evidence(P.test_cpp, seeded or rng) or "no RNG detected",
                g,
            )
        )

    # Python TST-11..14
    py = read(P.test_py)
    if py is None:
        for tid, desc in [
            ("TST-11", "Python Tensor NHWC/HWC"),
            ("TST-12", "Python VarShape"),
            ("TST-13", "Python allocating + _into"),
            ("TST-14", "Python negative"),
        ]:
            if tid == "TST-11" and python_image_layout_na(support_info):
                out.append(
                    Finding(
                        tid,
                        "test",
                        NA,
                        "Coordinate-list Tensor input -> Python image layouts N-A",
                        "",
                        g,
                    )
                )
            elif tid == "TST-12" and not has_vs:
                out.append(
                    Finding(
                        tid, "test", NA, "Tensor-only -> Python varshape N-A", "", g
                    )
                )
            else:
                out.append(
                    Finding(
                        tid,
                        "test",
                        GAP,
                        f"{desc}: python test file missing",
                        f"{rel(P.test_py)} (missing)",
                        g,
                        "Add the Python API test.",
                    )
                )
    elif grep(r"make_op_tests", py):
        # The repo's standard generator covers layouts / _into / negative API-surface tests.
        sl = re.search(r"supported_layouts\s*=\s*\{([^}]*)\}", py)
        ev = (
            f"{rel(P.test_py)}: make_op_tests("
            + (f"supported_layouts={{{sl.group(1).strip()}}}" if sl else "...")
            + ")"
        )
        out.append(
            Finding(
                "TST-11", "test", PASS, "Python Tensor layouts via make_op_tests", ev, g
            )
        )
        out.append(
            Finding(
                "TST-12",
                "test",
                PASS if has_vs else NA,
                "Python VarShape via make_op_tests"
                if has_vs
                else "Tensor-only -> Python varshape N-A",
                ev if has_vs else "",
                g,
            )
        )
        out.append(
            Finding(
                "TST-13",
                "test",
                PASS,
                "Python allocating + _into via make_op_tests",
                ev,
                g,
            )
        )
        out.append(
            Finding("TST-14", "test", PASS, "Python negative via make_op_tests", ev, g)
        )
    else:
        if python_image_layout_na(support_info):
            out.append(
                Finding(
                    "TST-11",
                    "test",
                    NA,
                    "Coordinate-list Tensor input -> Python image layouts N-A",
                    "",
                    g,
                )
            )
        else:
            out.append(
                _present(
                    P.test_py,
                    py,
                    "TST-11",
                    "test",
                    r"NHWC|HWC|supported_layouts",
                    "Python Tensor layouts exercised",
                    g,
                )
            )
        if has_vs:
            out.append(
                _present(
                    P.test_py,
                    py,
                    "TST-12",
                    "test",
                    r"VarShape|ImageBatch|var_shape|varshape",
                    "Python VarShape exercised",
                    g,
                )
            )
        else:
            out.append(
                Finding(
                    "TST-12", "test", NA, "Tensor-only -> Python varshape N-A", "", g
                )
            )
        out.append(
            _present(
                P.test_py,
                py,
                "TST-13",
                "test",
                rf"{P.pyname}_into|_into",
                "Python allocating + _into",
                g,
            )
        )
        out.append(
            _present(
                P.test_py,
                py,
                "TST-14",
                "test",
                r"pytest\.raises|raises\(",
                "Python negative cases",
                g,
            )
        )
    return out


def _present(path, text, tid, domain, pattern, desc, guideline, fix=None):
    hits = grep(pattern, text)
    if hits:
        return Finding(
            tid, domain, PASS, desc + " present", first_evidence(path, hits), guideline
        )
    return Finding(
        tid,
        domain,
        GAP,
        desc + " MISSING",
        f"{rel(path)}: pattern /{pattern}/ not found",
        guideline,
        fix or f"Add {desc}.",
    )


def matrix_mirror(P, t, support_info, g):
    """Best-effort axis-coverage: which declared dtypes/channels/layouts appear in the test."""
    lim = support_info.get("limitations") or {}
    if not t or not lim.get("dtypes"):
        return Finding(
            "TST-5",
            "test",
            MANUAL,
            "Matrix-mirror not auto-evaluated (no parsed matrix or empty test)",
            f"Inspect test cases in {rel(P.test_cpp)}",
            g,
        )
    # crude format-token presence: FMT_RGB8 (u8/3ch), FMT_RGBA8 (u8/4ch), FMT_*f32 (f32) ...
    fmt_tokens = set(re.findall(r"FMT_[A-Za-z0-9]+", t))
    covered_layouts = set()
    if grep(r"FMT_[A-Za-z0-9]+p\b|NCHW|CHW", t):
        covered_layouts.add("NCHW")
    if fmt_tokens or grep(r"NHWC|HWC", t):
        covered_layouts.add("NHWC")
    missing_layout = sorted(
        (lim.get("layouts") or set()) - covered_layouts - {"HWC", "CHW"}
    )
    if missing_layout:
        return Finding(
            "TST-5",
            "test",
            MANUAL,
            "Axis-coverage partial; verify modes/dtypes; layouts not clearly covered: "
            + ",".join(missing_layout),
            f"format tokens in test: {sorted(fmt_tokens)[:8]}",
            g,
            "Add positive cases for the uncovered axis values.",
        )
    return Finding(
        "TST-5",
        "test",
        MANUAL,
        "Axis-coverage (layout/dtype) looks present; MODE coverage needs human confirmation",
        f"format tokens: {sorted(fmt_tokens)[:8]}; layouts covered: {sorted(covered_layouts)}",
        g,
    )


def tolerance_discipline(P, t, g):
    near = grep(r"EXPECT_NEAR|ASSERT_NEAR", t)
    if not near:
        return Finding(
            "TST-8",
            "test",
            PASS,
            "Bit-exact comparisons (no EXPECT_NEAR)",
            "no EXPECT_NEAR/ASSERT_NEAR found",
            ".agents/guidance/OPTIMIZATION_GUIDELINES.md",
        )
    # any EXPECT_NEAR without an adjacent rationale comment -> MANUAL/needs-human
    lines = t.splitlines()
    justified = 0
    for ln, _ in near:
        start = max(0, ln - 3)
        ctx = " ".join(lines[start:ln])
        if re.search(
            r"//.*(toler|FMA|contract|precision|rationale|because)", ctx, re.I
        ):
            justified += 1
    n = len(near)
    return Finding(
        "TST-8",
        "test",
        MANUAL,
        f"{n} EXPECT_NEAR site(s), {justified} with rationale; bit-exact is the default — review",
        first_evidence(P.test_cpp, near),
        ".agents/guidance/OPTIMIZATION_GUIDELINES.md (do not silently bump tolerances)",
        "Justify each EXPECT_NEAR with a rationale comment or switch to EXPECT_EQ; never loosen silently.",
    )


# ============================================================================= BENCH domain
def check_bench(P: OpPaths, support_info, do_run):
    out = []
    g = ".agents/guidance/REVIEW_OP_GUIDELINES.md#bench / bench/README.md"
    planar = support_info.get("planar")
    planar_not_applicable = support_info.get("planar_not_applicable", False)

    # BEN-1/2 drivers + registration
    cml_cpp = read(REPO / "bench/cpp/CMakeLists.txt") or ""
    cml_py = read(REPO / "bench/python/CMakeLists.txt") or ""
    cpp_ok = P.bench_cpp.exists()
    out.append(
        Finding(
            "BEN-1",
            "bench",
            PASS if cpp_ok else GAP,
            "C++ bench present"
            + (
                ""
                if f"Bench{P.Op}" in cml_cpp or P.op in cml_cpp.lower()
                else " (registration unverified)"
            ),
            rel(P.bench_cpp) + ("" if cpp_ok else " (missing)"),
            g,
            ""
            if cpp_ok
            else "Add the C++ benchmark + register in bench/cpp/CMakeLists.txt.",
        )
    )
    py_ok = P.bench_py.exists()
    py_reg = (
        ""
        if (f"bench_{P.op}" in cml_py or P.op in cml_py.lower())
        else " (registration unverified)"
    )
    out.append(
        Finding(
            "BEN-2",
            "bench",
            PASS if py_ok else GAP,
            "Python bench present" + py_reg,
            rel(P.bench_py) + ("" if py_ok else " (missing)"),
            g,
            ""
            if py_ok
            else "Add the Python benchmark + register in bench/python/CMakeLists.txt.",
        )
    )

    # BEN-3 manifest
    manifest = load_bench_cfg(REPO / "bench/config/bench_params.json") or {}
    in_manifest = P.op in (manifest.get("operators") or {})
    out.append(
        Finding(
            "BEN-3",
            "bench",
            PASS if in_manifest else GAP,
            "Manifest entry in bench_params.json",
            f"operators.{P.op} {'present' if in_manifest else 'MISSING'}",
            g,
            ""
            if in_manifest
            else "Add the operator entry to bench/config/bench_params.json.",
        )
    )

    # BEN-4 config + tiers
    cfg = load_bench_cfg(P.bench_cfg)
    if not cfg or "configs" not in cfg:
        out.append(
            Finding(
                "BEN-4",
                "bench",
                GAP,
                "Operator bench config missing/invalid",
                rel(P.bench_cfg) + " (missing or no 'configs')",
                g,
                "Add bench/config/operators/<op>.json.",
            )
        )
        # without a config, the rest of bench can't be evaluated
        for tid in ("BEN-5", "BEN-6", "BEN-7", "BEN-8", "BEN-14"):
            out.append(
                Finding(
                    tid,
                    "bench",
                    GAP,
                    "Not evaluable — bench config missing",
                    rel(P.bench_cfg),
                    g,
                    "Add the bench config first.",
                )
            )
        return out
    configs = cfg["configs"]
    bad_tier = [
        k for k, c in configs.items() if c.get("tier") not in ("basic", "advanced")
    ]
    out.append(
        Finding(
            "BEN-4",
            "bench",
            PASS if not bad_tier else GAP,
            "Config present; all entries have a valid tier"
            if not bad_tier
            else "Configs with bad/missing tier: " + ",".join(bad_tier[:5]),
            f"{len(configs)} configs",
            g,
            "" if not bad_tier else "Set tier=basic|advanced on every config.",
        )
    )

    # BEN-5 layout axis on every config, except where the reviewed benchmark
    # semantics have no image-layout dimension (point sets, reductions, or an
    # operator whose purpose is to define/transform layout).
    no_layout = [
        k for k, c in configs.items() if "layout" not in c.get("string_axes", {})
    ]
    layout_na = support_info.get("bench_layout_na")
    if layout_na:
        unexpected_layout = [k for k in configs if k not in no_layout]
        out.append(
            Finding(
                "BEN-5",
                "bench",
                NA if not unexpected_layout else GAP,
                "layout axis not applicable to this benchmark's semantics"
                if not unexpected_layout
                else f"layout N-A but {len(unexpected_layout)} config(s) carry a dummy layout axis",
                "curated layout-axis N-A classification"
                if not unexpected_layout
                else "unexpected: " + ", ".join(unexpected_layout[:6]),
                g,
                ""
                if not unexpected_layout
                else "Remove the misleading layout axis from these configs.",
            )
        )
    else:
        out.append(
            Finding(
                "BEN-5",
                "bench",
                PASS if not no_layout else GAP,
                "layout axis on every config"
                if not no_layout
                else f"{len(no_layout)} config(s) missing the layout axis",
                "all configs carry string_axes.layout"
                if not no_layout
                else "missing: " + ", ".join(no_layout[:6]),
                g,
                ""
                if not no_layout
                else 'Add a truthful "layout" axis to every config.',
            )
        )

    # collect benched axis values per tier
    benched = collect_benched(configs)
    layouts = benched["all"]["layout"]
    has_fake = bool({"NCHW_FAKE", "CHW_FAKE"} & layouts)
    pair_issues = ()
    if has_fake:
        try:
            pair_issues = fake_planar_pairing_issues(configs)
        except BaselineError as exc:
            pair_issues = (f"cannot expand FakePlanar cases: {exc}",)

    # BEN-6 native and layout-conversion reference configs. A benchmark whose
    # semantics have no truthful layout-comparison axis must not grow dummy
    # NCHW/NCHW_FAKE rows even if the underlying operator accepts those layouts.
    # A declared FakePlanar row always owns its pairing contract, even when
    # support/curated inference otherwise classifies that comparison as N-A.
    if pair_issues:
        out.append(
            Finding(
                "BEN-6",
                "bench",
                GAP,
                f"{len(pair_issues)} unmatched/invalid FakePlanar case signature(s); "
                "each requires exactly one same-tier native NCHW/CHW Tensor case",
                " | ".join(pair_issues),
                REVIEW_OP_GUIDE,
                "Add one advanced Tensor native NCHW/CHW case with every axis except "
                "layout identical to each FakePlanar case.",
            )
        )
    elif support_info.get("bench_layout_na"):
        out.append(
            Finding(
                "BEN-6",
                "bench",
                NA,
                "layout comparison axis not applicable to this benchmark's semantics",
                "deterministic benchmark layout-axis N-A classification",
                g,
            )
        )
    elif planar:
        has_tensor = support_info.get("has_tensor")
        fake_planar_applicable = bool(
            has_tensor and not support_info.get("bench_rgb_na")
        )
        has_native = bool({"NCHW", "CHW"} & layouts)
        ok = has_native and (has_fake or not fake_planar_applicable)
        summary = (
            "Image-layout bench configs (NCHW + NCHW_FAKE)"
            if ok and fake_planar_applicable
            else (
                "Image-layout bench configs "
                "(NCHW; NCHW_FAKE N-A for scalar/RGB-N-A tensor benchmark)"
                if ok and has_tensor
                else (
                    "Image-layout bench configs "
                    "(NCHW; NCHW_FAKE N-A for var-shape-only op)"
                    if ok
                    else "Image-layout bench configs incomplete"
                )
            )
        )
        evidence = f"native NCHW/CHW={has_native} FakePlanar={has_fake}"
        out.append(
            Finding(
                "BEN-6",
                "bench",
                PASS if ok else GAP,
                summary,
                evidence,
                REVIEW_OP_GUIDE,
                ""
                if ok
                else (
                    "Add native NCHW and NCHW_FAKE configs (uchar4 tensor-only)."
                    if fake_planar_applicable
                    else (
                        "Add native NCHW configs; NCHW_FAKE is N-A for this "
                        "scalar/RGB-N-A tensor benchmark."
                        if has_tensor
                        else "Add native NCHW configs for the var-shape-only planar path."
                    )
                ),
            )
        )
    elif planar_not_applicable:
        out.append(
            Finding(
                "BEN-6",
                "bench",
                NA,
                "Image layouts are not applicable -> layout bench configs N-A",
                support_info.get("planar_reason", ""),
                g,
            )
        )
    else:
        out.append(
            Finding(
                "BEN-6",
                "bench",
                GAP,
                "Layout policy is incomplete; required bench configs cannot be determined",
                "SUP-10 requires NCHW/CHW or an operator-local inapplicability declaration",
                g,
                "Complete the operator's image-layout support contract.",
            )
        )

    # BEN-SIZE apples-to-apples: layouts of one dtype share an input size
    out.append(bench_size_uniformity(configs, g))

    # BEN-7 baseline completeness
    out.append(baseline_completeness(P, configs, g))

    # BEN-8 row-count consistency
    out.append(rowcount_consistency(P, cfg, g))

    # BEN-9 validate_baselines
    out.append(
        Finding(
            "BEN-9",
            "bench",
            MANUAL,
            "Run the internal baseline validator to confirm case-key/SKU validity",
            f"python3 bench/_internal/validate_baselines.py --operator {P.op}",
            g,
            "Fix any case-key/SKU validation errors it reports.",
        )
    )

    # BEN-10 parity / BEN-11 run-dependent
    out.append(
        Finding(
            "BEN-10",
            "bench",
            PASS,
            "C++/Python config parity is structural (shared config); full parity needs a run",
            "shared bench/config/operators/%s.json drives both languages" % P.op,
            g,
        )
    )
    out.append(
        Finding(
            "BEN-11",
            "bench",
            MANUAL,
            "Noise/parity quality and baseline currency require a GPU run",
            f"run: python3 run_bench.py --operator {P.op} (use --run to attempt)",
            g,
        )
    )

    # BEN-14 basic-tier floor (HARD)
    out.append(basic_floor(P, benched, support_info, g))

    # BEN-12/13 coverage stats + BEN-15/16 recommendations (advisory)
    out += coverage_analysis(P, benched, support_info, g)
    return out


def bench_size_uniformity(configs, g):
    """BEN-SIZE (advisory): the interleaved (NHWC), native-planar (NCHW) and fake-planar
    (NCHW_FAKE) configs of the *same dtype* should share the SAME input `shape`, so the three
    layouts are an apples-to-apples comparison. Calibrate the interleaved config to 1-2 ms; the
    planar / fake-planar configs reuse that shape and may legitimately run longer. We group configs
    by their `dtypes` tuple and flag any group whose configs span more than one distinct shape.
    """
    target_layouts = {"NHWC", "NCHW", "NCHW_FAKE"}
    groups = {}
    for k, c in configs.items():
        dtypes = c.get("dtypes", [])
        layouts = [
            lay
            for lay in c.get("string_axes", {}).get("layout", [])
            if lay in target_layouts
        ]
        if not dtypes or not layouts:
            continue
        shape = tuple(c.get("string_axes", {}).get("shape", []))
        # Group per *individual* dtype (not the dtype-tuple) so a config's dtype is compared
        # against every other config carrying it, regardless of how dtypes are bundled.
        for dt in dtypes:
            groups.setdefault(dt, []).append((k, tuple(layouts), shape))
    offenders = []
    for dt, members in groups.items():
        # Only a genuine cross-layout comparison: the dtype must span >1 of NHWC/NCHW/NCHW_FAKE.
        layouts_seen = {lay for _, lays, _ in members for lay in lays}
        if len(layouts_seen) < 2:
            continue
        distinct = {shape for _, _, shape in members}
        if len(distinct) > 1:
            detail = "; ".join(
                f"{name}[{','.join(lays)}]={'x'.join(shape) or '?'}"
                for name, lays, shape in members
            )
            offenders.append(f"dtype {dt}: {detail}")
    if not offenders:
        return Finding(
            "BEN-SIZE",
            "bench",
            PASS,
            "Each dtype's layouts (NHWC/NCHW/NCHW_FAKE) share one input size (apples-to-apples)",
            f"{len(groups)} dtype group(s) size-consistent",
            ".agents/guidance/MAKE_OP_GUIDELINES.md",
        )
    return Finding(
        "BEN-SIZE",
        "bench",
        REC,
        "Layouts of a dtype use different input sizes -> not an apples-to-apples comparison",
        " | ".join(offenders),
        ".agents/guidance/MAKE_OP_GUIDELINES.md",
        "Give each dtype's NHWC/NCHW/NCHW_FAKE configs the SAME shape (calibrate the interleaved "
        "config to 1-2 ms; planar/fake-planar reuse that shape and may run longer).",
    )


def collect_benched(configs):
    """Return benched axis values overall and per tier."""

    def empty():
        return {"layout": set(), "inputKind": set(), "dtypes": set(), "channels": set()}

    res = {"all": empty(), "basic": empty(), "advanced": empty()}
    for c in configs.values():
        tier = c.get("tier")
        sa = c.get("string_axes", {})
        ia = c.get("int64_axes", {})
        dts = c.get("dtypes", [])
        targets = [res["all"]]
        if tier in ("basic", "advanced"):
            targets.append(res[tier])
        for tgt in targets:
            tgt["layout"].update(sa.get("layout", []))
            tgt["inputKind"].update(sa.get("inputKind", []))
            tgt["dtypes"].update(dts)
            # Most configs encode channels in vector dtype names (uchar3,
            # float4). HQResize and similar scalar-element benchmarks expose a
            # separate numChannels axis; that is the authoritative image
            # channel count when present.
            if "numChannels" in ia:
                tgt["channels"].update(int(v) for v in ia["numChannels"])
            else:
                for d in dts:
                    m = re.search(r"(\d)$", d)
                    tgt["channels"].add(int(m.group(1)) if m else 1)
    return res


def basic_floor(P, benched, support_info, g):
    b = benched["basic"]
    has_tensor = support_info.get("has_tensor")
    has_vs = support_info.get("has_vs")
    planar = support_info.get("planar")
    supported_channels = (support_info.get("limitations") or {}).get(
        "channels"
    ) or set()
    missing = []
    # RGB = a 3-channel dtype present in basic
    if (
        not support_info.get("bench_rgb_na")
        and (not supported_channels or 3 in supported_channels)
        and 3 not in b["channels"]
    ):
        missing.append("RGB(3-channel dtype)")
    if has_tensor and "Tensor" not in b["inputKind"]:
        missing.append("Tensor")
    if has_vs and "VarShape" not in b["inputKind"]:
        missing.append("VarShape")
    if not support_info.get("bench_layout_na"):
        if "NHWC" not in b["layout"]:
            missing.append("interleaved NHWC")
        if planar and "NCHW" not in b["layout"]:
            missing.append("planar NCHW")
    ok = not missing
    return Finding(
        "BEN-14",
        "bench",
        PASS if ok else GAP,
        "Basic-tier minimum floor satisfied"
        if ok
        else "Basic-tier floor MISSING: " + ", ".join(missing),
        f"basic: dt={sorted(b['dtypes'])} ik={sorted(b['inputKind'])} lay={sorted(b['layout'])}",
        g,
        ""
        if ok
        else "Add basic configs covering: "
        + ", ".join(missing)
        + " (RGB×Tensor×VarShape-if-applicable×NHWC×NCHW-if-applicable).",
    )


def baseline_completeness(P, configs, g):
    sku = load_bench_cfg(REPO / "bench/config/sku_map.json") or {}
    skus = sorted(e["stem"] for e in sku.get("entries", []) if "stem" in e)
    if not skus:
        return Finding(
            "BEN-7", "bench", MANUAL, "No SKUs in sku_map.json to check against", "", g
        )
    missing = []
    invalid = []
    total_cases = 0
    for k, c in configs.items():
        bl = c.get("baselines", {})
        if not isinstance(bl, dict):
            invalid.append(f"{k}: baselines must be an object keyed by case-key")
            continue
        try:
            expected_case_keys = expected_case_keys_for_entry(k, c)
        except BaselineError as exc:
            invalid.append(f"{k}: {exc}")
            continue
        total_cases += len(expected_case_keys)
        for case_key in expected_case_keys:
            per_sku = bl.get(case_key)
            if per_sku is None:
                missing.append((case_key, "*"))
                continue
            if not isinstance(per_sku, dict):
                invalid.append(
                    f"{case_key}: baseline case payload must be an object keyed by SKU"
                )
                continue
            for s in skus:
                if s not in per_sku:
                    missing.append((case_key, s))
    if not missing and not invalid:
        return Finding(
            "BEN-7",
            "bench",
            PASS,
            f"Baselines complete for all {total_cases} case-keys × {len(skus)} SKUs",
            f"SKUs={skus}",
            g,
        )
    if invalid:
        sample = "; ".join(invalid[:4])
        return Finding(
            "BEN-7",
            "bench",
            GAP,
            f"Baseline completeness could not expand {len(invalid)} config(s)",
            f"e.g. {sample}",
            g,
            "Fix malformed benchmark axes before regenerating baselines.",
        )
    sample = "; ".join(f"{ck}→{s}" for ck, s in missing[:4])
    return Finding(
        "BEN-7",
        "bench",
        GAP,
        f"Baselines incomplete: {len(missing)} missing case/SKU entries",
        f"e.g. {sample}",
        g,
        "If a declared case is unsupported, remove it from the config axes; otherwise "
        "regenerate via MR baseline-regen CI for operator '%s' + "
        "bench/_internal/update_baseline.py — never fabricate." % P.op,
    )


def rowcount_consistency(P, cfg, g):
    configs = cfg["configs"]
    basic_rows = sum(
        rows_for_entry(c) for c in configs.values() if c.get("tier") == "basic"
    )
    adv_rows = sum(
        rows_for_entry(c) for c in configs.values() if c.get("tier") == "advanced"
    )
    test = read(REPO / "bench/tests/test_run_bench_config_key.py") or ""
    # look for an explicit per-op expected count if present
    m = re.search(rf'["\']{P.op}["\']\s*:\s*(\d+)', test)
    if m:
        exp = int(m.group(1))
        ok = exp in (basic_rows, basic_rows + adv_rows)
        return Finding(
            "BEN-8",
            "bench",
            PASS if ok else GAP,
            "Row-count matches test_run_bench_config_key.py"
            if ok
            else f"Row-count mismatch: computed basic={basic_rows} (adv={adv_rows}) vs test {exp}",
            f"computed basic_rows={basic_rows} advanced_rows={adv_rows}",
            g,
            ""
            if ok
            else "Recompute and update the per-op/global counts in test_run_bench_config_key.py.",
        )
    return Finding(
        "BEN-8",
        "bench",
        MANUAL,
        f"No per-op count in test; computed basic={basic_rows} adv={adv_rows}; verify global asserts",
        "bench/tests/test_run_bench_config_key.py",
        g,
        "Ensure the global basic/advanced key+row asserts include this operator's rows.",
    )


def coverage_analysis(P, benched, support_info, g):
    out = []
    lim = support_info.get("limitations") or {}
    input_kinds = set()
    if support_info.get("has_tensor"):
        input_kinds.add("Tensor")
    if support_info.get("has_vs"):
        input_kinds.add("VarShape")
    if "TensorBatch" in benched["all"]["inputKind"]:
        input_kinds.add("TensorBatch")

    # BEN-13 statistics (per-axis, per-tier)
    def axis_stat(axis, supported, key):
        b, a = benched["basic"][key], benched["advanced"][key]
        lines = []
        for v in (
            sorted(map(str, supported))
            if supported
            else sorted(map(str, benched["all"][key]))
        ):
            where = (
                "basic"
                if v in map(str, b)
                else ("advanced" if v in map(str, a) else "none")
            )
            lines.append(f"{v}:{where}")
        return f"{axis}: " + ", ".join(lines) if lines else f"{axis}: (none)"

    layout_stats = (
        "layout: N-A (non-image/layout-defining benchmark)"
        if support_info.get("bench_layout_na")
        else axis_stat("layout", lim.get("layouts"), "layout")
    )
    stats = [
        layout_stats,
        axis_stat(
            "inputKind",
            input_kinds,
            "inputKind",
        ),
        # bench dtype names (uchar3/float3...) don't match canonical Limitations tokens
        # (u8/f32...), so report the actually-benched dtype names by tier instead.
        axis_stat("dtypes(benched)", None, "dtypes"),
        axis_stat("channels", lim.get("channels"), "channels"),
    ]
    out.append(
        Finding(
            "BEN-13",
            "bench",
            REC,
            "Coverage statistics (per-axis × tier)",
            " | ".join(stats),
            g,
        )
    )
    # BEN-15/16 recommendations vs curated per-op basic-expected list
    _, _, basic_expected, _, _ = load_curated()
    rec = basic_expected.get(P.op)
    if rec:
        out.append(
            Finding(
                "BEN-15",
                "bench",
                REC,
                "Tiering check vs curated basic-expected list",
                f"expected in basic: {rec}",
                g,
                "Ensure listed combos are in basic; rest in advanced.",
            )
        )
    else:
        out.append(
            Finding(
                "BEN-15",
                "bench",
                REC,
                "No curated basic-expected list for this op (floor BEN-14 applies)",
                "advisory; add deterministic tool data or operator-local evidence "
                "if a stronger expectation is needed",
                g,
            )
        )
    # advisory: any supported layout/dtype entirely unbenched
    unbenched = []
    if not support_info.get("bench_layout_na"):
        for v in lim.get("layouts") or set():
            if v not in benched["all"]["layout"] and v not in ("HWC", "CHW"):
                unbenched.append("layout=" + v)
    for v in lim.get("dtypes") or set():
        pass  # dtype token mapping to bench dtype names is approximate; skip to avoid false recs
    if unbenched:
        out.append(
            Finding(
                "BEN-16",
                "bench",
                REC,
                "Unbenched supported axis values (advisory)",
                ", ".join(unbenched),
                g,
                "Consider adding to advanced (or basic if popular).",
            )
        )
    return out


# ============================================================================== DOCS domain
def check_docs(P: OpPaths, support_info):
    out = []
    g = ".agents/guidance/REVIEW_OP_GUIDELINES.md#docs / make_operator.rst / AGENTS.md"

    oplist = read(REPO / "docs/sphinx/operator_list.rst") or ""
    row = grep(rf":py:func:`cvcuda\.{re.escape(P.pyname)}`", oplist)
    out.append(
        Finding(
            "DOC-1",
            "docs",
            PASS if row else GAP,
            "operator_list.rst row present" if row else "operator_list.rst row MISSING",
            first_evidence(REPO / "docs/sphinx/operator_list.rst", row)
            or f"no :py:func:`cvcuda.{P.pyname}` row",
            g,
            "" if row else f"Add a row for cvcuda.{P.pyname} to operator_list.rst.",
        )
    )

    ops = read(REPO / "docs/sphinx/modules/python/operators.rst") or ""
    fn = grep(rf"cvcuda-autofunction::\s*cvcuda\.{re.escape(P.pyname)}\b", ops)
    into = grep(rf"cvcuda-autofunction::\s*cvcuda\.{re.escape(P.pyname)}_into\b", ops)
    ok = fn and into
    out.append(
        Finding(
            "DOC-2",
            "docs",
            PASS if ok else GAP,
            "Python autofunction directives (fn + _into)"
            if ok
            else "autofunction directive(s) missing",
            f"fn={'yes' if fn else 'no'} _into={'yes' if into else 'no'}",
            g,
            ""
            if ok
            else f"Add cvcuda-autofunction:: cvcuda.{P.pyname}{{,_into}} to operators.rst.",
        )
    )

    out.append(
        Finding(
            "DOC-3",
            "docs",
            MANUAL,
            "Confirm the operator appears in the C++ API reference",
            "check docs/sphinx C++ API listing / doxygen",
            g,
        )
    )

    # DOC-4 limitations-vs-code (cross-domain) — diff declared vs (enforced/tested) is partly in SUP-9/TST-5
    out.append(
        Finding(
            "DOC-4",
            "docs",
            MANUAL,
            "Limitations table vs code/tests consistency (cross-domain diff)",
            "compare SUP-6/7/8 declared vs SUP-9 enforced vs TST-5 tested",
            g,
            "Reconcile the Limitations table with the actual guards/tests.",
        )
    )

    pytext = read(P.pybind) or ""
    doc = grep(r'R"pbdoc|pbdoc\(|"""|Args:|Returns:', pytext)
    out.append(
        Finding(
            "DOC-5",
            "docs",
            PASS if doc else GAP,
            "Python binding docstrings present" if doc else "Python docstrings missing",
            first_evidence(P.pybind, doc) or rel(P.pybind),
            g,
            ""
            if doc
            else "Add pbdoc docstrings (args/returns/layouts) for the op + _into.",
        )
    )

    out.append(
        Finding(
            "DOC-6",
            "docs",
            MANUAL,
            "Doxygen param docs explain 'why' + match code",
            f"inspect {rel(P.header)}",
            g,
        )
    )

    # DOC-7 SPDX on the op's files (skip .json — JSON cannot carry comment headers)
    spdx_missing = []
    for f in [
        P.header,
        P.hpp,
        P.pybind,
        P.test_cpp,
        P.test_py,
        P.bench_cpp,
        P.bench_py,
    ]:
        if f.suffix == ".json":
            continue
        txt = read(f)
        if txt is not None and "SPDX-License-Identifier" not in txt[:600]:
            spdx_missing.append(rel(f))
    out.append(
        Finding(
            "DOC-7",
            "docs",
            PASS if not spdx_missing else GAP,
            "SPDX headers present on the op's files"
            if not spdx_missing
            else "SPDX header missing in: " + ", ".join(spdx_missing[:4]),
            "all checked files carry SPDX"
            if not spdx_missing
            else ", ".join(spdx_missing),
            "AGENTS.md",
            ""
            if not spdx_missing
            else "Add the SPDX 2026 header to the listed file(s).",
        )
    )
    return out


# ================================================================================= driver
def run(P: OpPaths, domains, curated, do_run):
    findings = []
    sup, support_info = check_support(P, curated)
    if "support" in domains:
        findings += sup
    if "test" in domains:
        findings += check_test(P, support_info)
    if "bench" in domains:
        findings += check_bench(P, support_info, do_run)
    if "docs" in domains:
        findings += check_docs(P, support_info)
    return findings


def render_md(P, findings, domains):
    icon = {PASS: "✅", GAP: "❌", NA: "➖", MANUAL: "🔍", REC: "💡"}
    lines = [f"# review-op: {P.Op}  (op={P.op}, py=cvcuda.{P.pyname})", ""]
    counts = {}
    for dom in domains:
        df = [f for f in findings if f.domain == dom]
        if not df:
            continue
        lines.append(f"## {dom}")
        for f in df:
            counts[f.status] = counts.get(f.status, 0) + 1
            lines.append(
                f"- {icon.get(f.status, '?')} **{f.status}** `{f.id}` — {f.summary}"
            )
            if f.evidence:
                lines.append(f"    - evidence: {f.evidence}")
            if f.status in (GAP,) and f.fix:
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
    lines.append("")
    lines.append("Completion = a re-run shows zero GAP and zero unresolved MANUAL.")
    return "\n".join(lines)


def render_json(P, findings, domains):
    counts = {}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
    return json.dumps(
        {
            "operator": P.Op,
            "op": P.op,
            "pyname": P.pyname,
            "domains": list(domains),
            "findings": [vars(f) for f in findings],
            "counts": counts,
            "exit_gap": counts.get(GAP, 0),
        },
        indent=2,
        sort_keys=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic per-operator review checker (CV-CUDA)."
    )
    ap.add_argument("operator", help="Operator name (PascalCase, e.g. CenterCrop)")
    ap.add_argument(
        "--domain", default="all", help="support|test|bench|docs|all (comma-separated)"
    )
    ap.add_argument("--format", default="md", choices=["md", "json"])
    ap.add_argument(
        "--out", default=None, help="write the report to this path as well as stdout"
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="attempt run-dependent bench checks (needs a built GPU env)",
    )
    ap.add_argument(
        "--fix",
        action="store_true",
        help="(wrapper-level) emphasize fix actions; the checker stays read-only",
    )
    args = ap.parse_args(argv)

    domains = (
        DOMAINS
        if args.domain == "all"
        else tuple(d.strip() for d in args.domain.split(",") if d.strip() in DOMAINS)
    )
    if not domains:
        ap.error("no valid --domain selected (choose from support,test,bench,docs,all)")

    if args.fix:
        print(
            "note: review_op.py is read-only; --fix is applied by the /review-op "
            "wrapper/agent per .agents/guidance/REVIEW_OP_GUIDELINES.md (see each GAP's 'fix' field).",
            file=sys.stderr,
        )

    P = resolve_op(args.operator)
    curated = load_curated()
    findings = run(P, domains, curated, args.run)

    report = (
        render_md(P, findings, domains)
        if args.format == "md"
        else render_json(P, findings, domains)
    )
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")

    return 1 if any(f.status == GAP for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
