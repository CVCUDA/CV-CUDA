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
"""Device-code identity proof for one translation unit — VER-6 evidence with no build and no GPU.

`tools/refactor_op.py --phase verify` leaves VER-6 (bit-exact) as a `MANUAL` leg and returns
`NEEDS-LOCAL-PROOF`, expecting a full CUDA build plus the frozen `Op<Operator>` tests. Agents
frequently cannot do that: no GPU, no configured build tree, a shared machine. VER-6 then goes
unproven and is deferred to CI.

This compiles one translation unit on both sides of a refactor and classifies the per-kernel
difference. It needs neither a build tree nor a GPU; a compile is a couple of seconds.

    python3 tools/device_code_proof.py AdvCvtColor --base HEAD~1
    python3 tools/device_code_proof.py --file src/cvcuda/priv/OpHistogram.cu --arch sm_75 \
            --arch sm_90 --stage ptx --format json

Tiers of device-code identity proof, strongest first:

  1. Object identity — `cuobjdump --dump-elf-symbols` on the built `.o` (MIG-11 in
     `.agents/guidance/MODERNIZE_OP_GUIDELINES.md`, via `tools/modernize_op.py`). Covers every
     arch the build emits. Needs a configured build tree *and* a completed build.
  2. `.nv_fatbin` section compare (`cmp`) on two built objects — shows a change was pure
     relocation. Also needs a build.
  3. **`--stage sass`** (default) — per-arch machine code for a single TU. Only linking needs a
     build; compiling one TU does not.
  4. **`--stage ptx`** — per-arch PTX for a single TU. Weaker: identical PTX is strong evidence of
     identical SASS, not a guarantee. Useful when ptxas cannot target the arch of interest.

A non-empty diff is not a failure, and an unexplained non-empty diff is not a pass. `--revert PATH`
runs the control compile — take PATH from `--base` while the rest of the change stands — so the
edit responsible for a delta can be isolated and named instead of guessed at.

Flags come from `compile_commands.json` when a build directory is available, because a
hand-rolled flag set drifts from the real build and drifted flags can change codegen. The normal
CMake build does not emit one; generate it with

    ninja -C build-rel -t compdb > build-rel/compile_commands.json

Without a build directory the tool synthesizes a representative flag set and renders the
CMake-generated `VersionDef.h` / `VersionUtils.h` headers into a scratch directory, which is what
makes a single TU compilable against an unconfigured source tree. That fallback set is
*representative*, not the build's: it is identical on both sides, so the comparison stays valid,
but it does not prove anything about the flags CI actually uses.

Both sides are materialized into the *same* scratch path and compiled one after the other, so
nothing path-dependent (assert strings, `-lineinfo` line tables) can masquerade as a device-code
delta.

Exit codes: 0 `MATCH`, 1 real delta or compile failure, 2 usage error, 3 `NEEDS-JUDGEMENT`
(signature-only / schedule-only deltas that a human must interpret).
"""

import argparse
import difflib
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import tu_cost  # noqa: E402  (compile-DB discovery + operator->TU mapping live there)

GUIDE = ".agents/guidance/REFACTOR_OP_GUIDELINES.md"

# Two arches from different codegen families by default. One arch proves one arch; the pair is a
# cheap hedge against a change that is a no-op on one generation and not on another.
DEFAULT_ARCHES = ("sm_75", "sm_90")

IDENTICAL, SIGNATURE_ONLY, SCHEDULE_ONLY, CHANGED, ADDED, REMOVED = (
    "identical",
    "signature-only",
    "schedule-only",
    "changed",
    "added",
    "removed",
)
REAL_DELTA = (CHANGED, ADDED, REMOVED)
# Classes that are neither a pass nor a failure: a human has to name the cause before either.
JUDGEMENT = (SIGNATURE_ONLY, SCHEDULE_ONLY)

MATCH, NEEDS_JUDGEMENT, DEVICE_CODE_CHANGED, FAILED = (
    "MATCH",
    "NEEDS-JUDGEMENT",
    "DEVICE-CODE-CHANGED",
    "PROOF-FAILED",
)
EXIT = {MATCH: 0, DEVICE_CODE_CHANGED: 1, FAILED: 1, NEEDS_JUDGEMENT: 3}

# The generated headers a CV-CUDA TU needs before anything else will parse, and the CMake
# templates they are rendered from. Values below are placeholders: they reach host code only, and
# holding them fixed guarantees the two sides cannot differ because of them.
GENERATED_HEADERS = (
    ("cmake", "CVCUDA", "cvcuda"),
    ("src/nvcv/cmake", "NVCV", "nvcv"),
)
STUB_VALUES = {
    "VERSION_FULL": "0.0.0",
    "VERSION_MAJOR": "0",
    "VERSION_MINOR": "0",
    "VERSION_PATCH": "0",
    "VERSION_TWEAK": "0",
    "VERSION_SUFFIX": "",
    "REPO_COMMIT": "0" * 40,
}

# nvcc rejects `-ptx` alongside the multi-target `--generate-code` list a real build uses, and the
# dependency/fatbin flags would write files we do not want. Everything else from the recorded
# command is kept verbatim.
DROP_EXACT = {"-c", "--compile", "-dc", "--device-c", "-MD", "-MMD", "-M", "-MM"}
DROP_WITH_ARG = {"-o", "--output-file", "-MT", "-MF", "--dependency-output"}
DROP_PREFIX = (
    "--generate-code",
    "-gencode",
    "-arch",
    "--gpu-architecture",
    "-code=",
    "--gpu-code",
    "-Xfatbin",
    "--compress-mode",
    "-MF",  # also in DROP_WITH_ARG: both attached (-MFpath) and separate (-MF path) forms occur
    "-MT",  # same
    "-o",  # same
)

FALLBACK_FLAGS = (
    "-forward-unknown-to-host-compiler",
    "--extended-lambda",
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-x",
    "cu",
)
FALLBACK_INCLUDE_DIRS = ("src/nvcv/src/include", "src", "src/cvcuda/include")


# --------------------------------------------------------------------------- PTX normalization
COMMENT_RX = re.compile(r"//.*$", re.M)
# Debug directives carry source line numbers, which move whenever a refactor moves a line. They
# are not device code, so they are stripped rather than reported.
DEBUG_RX = re.compile(r"^\s*\.(?:loc|file)\b.*$", re.M)
LABEL_RX = re.compile(r"\$L__[\w$]+")
DEPOT_RX = re.compile(r"__local_depot\d+")
PARAM_RX = re.compile(r"@F@_param_(\d+)")
DEF_START_RX = re.compile(r"^(?:\.\w+\s+)*\.(entry|func)\b")
DATA_START_RX = re.compile(r"^(?:\.\w+\s+)*\.(global|const|shared)\b")
ENTRY_NAME_RX = re.compile(r"\.entry\s+([\w$]+)")
FUNC_NAME_RX = re.compile(r"\.func\s*(?:\([^)]*\)\s*)?([\w$]+)")
DATA_NAME_RX = re.compile(r"([A-Za-z_$][\w$]*)\s*(?:\[[^\]]*\])?\s*(?:=|;)")
# nvcc stamps every anonymous-namespace symbol with a per-*compilation* discriminator, so the
# mangled name of an unchanged kernel differs between two runs of the same command. Blanking it
# keeps the report reproducible and lets the exact-name matcher work without `c++filt`; the
# replacement preserves length so the Itanium length prefix stays valid and still demangles.
ANON_RX = re.compile(r"_GLOBAL__N__[0-9a-fA-F]+")


def canonical_symbol(name):
    return ANON_RX.sub(lambda m: "_GLOBAL__N__" + "0" * (len(m.group(0)) - 12), name)


def _squeeze(text):
    """Collapse whitespace and drop empty lines so formatting can never read as a code delta."""
    out = []
    for line in text.splitlines():
        s = " ".join(line.split())
        if s:
            out.append(s)
    return "\n".join(out)


def _strip_noise(text):
    return _squeeze(DEBUG_RX.sub("", COMMENT_RX.sub("", text)))


def _canonical_labels(text):
    """Rename `$L__BB7_3` by order of first appearance.

    Label numbers span the whole module and are not stable: adding or reordering any kernel
    shifts them throughout the file, so they cannot be compared across compilations as-is.
    """
    mapping = {}
    for name in LABEL_RX.findall(text):
        mapping.setdefault(name, "$L%d" % len(mapping))
    return LABEL_RX.sub(
        lambda m: mapping[m.group(0)], DEPOT_RX.sub("__local_depot", text)
    )


@dataclass
class Unit:
    """One PTX definition (`.entry`/`.func`) or module-scope datum."""

    kind: str
    name: str
    params: list = field(default_factory=list)
    strict: str = ""  # body with parameters left at their declared index
    loose: str = ""  # body with parameters renumbered by order of first use
    lines: int = 0
    # Rank of each used parameter's declaration index, in order of first use. Renumbering leaves
    # this untouched; reordering the parameter list does not, which is how the two are told apart.
    use_order: tuple = ()

    @property
    def digest(self):
        return hashlib.sha256(self.loose.encode()).hexdigest()[:16]


def _split_definition(raw, name):
    """Normalize one definition into (param declarations, strict body, loose body).

    The mangled name is substituted out first because PTX spells every parameter as
    `<mangled>_param_N`; without that, a renamed kernel differs on every single `ld.param` line
    and nothing downstream can tell a rename from a rewrite.
    """
    text = _canonical_labels(_strip_noise(raw).replace(name, "@F@"))
    lines = text.splitlines()
    try:
        brace = next(i for i, ln in enumerate(lines) if ln == "{")
    except StopIteration:  # declaration without a body
        return [], text, text, ()
    header, body = lines[:brace], "\n".join(lines[brace:])
    params = [ln.rstrip(",") for ln in header if ln.startswith(".param")]

    # Parameter identity is taken from order of first use in the body, not from the declared
    # index: dropping an unused parameter renumbers every later one, and that renumbering is
    # exactly the difference this tool has to be able to tell apart from a real edit.
    order = {}
    for idx in PARAM_RX.findall(body):
        order.setdefault(idx, "@P%d@" % len(order))
    loose = PARAM_RX.sub(lambda m: order[m.group(1)], body)

    # First-use renaming alone would also erase a *swapped* parameter pair, because both sides
    # would rename to the same tokens. Recording where each used slot sits among the used slots
    # keeps a reorder distinguishable from a renumbering.
    first_use = [int(i) for i in order]
    rank = {index: position for position, index in enumerate(sorted(first_use))}
    return params, body, loose, tuple(rank[i] for i in first_use)


def parse_ptx(text):
    """Split a PTX module into its header directives, definitions, and module-scope data."""
    header, units = {}, []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line, stripped = lines[i], lines[i].strip()
        if stripped.startswith((".version", ".target", ".address_size")):
            key, _, value = stripped.partition(" ")
            header[key.lstrip(".")] = value.strip()
            i += 1
            continue
        if DEF_START_RX.match(line):
            start, saw_body = i, False
            while i < n:
                if lines[i] == "{":
                    saw_body = True
                if saw_body and lines[i] == "}":
                    break
                if not saw_body and lines[i].rstrip().endswith(";"):
                    break  # forward declaration
                i += 1
            end = min(i + 1, n)
            raw = "\n".join(lines[start:end])
            kind = "entry" if ".entry" in lines[start] else "func"
            rx = ENTRY_NAME_RX if kind == "entry" else FUNC_NAME_RX
            found = rx.search(raw)
            if found and saw_body:
                params, strict, loose, use_order = _split_definition(
                    raw, found.group(1)
                )
                units.append(
                    Unit(
                        kind,
                        canonical_symbol(found.group(1)),
                        params,
                        strict,
                        loose,
                        len(loose.splitlines()),
                        use_order,
                    )
                )
            i += 1
            continue
        if DATA_START_RX.match(line):
            start = i
            while i < n and not lines[i].rstrip().endswith(";"):
                i += 1
            end = min(i + 1, n)
            raw = "\n".join(lines[start:end])
            found = DATA_NAME_RX.search(lines[start][:400])
            if found:
                norm = _strip_noise(raw)
                units.append(Unit("data", found.group(1), [], norm, norm, len(raw)))
            i += 1
            continue
        i += 1
    return header, units


SASS_FUNC_RX = re.compile(r"^\s*Function\s*:\s*(\S+)")


def parse_sass(text):
    """Split `cuobjdump -sass` output into per-kernel instruction streams.

    Two normalizations, mirroring PTX's: `loose` is the disassembled instruction text alone,
    `strict` adds the encoding words. Instruction addresses are dropped from both — a single
    added instruction shifts every later address, which would swamp the diff with noise that
    says nothing beyond "the code got longer".
    """
    header, units = {}, []
    name, insns, encs = None, [], []

    def flush():
        if name is None:
            return
        loose = "\n".join(insns)
        units.append(
            Unit(
                "entry",
                canonical_symbol(name),
                [],
                loose + "\n;;\n" + "\n".join(encs),
                loose,
                len(insns),
            )
        )

    for line in text.splitlines():
        found = SASS_FUNC_RX.match(line)
        if found:
            flush()
            name, insns, encs = found.group(1), [], []
            continue
        stripped = line.strip()
        if stripped.startswith((".target", "code for")):
            header["target"] = stripped.split()[-1]
            continue
        if name is None or not stripped.startswith("/*") or "*/" not in stripped:
            continue
        body = stripped.split("*/", 1)[1].strip()
        if not body:  # continuation line carrying only the control word
            encs.append(stripped)
            continue
        text_part, sep, encoding = body.rpartition("/*")
        insns.append(" ".join((text_part if sep else body).split()))
        if sep:
            encs.append(encoding.strip())
    flush()
    return header, units


PARSERS = {"ptx": parse_ptx, "sass": parse_sass}


# --------------------------------------------------------------------------- classification
def demangle(names):
    """Demangled names, for the report only. Matching never depends on `c++filt` being present."""
    names = [n for n in dict.fromkeys(names) if n.startswith("_Z")]
    if not names or not shutil.which("c++filt"):
        return {}
    proc = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True
    )
    if proc.returncode != 0:
        return {}
    out = proc.stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {}


def _base_name(demangled):
    """The demangled name with its argument list removed, so a kernel that lost a parameter still
    matches its own previous self."""
    depth = 0
    for i in range(len(demangled) - 1, -1, -1):
        char = demangled[i]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                return demangled[:i].strip()
    return demangled


def _short(name, width=76):
    """Elide the middle, not the tail: sibling instantiations of one kernel template differ in
    their *last* template argument, so a right-truncated name makes two distinct kernels look
    like the same row."""
    if len(name) <= width:
        return name
    return name[: width - 26] + "…" + name[-25:]


def match_units(base, head, names):
    """Pair units across the two sides: exact mangled name, then demangled base name, then a
    unique identical body. Whatever is left is genuinely added or removed."""
    pairs, left, right = [], list(base), list(head)

    def take(key_of):
        nonlocal left, right
        index = {}
        for unit in right:
            index.setdefault(key_of(unit), []).append(unit)
        keep = []
        for unit in left:
            bucket = index.get(key_of(unit), [])
            if len(bucket) == 1 and bucket[0] in right:
                pairs.append((unit, bucket[0]))
                right.remove(bucket[0])
            else:
                keep.append(unit)
        left = keep

    take(lambda u: (u.kind, u.name))
    take(lambda u: (u.kind, _base_name(names.get(u.name, u.name))))
    take(lambda u: (u.kind, u.digest))
    return pairs, left, right


def _diff_lines(before, after):
    return [
        ln
        for ln in difflib.unified_diff(before, after, lineterm="", n=0)
        if ln[:1] in "+-" and not ln.startswith(("---", "+++"))
    ]


def classify(before, after):
    """One of: identical / signature-only / changed, with the evidence a reader needs.

    `signature-only` is never a pass on its own: it is what removing an unused parameter looks
    like, and equally what silently swapping two parameters looks like, so the detail has to carry
    the parameter diff a human needs in order to tell those apart.
    """
    if before.name == after.name and before.strict == after.strict:
        return IDENTICAL, ""
    if before.loose == after.loose:
        if before.use_order != after.use_order:
            return SIGNATURE_ONLY, (
                "parameter use order changed %s -> %s — a reorder, not a renumbering"
                % (list(before.use_order), list(after.use_order))
            )
        if before.name == after.name and before.params == after.params:
            # SASS only: the instruction stream is the same but its encoding is not, which means
            # ptxas scheduled it differently (stall counts, barriers, reuse flags).
            return SCHEDULE_ONLY, (
                "same %d-instruction stream, different encoding/scheduling"
                % before.lines
            )
        detail = []
        if before.name != after.name:
            detail.append("mangled name changed")
        if before.params != after.params:
            # Index the declarations so a dropped parameter reads as "slot 3 went away" rather
            # than as six renamed slots.
            def slots(params):
                return ["[%d] %s" % (i, p) for i, p in enumerate(params)]

            changes = _diff_lines(slots(before.params), slots(after.params))
            detail.append(
                "params %d -> %d: %s"
                % (
                    len(before.params),
                    len(after.params),
                    " ".join(c.replace("@F@_param_", "p") for c in changes[:4]),
                )
            )
        return SIGNATURE_ONLY, "; ".join(detail) or "declaration differs"
    diff = _diff_lines(before.loose.splitlines(), after.loose.splitlines())
    # The instruction-count delta is the first question asked of a non-empty diff: "16 shorter
    # with the same FP count" points at code motion, an unchanged count at scheduling.
    return CHANGED, "%d changed line(s), %+d instructions; e.g. %s" % (
        len(diff),
        after.lines - before.lines,
        " | ".join(d[:60] for d in diff[:3]),
    )


# --------------------------------------------------------------------------- compilation
def stub_generated_headers(dest: Path):
    """Render the CMake-generated version headers so a TU compiles against an unconfigured tree.

    Rendered once from the working tree and shared by both sides: the values are host-only, and a
    single copy makes it impossible for the stub itself to show up as a device-code delta. The
    directory is appended *after* the recorded include paths, so a real build tree's generated
    headers still win when one is available.
    """
    written = []
    for tmpl_dir, prefix, incdir in GENERATED_HEADERS:
        for name, out in (
            ("VersionDef.h.in", "%s/VersionDef.h" % incdir),
            ("VersionUtils.h.in", "%s/detail/VersionUtils.h" % incdir),
        ):
            src = REPO / tmpl_dir / name
            if not src.is_file():
                continue
            text = src.read_text(encoding="utf-8").replace("@LIBPREFIX@", prefix)
            for key, value in STUB_VALUES.items():
                text = text.replace("@%s@" % key, value)
            path = dest / out
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            written.append(out)
    return written


def include_dirs(argv):
    out = []
    for i, arg in enumerate(argv):
        if arg.startswith("-I") and len(arg) > 2:
            out.append(arg[2:])
        elif arg in ("-I", "-isystem", "--include-path") and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif arg.startswith("-isystem") and len(arg) > 8:
            out.append(arg[8:])
    return out


def moved_tops(argv, tus, build_dir: Path):
    """Top-level repo directories that must be materialized per side and redirected into the
    scratch tree. The build directory is excluded on purpose: its generated headers are not
    versioned, so there is no per-ref copy of them to redirect to."""
    tops = {"src"}
    candidates = [Path(p) for p in include_dirs(argv)] + list(tus)
    for path in candidates:
        try:
            resolved = Path(path).resolve()
            relative = resolved.relative_to(REPO)
        except (ValueError, OSError):
            continue
        if not relative.parts or resolved == build_dir or build_dir in resolved.parents:
            continue
        tops.add(relative.parts[0])
    return sorted(tops)


def materialize(ref, dest: Path, paths):
    """Put `<ref>`'s (or the working tree's) copy of `paths` at `dest`.

    `git archive` reads the object database only — it never touches the index, HEAD, or any other
    worktree, which matters on a machine where several worktrees share one repository.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if ref is None:
        producer = subprocess.Popen(
            ["tar", "-C", str(REPO), "-cf", "-", *paths],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        producer = subprocess.Popen(
            ["git", "-C", str(REPO), "archive", "--format=tar", ref, "--", *paths],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    extract = subprocess.run(
        ["tar", "-C", str(dest), "-xf", "-"], stdin=producer.stdout, capture_output=True
    )
    producer.stdout.close()
    if producer.wait() != 0 or extract.returncode != 0:
        err = (producer.stderr.read().decode() if producer.stderr else "") or (
            extract.stderr.decode()
        )
        raise RuntimeError(
            "could not materialize %s: %s" % (ref or "working tree", err.strip())
        )


INCLUDE_FLAGS = ("-I", "-isystem", "--include-path=")


def redirect(argv, tops, scratch: Path):
    """Point every repo-internal path in a recorded command at the materialized side.

    Include flags are unpacked first. Missing that leaves `-I<repo>/src` untouched, and then the
    base side compiles its own `.cu` against the *working tree's* headers — so a refactor that
    moved code into a shared header would compare that header against itself and report a
    guaranteed match.
    """
    prefixes = [str(REPO / top) for top in tops]
    repo_len = len(str(REPO))

    def swap(token):
        for prefix in prefixes:
            if token == prefix or token.startswith(prefix + "/"):
                return str(scratch) + token[repo_len:]
        return token

    out = []
    for token in argv:
        for flag in INCLUDE_FLAGS:
            width = len(flag)
            if token.startswith(flag) and len(token) > width:
                out.append(flag + swap(token[width:]))
                break
        else:
            out.append(swap(token))
    return out


def compile_flags(entry, tu: Path, build_dir: Path):
    """The nvcc argv for one TU, minus everything that conflicts with `-ptx`.

    Returns (argv-without-source, source-path, flag-source-label).
    """
    if entry is None:
        argv = [
            "nvcc",
            *FALLBACK_FLAGS,
            *["-I%s" % (REPO / d) for d in FALLBACK_INCLUDE_DIRS],
        ]
        return argv, str(tu), "synthesized (no compile_commands.json entry)"

    raw = shlex.split(entry["command"])
    out, skip = [], False
    tu_resolved = str(tu.resolve())
    for i, arg in enumerate(raw):
        if skip:
            skip = False
            continue
        if arg in DROP_WITH_ARG:
            skip = True
            continue
        if arg in DROP_EXACT or arg.startswith(DROP_PREFIX):
            continue
        if arg in ("-I", "-isystem", "--include-path") and i + 1 < len(raw):
            # Kept as a pair so the path that follows is never mistaken for the source positional.
            out += [arg, raw[i + 1]]
            skip = True
            continue
        try:
            if str(Path(arg).resolve()) == tu_resolved:
                continue  # the source positional is re-added against the scratch tree
        except OSError:
            pass
        out.append(arg)
    return out, str(tu), "compile_commands.json (%s)" % tu_cost.rel(build_dir)


def emit_device_code(argv, source, arch, stage, out_path: Path, cwd):
    """Compile one TU and return its device code as text, for `stage` in {ptx, sass}."""
    phase = "-ptx" if stage == "ptx" else "-cubin"
    proc = subprocess.run(
        [*argv, phase, "-arch=%s" % arch, source, "-o", str(out_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None, "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
    if stage == "ptx":
        return out_path.read_text(encoding="utf-8", errors="replace"), None
    dump = subprocess.run(
        ["cuobjdump", "-sass", str(out_path)], capture_output=True, text=True
    )
    if dump.returncode != 0:
        return None, (dump.stderr or dump.stdout).strip()[:400]
    return dump.stdout, None


# --------------------------------------------------------------------------- reporting
def limits(stage):
    common = [
        "**Proves:** device code for the arch(es) compiled above, for these translation units "
        "only — nothing about any arch that was not compiled.",
        "**Does not prove** anything host-side: validation order, thrown exception types, error "
        "message text, workspace sizing, API/ABI. Those need the other VER legs and review.",
        "**Not a substitute** for the frozen `Op<Operator>` tests. This is evidence *for* VER-6; "
        "reporting it never licenses skipping the test run.",
    ]
    if stage == "ptx":
        return common + [
            "**PTX, not SASS.** Identical PTX is strong evidence of identical machine code but "
            "not a guarantee — ptxas may still schedule or allocate differently. Re-run with "
            "`--stage sass` for the stronger form; it costs about the same."
        ]
    return common + [
        "**SASS for one arch.** Stronger than PTX and stricter: a benign signature change moves "
        "the parameter constant-bank offsets, so it surfaces here as a real delta that PTX would "
        "have classed `signature-only`. Still weaker than comparing built objects "
        "(`cuobjdump --dump-elf-symbols`), which covers every arch the build emits."
    ]


def verdict_of(results):
    if any(r.get("error") for r in results):
        return FAILED
    classes = [u["class"] for r in results for u in r["units"]]
    if any(c in REAL_DELTA for c in classes):
        return DEVICE_CODE_CHANGED
    if any(c in JUDGEMENT for c in classes):
        return NEEDS_JUDGEMENT
    return MATCH


def _md_cell(text):
    """Escape pipe characters and newlines so a detail string doesn't break a Markdown table row."""
    return str(text).replace("|", r"\|").replace("\n", "<br>")


def render_md(payload):
    lines = [
        "# device-code-proof: %s" % (payload["operator"] or "files"),
        "",
        "- base: `%s` -> head: `%s`" % (payload["base"], payload["head"]),
        "- stage: %s" % payload["stage"],
        "- flags: %s" % payload["flag_source"],
        "- arches: %s" % ", ".join(payload["arches"]),
    ]
    if payload.get("reverted"):
        lines.append(
            "- control compile: %s restored from `%s`"
            % (", ".join("`%s`" % p for p in payload["reverted"]), payload["base"])
        )
    if payload["skipped"]:
        lines.append(
            "- skipped (no device code): %s"
            % ", ".join("`%s`" % s for s in payload["skipped"])
        )
    lines.append("")
    for result in payload["results"]:
        lines.append("## `%s` @ %s" % (result["tu"], result["arch"]))
        if result.get("error"):
            lines += ["", "```", result["error"], "```", ""]
            continue
        counts = result["counts"]
        lines += [
            "",
            "| device symbol | class | detail |",
            "|---|---|---|",
        ]
        for unit in result["units"]:
            if unit["class"] == IDENTICAL and counts.get(IDENTICAL, 0) > 6:
                continue  # the interesting rows are the ones that are not identical
            lines.append(
                "| `%s` | %s | %s |"
                % (
                    _md_cell(_short(unit["name"])),
                    unit["class"],
                    _md_cell(_short(unit["detail"], 110)),
                )
            )
        if counts.get(IDENTICAL, 0) > 6:
            lines.append(
                "| _(%d identical symbols omitted)_ | identical | |" % counts[IDENTICAL]
            )
        lines += [
            "",
            ", ".join("%s %d" % (k, v) for k, v in sorted(counts.items())),
            "",
        ]
    lines += ["## verdict", "", "**%s**" % payload["verdict"], ""]
    if payload["verdict"] == NEEDS_JUDGEMENT:
        lines += [
            "A non-empty diff is not a failure, and an unexplained non-empty diff is not a pass — "
            "the deliverable is the named cause. Signature-only deltas are *expected* when a "
            "signature legitimately changed (an unused parameter removed), and are also what a "
            "silently reordered parameter list looks like. Isolate the cause with a control "
            "compile (`--revert <file>`, then narrow to the hunk), then record the judgement with "
            "`--accept-signature-change` — the flag records it, it does not make it.",
            "",
        ]
    if payload["verdict"] == DEVICE_CODE_CHANGED:
        lines += [
            "Bisect before concluding: re-run with `--revert <file>` for each file the change "
            "touched until the delta disappears, then narrow to the hunk. Classify the isolated "
            "edit as code motion / parameter renumbering / scheduling / a real semantic change. "
            "Only the last is a VER-6 failure.",
            "",
        ]
    lines += ["## what this does and does not prove", ""] + [
        "- " + item for item in payload["limits"]
    ]
    lines += ["", "Ref: `%s` (VER-6)." % GUIDE]
    return "\n".join(lines)


# --------------------------------------------------------------------------- driver
def overlay(ref, dest: Path, paths):
    """Restore `paths` from `ref` on top of an already-materialized side — the control compile.

    A non-empty diff says codegen changed but not why. Reverting one candidate edit and
    recompiling is what turns that into a named cause, and doing it here means the working tree
    never has to be edited back and forth to run the experiment.
    """
    for name in paths:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if ref is None:
            shutil.copyfile(REPO / name, target)
            continue
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", "%s:%s" % (ref, name)], capture_output=True
        )
        if blob.returncode != 0:
            raise RuntimeError("cannot read %s from %s" % (name, ref))
        target.write_bytes(blob.stdout)


def run_side(ref, scratch: Path, plan, arches, stage, stub_inc: Path, tops, revert=()):
    """Materialize one side into the shared scratch path and emit its device code.

    Both sides reuse `scratch`, so the two compiles see byte-identical source paths. `cwd` is the
    scratch root rather than the build directory the command was recorded in: resolving a relative
    include against the original build tree would silently give both sides the *same* headers.
    """
    if scratch.exists():
        shutil.rmtree(scratch)
    materialize(ref, scratch, tops)
    if revert:
        overlay(revert[0], scratch, revert[1])
    out = {}
    for tu, argv, _label in plan:
        rel_tu = tu_cost.rel(tu)
        source = redirect([str(tu)], tops, scratch)[0]
        flags = redirect(argv, tops, scratch) + ["-I%s" % stub_inc]
        for arch in arches:
            out[(rel_tu, arch)] = emit_device_code(
                flags, source, arch, stage, scratch.parent / "out.bin", scratch.parent
            )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Device-code identity proof for a CV-CUDA refactor: VER-6 evidence "
        "with no build tree and no GPU."
    )
    parser.add_argument("operator", nargs="?", help="operator name, e.g. AdvCvtColor")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="prove a specific translation unit instead of an operator (repeatable)",
    )
    parser.add_argument(
        "--base", default="HEAD", metavar="REF", help="the 'before' ref (default HEAD)"
    )
    parser.add_argument(
        "--head",
        metavar="REF",
        help="the 'after' ref (default: the working tree)",
    )
    parser.add_argument(
        "--arch",
        action="append",
        default=[],
        metavar="SM",
        help="target to compile for, repeatable (default %s)"
        % " ".join(DEFAULT_ARCHES),
    )
    parser.add_argument(
        "--stage",
        default="sass",
        choices=["sass", "ptx"],
        help="device-code form to compare; sass is stronger and costs the same (default sass)",
    )
    parser.add_argument(
        "--build-dir",
        default="build-rel",
        metavar="DIR",
        help="build dir holding compile_commands.json; falls back to a synthesized flag set",
    )
    parser.add_argument(
        "--revert",
        action="append",
        default=[],
        metavar="PATH",
        help="control compile: take PATH from --base while the rest of the head side stands, to "
        "isolate which edit caused a delta (repeatable)",
    )
    parser.add_argument(
        "--accept-signature-change",
        action="store_true",
        help="record that the signature-only/schedule-only deltas were read and judged correct",
    )
    parser.add_argument(
        "--keep",
        metavar="DIR",
        help="write both sides' device code here for inspection",
    )
    parser.add_argument("--out", metavar="PATH", help="write the report to a file")
    parser.add_argument("--format", default="md", choices=["md", "json"])
    args = parser.parse_args(argv)

    if not args.operator and not args.file:
        parser.error("give an operator name or at least one --file")
    # Validate --file arguments before expensive checks: a path that escapes the
    # repository would be compiled from disk on both sides and return a guaranteed
    # MATCH for code that was never versioned at --base.
    sources = []
    for raw in args.file:
        path = Path(raw).resolve()
        if not path.is_file():
            parser.error("no such file: %s" % raw)
        try:
            path.relative_to(REPO)
        except ValueError:
            parser.error("--file must point inside the repository: %s" % raw)
        sources.append(path)
    # Validate --revert here too, before the nvcc check, so out-of-repo paths are caught
    # cheaply and do not trigger the nvcc-not-found diagnostic first.
    for raw in args.revert:
        path = Path(raw).resolve()
        try:
            path.relative_to(REPO)
        except ValueError:
            parser.error("--revert must point inside the repository: %s" % raw)
    if not shutil.which("nvcc"):
        parser.error(
            "nvcc not found on PATH; this proof needs the CUDA toolkit (no GPU required)"
        )
    if args.stage == "sass" and not shutil.which("cuobjdump"):
        parser.error("cuobjdump not found on PATH; re-run with --stage ptx")

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = REPO / build_dir
    commands = (
        tu_cost.load_commands(build_dir)
        if (build_dir / "compile_commands.json").is_file()
        else {}
    )

    if args.operator:
        sources += tu_cost.operator_sources(args.operator)
    if not sources:
        parser.error("no translation units found for %s" % args.operator)
    # A .cpp TU has no device code, so PTX says nothing about it. Naming the skips keeps this
    # tool's blind spot visible in its own report instead of implying whole-operator coverage.
    skipped = [tu_cost.rel(s) for s in sources if s.suffix != ".cu"]
    tus = [s for s in sources if s.suffix == ".cu"]
    if not tus:
        parser.error(
            "no .cu translation units in %s — PTX proves nothing about host-only sources"
            % (args.operator or "the given files")
        )

    arches = tuple(args.arch) or DEFAULT_ARCHES
    plan = []
    for tu in tus:
        entry = commands.get(str(tu.resolve()))
        argv_flags, _source, label = compile_flags(entry, tu, build_dir)
        plan.append((tu, argv_flags, label))

    all_argv = [a for _tu, flags, _l in plan for a in flags]
    tops = moved_tops(all_argv, tus, build_dir)
    reverted = []
    for raw in args.revert:
        path = Path(raw).resolve()
        try:
            reverted.append(str(path.relative_to(REPO)))
        except ValueError:
            parser.error("--revert must point inside the repository: %s" % raw)

    with tempfile.TemporaryDirectory(prefix="device_code_proof_") as td:
        root = Path(td)
        stub_inc = root / "generated"
        stub_generated_headers(stub_inc)
        scratch = root / "tree"
        try:
            before = run_side(
                args.base, scratch, plan, arches, args.stage, stub_inc, tops
            )
            after = run_side(
                args.head,
                scratch,
                plan,
                arches,
                args.stage,
                stub_inc,
                tops,
                revert=(args.base, reverted) if reverted else (),
            )
        except RuntimeError as exc:
            parser.exit(1, "error: %s\n" % exc)
        if args.keep:
            keep = Path(args.keep)
            keep.mkdir(parents=True, exist_ok=True)
            for label, side in (("base", before), ("head", after)):
                for (tu, arch), (text, _err) in side.items():
                    if text:
                        name = "%s.%s.%s.%s" % (Path(tu).name, arch, label, args.stage)
                        (keep / name).write_text(text, encoding="utf-8")

    results = []
    for tu, _flags, _label in plan:
        for arch in arches:
            key = (tu_cost.rel(tu), arch)
            base_text, base_err = before[key]
            head_text, head_err = after[key]
            record = {"tu": key[0], "arch": arch, "units": [], "counts": {}}
            if base_err or head_err:
                record["error"] = (
                    "base: %s" % base_err if base_err else "head: %s" % head_err
                )
                results.append(record)
                continue
            base_hdr, base_units = PARSERS[args.stage](base_text)
            head_hdr, head_units = PARSERS[args.stage](head_text)
            if base_hdr != head_hdr:
                record["error"] = (
                    "module headers differ (%s vs %s) — the two sides are not comparable"
                    % (base_hdr, head_hdr)
                )
                results.append(record)
                continue
            names = demangle([u.name for u in base_units + head_units])
            pairs, only_base, only_head = match_units(base_units, head_units, names)
            for old, new in pairs:
                klass, detail = classify(old, new)
                record["units"].append(
                    {
                        "name": _base_name(names.get(new.name, new.name)),
                        "mangled": new.name,
                        "kind": new.kind,
                        "class": klass,
                        "detail": detail,
                    }
                )
            for unit, klass in [(u, REMOVED) for u in only_base] + [
                (u, ADDED) for u in only_head
            ]:
                record["units"].append(
                    {
                        "name": _base_name(names.get(unit.name, unit.name)),
                        "mangled": unit.name,
                        "kind": unit.kind,
                        "class": klass,
                        "detail": "%s only on the %s side"
                        % (unit.kind, "base" if klass == REMOVED else "head"),
                    }
                )
            record["units"].sort(key=lambda u: (u["class"] == IDENTICAL, u["mangled"]))
            for unit in record["units"]:
                record["counts"][unit["class"]] = (
                    record["counts"].get(unit["class"], 0) + 1
                )
            results.append(record)

    verdict = verdict_of(results)
    if verdict == NEEDS_JUDGEMENT and args.accept_signature_change:
        verdict = MATCH
    payload = {
        "operator": args.operator,
        "base": args.base,
        "head": args.head or "<working tree>",
        "stage": args.stage,
        "arches": list(arches),
        "flag_source": "; ".join(dict.fromkeys(label for _tu, _flags, label in plan)),
        "reverted": reverted,
        "skipped": skipped,
        "results": results,
        "verdict": verdict,
        "signature_change_accepted": bool(args.accept_signature_change),
        "limits": limits(args.stage),
    }
    report = (
        json.dumps(payload, indent=2, sort_keys=True)
        if args.format == "json"
        else render_md(payload)
    )
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
