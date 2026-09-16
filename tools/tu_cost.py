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
"""Per-translation-unit compile time and object size, for simplification evidence.

A simplification is supposed to make the build cheaper as well as the source shorter, but the
repo has no way to say so: the optimization workflow measures runtime, and the refactor workflow
measures LOC. Both miss "this deleted a 20-entry dispatch table and the file compiles 6% faster".

This measures one operator's translation units by replaying the exact command CMake recorded for
them, so the flags match a real build. Run it on two revisions and diff:

    python3 tools/tu_cost.py Invert --build-dir build-rel --out /tmp/base.json   # on main
    python3 tools/tu_cost.py Invert --build-dir build-rel --compare /tmp/base.json

Compile time is the best of --repeat runs (default 3), because best-of is far more stable than
the mean under a noisy machine: a slow run means something else took the core, never that the
compiler did less work. Object size is exact and needs no repetition.

Requires a configured build directory with compile_commands.json:

    cmake -S . -B build-rel -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENT_TOOLS = REPO / ".agents" / "tools"
if str(AGENT_TOOLS) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS))

from operator_source_map import (  # noqa: E402
    SHARED_KERNEL_SOURCES,
    all_op_names,
    legacy_belongs,
)


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def operator_sources(op: str):
    """The translation units CMake will compile for this operator — the same set the refactor and
    optimization tools attribute to it, so the three agree on what "this operator" means.
    """
    priv = REPO / "src/cvcuda/priv"
    out = [c for c in (priv / f"Op{op}.cu", priv / f"Op{op}.cpp") if c.exists()]
    legacy = priv / "legacy"
    if legacy.is_dir():
        ops = all_op_names()
        out += [
            g
            for g in sorted(legacy.glob("*.c*"))
            if legacy_belongs(g.stem, op.lower(), ops)
        ]
    for extra in SHARED_KERNEL_SOURCES.get(op.lower(), []):
        cand = priv / extra
        if cand.exists() and cand not in out and cand.suffix in (".cu", ".cpp"):
            out.append(cand)
    return out


def load_commands(build_dir: Path):
    cc = build_dir / "compile_commands.json"
    if not cc.is_file():
        sys.exit(
            f"error: {cc} not found. Configure with:\n"
            f"  cmake -S . -B {build_dir} -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        )
    by_file = {}
    for entry in json.loads(cc.read_text()):
        by_file[str(Path(entry["file"]).resolve())] = entry
    return by_file


def object_sizes(obj: Path):
    """Total object size, plus the .text delta when binutils is available. .text is the honest
    number for 'did this kernel get smaller' — total size moves with debug info too."""
    total = obj.stat().st_size
    text = None
    try:
        r = subprocess.run(["size", "-A", str(obj)], capture_output=True, text=True)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == ".text":
                    text = int(parts[1])
    except (OSError, ValueError):
        pass
    return total, text


def measure(entry, repeat, tmpdir: Path):
    """Replay the recorded compile with -o redirected into a scratch dir, so the real build tree
    is never written to and a failed measurement cannot poison an incremental build."""
    argv = shlex.split(entry["command"])
    stem = Path(entry["file"]).name
    obj = tmpdir / (stem + ".o")
    for i, a in enumerate(argv):
        if a == "-o" and i + 1 < len(argv):
            argv[i + 1] = str(obj)
            break
    else:
        argv += ["-o", str(obj)]

    # Any other output the recorded command produces has to be redirected too, or the replay
    # writes a depfile into the real build tree and can desynchronise an incremental build. This
    # project's CUDA commands carry none today, so the redirect is a guard, not a fix.
    for i, a in enumerate(argv):
        if a in ("-MF", "--dependency-output") and i + 1 < len(argv):
            argv[i + 1] = str(tmpdir / (stem + ".d"))
        elif a.startswith("-MF") and len(a) > 3:
            argv[i] = "-MF" + str(tmpdir / (stem + ".d"))

    best = None
    for _ in range(repeat):
        if obj.exists():
            obj.unlink()
        t0 = time.perf_counter()
        proc = subprocess.run(
            argv, cwd=entry.get("directory", str(REPO)), capture_output=True, text=True
        )
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            return {
                "error": (proc.stderr or proc.stdout).strip().splitlines()[-1][:200]
            }
        best = dt if best is None else min(best, dt)
    total, text = object_sizes(obj)
    return {"seconds": round(best, 3), "obj_bytes": total, "text_bytes": text}


def fmt_delta(now, base, unit="", pct=True):
    if base is None or now is None:
        return "n/a"
    d = now - base
    body = f"{d:+,.3f}{unit}" if isinstance(d, float) else f"{d:+,}{unit}"
    if pct and base:
        body += f" ({d / base * 100:+.1f}%)"
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-TU compile time and object size for a CV-CUDA operator."
    )
    ap.add_argument(
        "operator", nargs="?", help="Operator name (PascalCase, e.g. Invert)"
    )
    ap.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="measure a specific source file instead of an operator (repeatable)",
    )
    ap.add_argument(
        "--build-dir",
        default="build-rel",
        metavar="DIR",
        help="configured build dir holding compile_commands.json (default build-rel)",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=3,
        metavar="N",
        help="compiles per TU; the best is reported (default 3)",
    )
    ap.add_argument("--out", metavar="PATH", help="write the measurement as JSON")
    ap.add_argument(
        "--compare", metavar="PATH", help="diff against a JSON emitted by --out"
    )
    ap.add_argument("--format", default="md", choices=["md", "json"])
    args = ap.parse_args(argv)

    if not args.operator and not args.file:
        ap.error("give an operator name or at least one --file")
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = REPO / build_dir
    commands = load_commands(build_dir)

    # Load the baseline up front and fail as a CLI error: a missing or truncated --compare file
    # is a user mistake, and a traceback tells them less than the path that failed to parse.
    baseline = {}
    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text()).get(
                "translation_units", {}
            )
        except (OSError, ValueError) as exc:
            ap.error(f"--compare {args.compare}: {exc}")

    sources = [Path(f).resolve() for f in args.file]
    if args.operator:
        sources += operator_sources(args.operator)

    results = {}
    with tempfile.TemporaryDirectory(prefix="tu_cost_") as td:
        for src in sources:
            key = rel(src)
            entry = commands.get(str(Path(src).resolve()))
            if entry is None:
                results[key] = {
                    "error": "no compile_commands.json entry (not built by this config)"
                }
                continue
            results[key] = measure(entry, args.repeat, Path(td))

    payload = {
        "operator": args.operator,
        "build_dir": rel(build_dir),
        "repeat": args.repeat,
        "translation_units": results,
    }
    if args.format == "json":
        # --compare must still honour --format json: emit the deltas as data rather than
        # silently downgrading an explicit request to Markdown.
        if baseline:
            payload["baseline"] = baseline
            payload["delta"] = {
                k: {
                    "seconds": round(r["seconds"] - baseline[k]["seconds"], 3),
                    "obj_bytes": r["obj_bytes"] - baseline[k]["obj_bytes"],
                    "text_bytes": (
                        None
                        if r.get("text_bytes") is None
                        or baseline[k].get("text_bytes") is None
                        else r["text_bytes"] - baseline[k]["text_bytes"]
                    ),
                }
                for k, r in results.items()
                if "error" not in r and "error" not in baseline.get(k, {"error": 1})
            }
        report = json.dumps(payload, indent=2, sort_keys=True)
    else:
        base = baseline
        lines = [f"# tu-cost: {args.operator or 'files'}  (best of {args.repeat})", ""]
        if base:
            lines += [
                "| translation unit | compile s | Δ | .text | Δ |",
                "|---|---:|---:|---:|---:|",
            ]
        else:
            lines += [
                "| translation unit | compile s | obj bytes | .text bytes |",
                "|---|---:|---:|---:|",
            ]
        for key in sorted(results):
            r = results[key]
            if "error" in r:
                lines.append(
                    f"| `{key}` | — | {r['error'][:60]} | | |"
                    if base
                    else f"| `{key}` | — | {r['error'][:60]} | |"
                )
                continue
            if base:
                b = base.get(key, {})
                lines.append(
                    f"| `{key}` | {r['seconds']:.3f} | {fmt_delta(r['seconds'], b.get('seconds'), 's')} "
                    f"| {r['text_bytes'] if r['text_bytes'] is not None else 'n/a'} "
                    f"| {fmt_delta(r['text_bytes'], b.get('text_bytes'))} |"
                )
            else:
                lines.append(
                    f"| `{key}` | {r['seconds']:.3f} | {r['obj_bytes']:,} "
                    f"| {r['text_bytes'] if r['text_bytes'] is not None else 'n/a'} |"
                )
        ok = [r for r in results.values() if "error" not in r]
        if ok:
            lines += [
                "",
                f"total compile: {sum(r['seconds'] for r in ok):.3f}s across {len(ok)} TU(s)",
            ]
        report = "\n".join(lines)

    print(report)
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 1 if any("error" in r for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
