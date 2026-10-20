[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Device Code Proof

Procedure for the `/device-code-proof` dispatcher (`.agents/skills/op-tools/skills/device-code-proof/SKILL.md`);
the shared operating contract is `conventions.md`. The substance is the deterministic checker
`tools/device_code_proof.py` and the VER-6 leg of `.agents/guidance/REFACTOR_OP_GUIDELINES.md`;
do not restate the classification rules here.

This exists because VER-6 is otherwise unprovable in most agent environments: `tools/refactor_op.py
--phase verify` returns `NEEDS-LOCAL-PROOF` and asks for a full CUDA build plus the frozen
`Op<Operator>` tests, which needs a GPU and a configured build tree. Compiling **one translation
unit** needs neither, and takes seconds.

## Which tier to use

| tier | needs | strength |
|---|---|---|
| object device identity — `cuobjdump --dump-elf-symbols` on the built `.o` | a completed build | strongest; every arch the build emits |
| `cmp` on the extracted `.nv_fatbin` section of two objects | a completed build | strong; shows a change was pure relocation |
| **`device_code_proof.py --stage sass`** | nvcc only | very strong, per-arch; catches what PTX cannot |
| **`device_code_proof.py --stage ptx`** | nvcc only | strong; PTX-identical is *not* SASS-identical |
| `nvcc -cuda` on the TU | nvcc only | syntax only |

Use the strongest tier your environment allows, and say which one you ran. The object tier is
`MIG-11` in `.agents/guidance/MODERNIZE_OP_GUIDELINES.md` (arriving with the modernize-op harness,
via `tools/modernize_op.py --phase verify --obj-base ... --obj-new ...`); prefer it whenever a build
already exists. This skill is the **no-build-tree** rung of the same idea, not a competitor, and it
does not license skipping a measurement that a build would have made possible.

## Static resource comparison (not an identity proof)

```bash
python3 tools/device_code_proof.py <Op> --base <ref> --head <ref> \
  --arch sm_80 --stage resources --keep /tmp/resources --format json
```

This reuses the same materialization and compile flags, adding ptxas verbose diagnostics
and cubin resource/SASS dumps. Each kernel reports registers, stack bytes, spill
load/store bytes, cubin local/static shared bytes, static LDL/STL counts and signed
deltas. Explicit zero is distinct from missing data; unmatched symbols stay unknown.
`--keep` preserves the diagnostics for inspection. Register/shared-memory increases
can constrain occupancy, but block size, dynamic shared memory, device limits and
allocation granularity are needed to assess it; the tool does not guess occupancy.

Exit 0 is `RESOURCE-REPORT` (complete data, no increases), 1 is `RESOURCE-INCREASE`
or `PROOF-FAILED` (compile/dump error), and 3 is `RESOURCE-DATA-MISSING` (including
empty/unmatched kernels). Missing data takes precedence over increases, which remain
listed per kernel. No resource status is a performance PASS or VER-6 proof.

ptxas spill bytes and static instruction counts are not executed traffic. Stack size
is not the live local working set; `LOCAL:0` does not exclude spills. Confirm runtime
local traffic, cache behavior, occupancy and timing with Nsight Compute on the actual
target device. Synthesized flags remain representative, not the release build's flags.

## Workflow

1. **prove** — `python3 tools/device_code_proof.py <Operator> --base <ref>` (or `--file <TU>`;
   `--head <ref>` when comparing two commits rather than the working tree; `--format json`).
   Defaults to `--stage sass` and two arches. Flags are lifted from `compile_commands.json` when a
   build dir exists — a hand-rolled flag set drifts from the real build and drifted flags change
   codegen. The normal CMake build does not write one:
   `ninja -C build-rel -t compdb > build-rel/compile_commands.json`. With no build dir the tool
   synthesizes a representative flag set and renders the generated `VersionDef.h` headers into a
   scratch dir, which is what makes an unconfigured tree compilable at all.

   Costs measured on one operator TU, no GPU: ~2 s per compile, ~4 s for a one-arch two-side proof,
   ~9 s for a three-TU legacy operator. Compiling with the project's full warning set is free — do
   it and report the warning count.

2. **classify** — the tool reports each kernel as `identical`, `signature-only` (mangled name
   and/or parameter slots moved, instruction stream unchanged), `schedule-only` (SASS: same
   instructions, different encoding/scheduling), `changed`, `added`, or `removed`. Exit 0 only for
   a clean match; 3 for anything needing judgement; 1 for a real delta.

3. **bisect with a control compile** — **a non-empty diff is not a failure, and an unexplained
   non-empty diff is not a pass.** The deliverable is the *named cause*. Do not stop at "codegen
   changed" and do not hand-wave it. Re-run with one candidate edit reverted until the delta
   disappears:
   ```bash
   python3 tools/device_code_proof.py <Op> --base <ref> --revert src/cvcuda/priv/<File>.cuh
   ```
   `--revert` takes that file from `--base` while the rest of the change stands, so the working
   tree never has to be edited back and forth. When one file is not granular enough, revert a
   single hunk in the working tree and re-run. Then name what the isolated edit was:
   code motion (a hoist above a branch), parameter renumbering, instruction-neutral scheduling, or
   a **real semantic change** — only the last is a VER-6 failure. A worked example: two kernels 16
   instructions shorter at an unchanged FP instruction count turned out to be a hoisted pair of
   integer ceil-divides; reverting just that hoist made all 12 kernels byte-identical.

4. **report** — paste the verdict, the per-class counts, the arch(es), the flag source, and the
   named cause of every non-identical kernel. `--accept-signature-change` records a judgement you
   have made and written down; it does not make one.

## What this does not prove

State these whenever you cite the result; the tool prints them too.

- Only the **arch(es) actually compiled**. Nothing about any other arch.
- **Nothing host-side**: validation order, thrown exception types, error message text, workspace
  sizing, API/ABI. Those are VER-1..VER-5 and review.
- Identical PTX does **not** guarantee identical SASS — that is why `--stage sass` exists.
- `signature-only` and `schedule-only` are *expected* for some legitimate changes and are also
  what a silently reordered parameter list looks like. They are never auto-accepted.
- It is **evidence for** VER-6, not a replacement for the frozen `Op<Operator>` test run. Having a
  clean proof never licenses skipping the tests.

## Prompt Handling

Requests like "prove this refactor is bit-exact", "/device-code-proof Flip", "did the kernels
change?", "VER-6 says NEEDS-LOCAL-PROOF and I cannot build", or "why did the SASS change?" trigger
this skill.
