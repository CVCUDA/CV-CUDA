[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA Bug Fix Guidelines

How to reproduce and patch a bug in CV-CUDA. These guidelines apply to both
humans and AI agents and are derived from the project's recent bug-fix PRs.
For *reviewing* a bug-fix PR, see [REVIEW_PR_GUIDELINES.md](REVIEW_PR_GUIDELINES.md).

The single most important rule:

> **Land a deterministic failing test first, then land the fix on top of it,
> in the same PR.** CI should record a real failure against pre-fix `main`,
> and the same CI should turn green when the fix commit is added.

If you cannot produce a deterministic test that fails before the fix and
passes after, you don't yet understand the bug well enough to fix it.

## 1. Frame the bug

Before touching code, answer in writing:

- **Is there an existing public issue?** Search public GitHub issues for the
  symptom *before* you start. If an issue exists, link it in the PR description
  and reference it in relevant commits when it adds useful traceability. If the
  bug should be public but has no issue, open one first; do not fix silently.
- **What is the observed failure?** Crash, OOB, garbled output, hang,
  flaky test, sanitizer report, user issue. Quote the exact error or symptom.
- **What input triggers it?** Specific shapes, dtypes, formats, parameters,
  matrices, multi-GPU topology, stream configuration. Reduce to the smallest
  reproducer you can.
- **Where is the root cause?** Read the relevant code path top-to-bottom.
  Don't fix the symptom in the wrong layer.
- **What is the blast radius?** Which other operators / call sites share the
  same code path or the same class of mistake? See *Sibling search* below.

For non-trivial bugs, capture this in the PR description or commit body so
reviewers can verify your understanding without re-deriving it.

### Failure locale is not root-cause locale

The test that fails is often *not* the test with the bug. A producer
test corrupts shared GPU state — context, a cached buffer, a stream
that swallows its own error in a `noexcept` destructor — and the next
test in pytest order is the one that reports `cudaErrorIllegalAddress`,
an OOB, or a wrong result. Investigating the reporter directly burns
days. Before forming a hypothesis:

- **Identify the test that ran immediately before the failure.** Pull
  the prior test name and parametrization from the CI log. When the
  symptom is `cudaErrorIllegalAddress` on a generic `Stream.sync()`,
  `cuda.synchronize()`, or teardown call, the producer is almost
  certainly upstream.
- **Check whether the symptom moves across runs.** If the "failing
  test" name changes between CI attempts but the failure mode is the
  same, the failing test is downstream of a context-poisoning bug —
  pytest ordering is the only thing varying. Stop investigating the
  reporter; find the producer.
- **Look for unifying hypotheses across seemingly unrelated symptoms.**
  Multiple distinct-looking failures (e.g. a sanitizer OOB on operator
  X and a `cudaErrorIllegalAddress` on operator Y) often share one
  root cause in shared infrastructure (`ImageBatchVarShape::exportData`,
  `Resource::submitSync`, a Python-binding cache). Build the unifying
  hypothesis before patching either symptom; otherwise you will land
  two partial fixes for one bug and miss the third sibling.
- **Beware errors swallowed in `noexcept` destructors.** A logged-and-
  consumed CUDA error in a `Stream` or `Resource` destructor leaves
  the CUDA context poisoned, so the failure surfaces on the next CUDA
  call from the next test. When tracing a confusing failure, audit
  destructors and `CheckLog`/`try-catch` paths upstream of the reporter
  for swallowed errors.

## 2. Reproduce as a deterministic test

The repro test is the first deliverable, not an afterthought. It must:

- **Live in the existing test tree.** C++ tests under `tests/cvcuda/system/`
  or `tests/nvcv_types/system/`; Python tests under `tests/cvcuda/python/`.
  Match the existing naming and parametrization style of sibling tests.
- **Fail deterministically against pre-fix `main`.** Not "fails sometimes",
  not "fails under load". If the bug is a race, force ordering with a
  host-side sleep, a synthetic kernel delay, or a known-stalled producer
  stream so the failure window is wide and reproducible.
- **Assert the actual symptom, not a proxy.** OOB → run under
  `compute-sanitizer` and assert clean output. Garbled output → assert exact
  expected values. Crash → assert no exception / no `cudaErrorIllegalAddress`.
- **Be small.** One or two parametrized cases that pin the contract. Don't
  bundle a fuzz suite into the regression test.
- **Be self-contained.** No external data, no network, no flaky timing
  assumptions beyond what you explicitly engineered.

If the bug only surfaces on specific hardware (multi-GPU, specific SKU),
note that in the test docstring and gate the test appropriately, but still
make it deterministic on that hardware.

## 3. Get failure evidence from CI on the test-only commit

**Reproduce the bug locally before pushing.** Each CI run on this repo
costs roughly 50–70 minutes of GPU/cluster time; treating CI as an
interactive debugger is expensive and antisocial. Produce the failure
on a developer machine first, or document the exact hardware constraint
that prevents this (multi-GPU topology, specific SKU, sanitizer-only)
and capture local evidence in the commit body.

Once the repro is deterministic locally, push the test commit by itself
and let CI run **once**. The goal is to produce a durable, link-able CI
failure that:

- proves the bug exists on `main` at this SHA, and
- becomes the green CI run that proves the fix works once the fix commit
  is added on top.

Do not loop CI to chase a bug. Pushing instrumentation tweaks, sanitizer
configuration experiments, or speculative fixes hoping the matrix will
catch the bug for you wastes cluster time and obscures the test/fix
pair the PR is meant to demonstrate. If a CI run surfaces evidence local
runs missed, extract that evidence and reproduce locally before the
next push.

**Keep repro experiments on a dedicated branch.** Diagnostic and
instrumentation commits made while reproducing a bug belong on a
separate `fix/<topic>` branch off the parent, not on the user's
existing MR or optimization branch — even when the failure fired on
that branch. The user's branch is the artifact under review;
debugging detritus on it muddies the diff and the history.

Conventions used in recent PRs:

- Branch name: `fix/<short-slug>` (e.g. `fix/minarearect-numpoints-oob`,
  `fix/hqresize-degenerate-roi-fpe`).
- Test commit message: `test: deterministic repro for <bug>` or
  `test: add regression for <bug>`. Body should explain *why* the test fails
  pre-fix, not just what it asserts.
- Explicitly state in the commit body: the test is committed first so CI
  records the failure; the fix lands on top.

If the failure does not surface in CI (e.g. only fires under
`compute-sanitizer`, or only on a specific GPU not in the default matrix),
say so in the commit body and capture local evidence (sanitizer output,
`nvidia-smi`, repro logs) in the PR description.

## 4. Patch the bug

The fix commit goes on top of the test commit on the same branch / same PR.

Guidance gathered from recent fixes:

- **Validate user input at the C API boundary**, not inside a CUDA kernel.
  The codebase has two distinct validation idioms; use whichever matches
  the layer you are editing:
  - **C API / operator layer** (`src/cvcuda/Op*.cpp`,
    `src/cvcuda/priv/Op*.cpp`): `throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "...")`.
    Exceptions are caught and translated to status codes by `ProtectCall`.
  - **Legacy CUDA kernel layer** (`src/cvcuda/priv/legacy/*.cu`):
    `LOG_ERROR("...")` followed by `return ErrorCode::INVALID_PARAMETER`.
  Do **not** use `NVCV_ASSERT` for input validation — `NVCV_ASSERT` aborts
  the process and bypasses `ProtectCall` exception safety. Asserts are for
  internal invariants only.
- **Clamp at the source, not at every consumer.** When a bug originates from
  a single arithmetic primitive (e.g. `__float2int_rd` saturating to
  `INT32_MAX`), clamp inside that primitive so every downstream caller is
  fixed in one place.
- **Beware compiler-level UB.** Out-of-range enum casts, signed integer
  overflow, NaN→int conversions, dereferencing past-end pointers — the
  compiler is licensed to elide bounds checks on UB grounds. Fixing these at
  the *source of UB* is mandatory; defence-in-depth at the kernel is a
  bonus, not a substitute.
- **Multi-GPU safety.** Per-GPU state must use `PerDeviceResource<...>`,
  matching the pattern in `HistogramEq`. Save and restore the current CUDA
  device around teardown loops, including the primary handle, not just the
  aux handles.
- **Stream / event lifecycle.** Wrap CUDA events in RAII guards on exception
  paths. Do not call `cudaStreamWaitEvent` if the matching `cudaEventRecord`
  failed. Honor and populate the CAI v3 / DLPack stream contract on both
  input (`load`) and output (`cuda()` / `__dlpack__`) sides.
- **Python-binding caches.** A custom `Key` must implement `doGetHash`,
  `doIsCompatible`, and `payloadSize` consistently — they jointly define
  cache identity. Strict equality is the safe default unless an op
  explicitly tolerates oversized reuse.
- **Don't add comments that paraphrase the diff.** The fix's *why* belongs
  in the commit message; the code itself should be self-explanatory after
  the change.

### Sibling search

Almost every fix in the recent history had at least one sibling site that
needed the same patch. Before declaring a fix complete, grep for:

- The same operator's tensor and `_var_shape` paths.
- The same operator's `legacy/` and non-legacy implementations.
- Other operators that include the same template / utility (e.g.
  `InterpolationWrap`, `BorderWrap`, `filter_utils`, `ProtectCall`).
- The matching half of any contract: input *and* output, wrap *and*
  unwrap, alloc *and* free, record *and* wait.
- C++ and Python wrappers of the same operator.

If you only fix one half, file a follow-up; do not silently leave the other
half broken.

## 5. Validate before pushing the fix

Required for any non-trivial fix:

- **C++ test suites green:** `cvcuda_test_system`,
  `nvcv_test_cudatools_system`. Note any *pre-existing* unrelated failures
  in the commit body so reviewers don't conflate them with your change.
- **Python suite green:** the relevant `tests/cvcuda/python/` subset, and
  ideally the full pytest suite.
- **Memory bugs:** clean run under `compute-sanitizer`. Quote
  `ERROR SUMMARY: 0 errors` in the commit body.
- **Sample / reporter repro:** if the bug came from an external report or
  GitHub issue, re-run the original reporter's script and confirm it now
  passes.
- **Multi-GPU bugs:** validate on multi-GPU hardware (or document that you
  could not, with a CI link that does).

For bug fixes that touch operators with benchmarks, sanity-check that the
fix doesn't regress benchmark numbers materially (per-operator regression
gating is wired into CI; respect its verdict).

### Tooling caveats that produce false signals

Local validation is only useful if the tooling is honest. The following
caveats have masked real fixes or faked regressions on this repo:

- **Reinstall the Python wheel after every C++ rebuild.** Python tests
  and benchmarks resolve `libcvcuda.so` via the installed wheel's
  RPATH, not the build tree. A freshly-built C++ artifact paired with
  a stale wheel runs the *old* C++ under the new Python — a silent ABI
  mismatch that can hide a fix or fake a regression. Either reinstall
  the wheel or set `LD_LIBRARY_PATH` to the build tree explicitly
  before running pytest or bench.
- **`ncu` does not see bank-conflict or memory-stall latency directly.**
  Concluding "the kernel is unchanged" from `ncu` SOL deltas alone
  misses real perf shifts on memory-bound kernels. Cross-check against
  wall-clock bench runs.
- **DRAM placement perturbs memory-bound kernel timing.** A different
  `cudaMalloc` history (dummy allocations, fragmentation, even alloc
  order across tests) can produce several percent of spread on
  memory-bound operators within a single process. If a bench delta is
  in that range, investigate placement and pinning before claiming the
  fix caused the change — and never relax a parity threshold to make
  the gate pass; the trip is the signal.

## 6. Commit message conventions

If a public GitHub issue describes this bug, link it in the PR description and
add commit trailers when the reference is useful for downstream tooling,
release notes, or `git log` searches. Do not add placeholder issue references.

For the fix commit:

```
fix: <imperative one-line summary, scoped to the operator or subsystem>

<Root cause: 1–3 sentences. Reference CWE-XXX for memory bugs when
appropriate (e.g. CWE-125 OOB read).>

<Fix: what was changed and why this layer is the right place.>

<Validation: which suites passed, sanitizer output, reporter script.>

Fixes https://github.com/CVCUDA/CV-CUDA/issues/<n>    # use when closing a public GitHub issue
```

For the test-only commit, use the same trailers as the fix commit:

```
test: <deterministic repro | regression test> for <bug>

<Why this test fails on pre-fix main and passes on post-fix.>
<Anything subtle about the test setup (e.g. injected delay, specific
shape that triggers the bug).>

Fixes https://github.com/CVCUDA/CV-CUDA/issues/<n>    # use when closing a public GitHub issue
```

Conventional-commit prefixes used in this repo: `fix:`, `test:`, `feat:`, `perf:`,
`refactor:`, `docs:`, `chore:`, `ci:`, `build:`, `revert:`. Use `fix:`
for behavior-changing patches and `test:` for the matching repro.

## 7. Anti-patterns

- **"While I'm here" cleanups in a bug-fix PR.** Bug fixes should be
  reviewable as bug fixes. Refactors, renames, and unrelated tweaks belong
  in a separate PR.
- **Hiding flakes with retries or tolerance bumps.** If a test is flaky,
  diagnose the race; do not add `@retry` or widen tolerances.
- **Skipping the failing CI run.** If a fix lands without a corresponding
  pre-fix red CI on the same branch, future maintainers cannot tell whether
  the fix is necessary.
- **Bypassing pre-commit / signing hooks.** Never use `--no-verify` or
  `--no-gpg-sign` to push a fix. Fix the underlying hook failure.
- **`git commit --amend` / force-push to overwrite review history** once
  the PR is under review. Land follow-ups as new commits; let the merge
  flow squash if needed.
- **Treating a CI flake on rare hardware as a regression in the most recent
  commit.** Investigate the current code path on the failing
  configuration first; the bug is often older than the commit that
  triggered the failing run.
- **Pushing diagnostic commits onto someone else's MR or optimization
  branch.** Repro experiments belong on a fresh `fix/<topic>` branch off
  the parent (see §3).
- **Anecdotal session-specific comments in code.** Phrases like
  "around bench #146", "on a 21 GB box", or "free drops 20→5 GB"
  describe a single debugging session, not the code's invariants —
  they rot the moment the symptom shifts. Explain the *mechanism*
  (what the code guarantees, what would otherwise break) and leave
  session-specific evidence in the commit message or PR description
  where it belongs.

## 8. Debugging tooling

Tools that have repeatedly pinned amorphous failures to specific bugs on
this codebase. Pick by symptom.

- **`compute-sanitizer --tool memcheck <cmd>`** — Catches GPU OOB,
  uninitialized memory, and misaligned access. For context-poisoning
  bugs (see §1) run it twice: against the producer test to surface the
  real OOB, then against the reporter to confirm silence after the fix.
- **`compute-sanitizer --tool racecheck` / `--tool synccheck`** —
  Detect shared-memory races and missing or divergent `__syncthreads`
  inside a kernel. Add `--print-limit 0` when an OOB fires thousands of
  times and you need every offending address.
- **ASAN / UBSAN host build** — Catches host-side OOB, signed overflow,
  NaN→int conversions, and OOB enum casts. Configure with
  `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`.
- **`CUDA_LAUNCH_BLOCKING=1`** — Serializes kernel launches so
  `cudaErrorIllegalAddress` fires on the offending launch instead of
  the downstream sync. Use this whenever §1's "find the producer"
  workflow is in play.
- **Nsight Systems (`nsys profile --trace=cuda,osrt,nvtx`)** — Timeline
  view of stream interactions; the right tool for the legacy-default
  vs. `cudaStreamNonBlocking` race class. Look for user-stream kernel
  launches overlapping an unfinished default-stream H2D.
- **`cuda-gdb` on a `-G`-built operator (`-DCMAKE_CUDA_FLAGS="-G"`)** —
  Step into kernel code, inspect registers and shared memory. Expensive
  to build; reserve for kernels you cannot diagnose any other way.
- **`gdb` on the host process** — For crashes in operator dispatch or
  the Python binding layer. Pybind11 frames are noisy but the trace
  reaches the actual C++ source.
- **`pytest --collect-only -q`** — Print test execution order; use it
  to identify the test that runs immediately before a failure (the
  producer candidate per §1).
- **`pytest -x <full::path::to::test>`** — Run a suspected failing test
  in a fresh process. If it passes in isolation but fails in the full
  suite, the bug is upstream.
- **`ldd` on the cvcuda Python `.so`** — Verify which `libcvcuda.so`
  Python actually resolves. Catches the wheel-vs-build-tree ABI
  mismatch from §5's tooling caveats.
- **`nvidia-smi`** — Confirm the GPU is visible, no other process is
  hogging memory, and no ECC errors are pending.
- **`ncu --set full <cmd>`** — Full kernel profile for perf work. Read
  with §5's caveats: bank-conflict and memory-stall latency are not
  exposed by SOL metrics, so cross-check against wall-clock bench.
- **Wall-clock bench (`bench/run_bench.py --lang python` and the C++ bench
  binaries)** — Authoritative source when `ncu` and reality disagree.
- **`git bisect run <repro-script>`** — Pin a regression to the
  introducing commit when the bug predates the most recent change.
  Write the script to exit 0/non-zero deterministically.

## 9. After merge

- If the fix exposed a class of bug (UB at a primitive, missing per-device
  resource, missing stream contract), open follow-ups for the siblings you
  did not fix.
- If the bug originated from a public GitHub issue, post the merged commit
  / release and close the issue.
