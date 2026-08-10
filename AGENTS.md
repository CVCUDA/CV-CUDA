[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA Agent Guidelines

These instructions are for AI coding agents working in the CV-CUDA repository.
They define shared agent behavior and point to the project documents that own
the detailed workflows. This file (`AGENTS.md`) is the single canonical entry
point, read natively by Codex and Cursor; `CLAUDE.md` is a symlink to it for
Claude Code.

Skills are defined once under `.agents/skills/<name>/SKILL.md` (the tool-agnostic
standard, read directly by Codex and Cursor); `.claude/skills` is a symlink to
`.agents/skills` for Claude Code. Edit the canonical files under `.agents/skills`
only — never the symlinked copies.

Topic guidance documents live under `.agents/guidance/<name>.md`; see the
"Authoritative docs" section below. This file and `README.md` remain at the
repository root as the canonical entry points.

Shared helper modules imported by skill scripts live under `.agents/tools/`.

## Repository map

| Path | Purpose |
|------|---------|
| `src/` | C++ core library, C API, private operator implementations, and nvcv types |
| `python/` | pybind11-based Python bindings and wheel packaging |
| `tests/` | C++ googletest and Python pytest test suites |
| `bench/` | C++ and Python nvbench benchmarks; shared config in `bench/config/` |
| `samples/` | Example applications and interoperability samples |
| `docs/` | Sphinx and Doxygen documentation sources |
| `docker/` | Multi-arch builder and development Docker images |
| `ci/` | CI tooling and pipeline configuration |
| `lint/` | Pre-commit hooks and repository checks |

## Authoritative docs

- Project overview and compatibility: `README.md`
- Installation and source builds: `docs/sphinx/installation.rst`
- Tests: `tests/README.md`
- Benchmarks: `bench/README.md`
- Samples: `samples/README.md`
- New operators: `.agents/guidance/MAKE_OP_GUIDELINES.md` (narrative how-to:
  `docs/sphinx/advanced/make_operator.rst`); `/make-op`, `/make-op-scaffold`,
  `/make-op-verify` skills
- Per-operator coverage review (support/test/bench/docs): `.agents/guidance/REVIEW_OP_GUIDELINES.md`
  (`/review-op` skill)
- Bug fixes: `.agents/guidance/BUGFIX_GUIDELINES.md`
- Operator optimization: `.agents/guidance/OPTIMIZATION_GUIDELINES.md`
  (`/optimize-op` skill)
- Per-operator refactoring / redundancy reduction: `.agents/guidance/REFACTOR_OP_GUIDELINES.md`
  (`/refactor-op` skill)

Use those documents as the source of truth. Do not duplicate their checklists
in tool-specific prompts unless a command needs a short output template.

## Working tree rules

- Check `git status --short --branch` before editing and again before
  summarizing work.
- Treat untracked or modified files you did not create as user-owned. Do not
  remove, overwrite, reset, or check them out unless explicitly asked.
- Keep edits scoped to the requested task. Do not mix cleanup, formatting, or
  unrelated refactors into functional changes.
- Prefer `rg` and `rg --files` for repository searches.
- Follow existing local patterns before adding helpers or abstractions.
- Use structured parsers or existing generators when the repository provides
  them.

## Repository invariants

- New source, script, and documentation files need NVIDIA Apache 2.0 SPDX
  headers. Files created in 2026 should use `Copyright (c) 2026`, not a range.
- Requirements `.txt` files under `tests/`, `bench/`, `samples/`, `docker/`,
  and `docs/` are generally generated from `.template` files and `versions.env`.
  Do not hand-edit generated requirements. Update the matching source template
  or `versions.env`, then run `bash generate_requirements.sh`.
- Changes touching CUDA 12 paths usually need matching CUDA 13 coverage, and
  vice versa. Check requirements, Docker, CI, and docs for paired updates.
- Public C, C++, and Python API changes need matching docs, tests, and review
  of ABI/API compatibility expectations.
- Operators that consume images support both interleaved (`NHWC`/`HWC`) and planar
  (`NCHW`/`CHW`) layouts by default. If image layouts do not apply, declare
  `Planar image layouts: Not applicable` with a `Reason` in the operator's public
  C-header Limitations contract.
- Comments should explain why something non-obvious is necessary. Do not add
  comments that restate the code.

## Validation ladder

Use the narrowest validation that proves the change, then state exactly what
was and was not run.

Common checks:

```bash
bash generate_requirements.sh --check
pre-commit run --files <changed files>
cmake --preset dev
cmake --build --preset dev
bash build.sh release build-rel -DBUILD_TESTS=1
build-rel/bin/run_tests.sh
build-rel/bin/run_tests.sh cvcuda,cpp
build-rel/bin/run_tests.sh cvcuda,python
```

For targeted C++ work, prefer building and running the relevant executable
under `build-rel/bin/`, such as `cvcuda_test_system` or
`nvcv_test_cudatools_system`. For Python work, run the relevant pytest file or
the generated `cvcuda_test_python` wrapper when available.

GPU, CUDA toolkit, Docker, profiler, and `compute-sanitizer` checks may not be
available in every agent environment. If a required check cannot be run, say so
plainly and identify the missing prerequisite.

## Task policy

- Reviews: act as a reviewer, lead with findings ordered by severity, and
  include file and line references.
- Per-operator coverage review: follow `.agents/guidance/REVIEW_OP_GUIDELINES.md`; run the
  deterministic checker `tools/review_op.py <Operator> [--domain support|test|bench|docs]`
  (or the `/review-op` skill), resolve every `MANUAL`, and close `GAP`s.
- Bug fixes: follow `.agents/guidance/BUGFIX_GUIDELINES.md`; add the deterministic failing
  regression test before the fix and search sibling code paths.
- Optimization: follow `.agents/guidance/OPTIMIZATION_GUIDELINES.md`; establish benchmark
  coverage, profile before coding, and preserve correctness. Gate readiness and the
  definition-of-done with `tools/optimize_op.py <Op> --phase preflight|evidence`
  (or the `/optimize-op` skill); close every `GAP` before the campaign is done.
- Per-operator refactoring: follow `.agents/guidance/REFACTOR_OP_GUIDELINES.md`;
  use `tools/refactor_op.py <Operator>` to assess redundancy opportunities and
  `tools/refactor_op.py <Operator> --phase verify` to gate parity before review.
- New operators: follow `.agents/guidance/MAKE_OP_GUIDELINES.md` (narrative how-to in
  `docs/sphinx/advanced/make_operator.rst`). Propose + get user approval of the spec (semantics +
  cited reference oracle + support matrix), scaffold with `tools/mkop/mkop.sh`, then gate with
  `tools/make_op.py <Op> --phase scaffold|done` (or the `/make-op` family). The done-gate is the
  deterministic regression checklist — independent CPU gold reference, bit-exact coverage of every
  declared variant, required equivalent-layout parity, complement negatives, docs + relnote,
  benched dtypes, tests that run & pass; it reuses `/review-op` and `/optimize-op` preflight and
  hands off to `/optimize-op`. Use `/make-op-scaffold [--bare]` to delegate the implementation.
