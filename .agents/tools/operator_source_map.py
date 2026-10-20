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
"""Operator kernel-source attribution shared by tools/review_op.py and tools/refactor_op.py:
name-based legacy-kernel ownership plus explicit per-op kernel sources the resolvers' exact
Op<Name>.cu/.cpp candidates and legacy globs cannot find (unrelated names like filter.cu, or
multi-file kernels like OpHQResize2D.cu). Private Op<Name>.hpp class headers stay excluded
for every operator.

The module also carries the inverse direction - `attribute_paths()` maps changed repo paths
back to the operators they can affect, allowing automation to scope a change's benchmark run.
Attribution must stay complete in that direction: an unmapped source
file silently produces an empty benchmark scope, so tools/tests/test_operator_source_map.py
requires every operator-tree file to resolve to an operator or be declared in BROAD_SOURCES
or UNOWNED_SOURCES."""

import functools
import posixpath
import re
from pathlib import Path
from types import MappingProxyType

_REPO = Path(__file__).resolve().parents[2]


def all_op_names():
    """Lowercased operator names derived from the public Op*.h headers."""
    hdr_dir = _REPO / "src/cvcuda/include/cvcuda"
    if not hdr_dir.is_dir():
        return set()
    return {h.stem[2:].lower() for h in hdr_dir.glob("Op*.h") if h.stem != "Operator"}


def legacy_belongs(file_stem, op, all_ops):
    """True if a legacy kernel file belongs to `op`, tolerating the underscores the op name
    drops (op `pillowresize` owns `pillow_resize.cu`, `convertto` owns `convert_to.cu`). The
    longest matching op name wins, so a file is never mis-attributed to an op whose name is
    merely a prefix of the real owner's (e.g. `gaussian` must not claim `gaussian_noise.cu`).
    """
    key = file_stem.replace("_", "").lower()
    if not key.startswith(op):
        return False
    return not any(o != op and len(o) > len(op) and key.startswith(o) for o in all_ops)


# Filename suffixes that decorate an operator's own name rather than naming a different
# operator. Stripping one yields an EXACT operator name, so attribution never has to guess which
# of several candidate prefixes wins.
#
# Order is semantic, not cosmetic: the first suffix that matches is the one stripped, so a suffix
# that ends with another must be tried first. `EraseCopyPolicy.hpp` is the live case - strip
# "CopyPolicy" and it resolves to `erase`; strip "Policy" first and it becomes "erasecopy", which
# is not an operator, so the file silently attributes to nothing. Sorting by descending length
# makes that ordering a property of the data instead of something a future edit can quietly break.
# Kept a tuple rather than a frozenset precisely because a set has no order to preserve.
OPERATOR_NAME_SUFFIXES = tuple(
    sorted(
        [
            "_var_shape",
            "CopyPolicy",
            "Policy",
            "Config",
            "_utils",
            "_util",
            "_common",
            "_planar",
        ],
        key=len,
        reverse=True,
    )
)


SHARED_KERNEL_SOURCES = MappingProxyType(
    {
        # Shared headers are registered so refactor_summary's numstat books their lines against
        # the operators that include them, rather than reporting a pure win it cannot see.
        # Channel-axis validation/dispatch preamble (scalar dtype, channels on the C axis).
        "adjusthue": ("ChannelAxisCommon.cuh",),
        "adjustsaturation": ("ChannelAxisCommon.cuh",),
        "jpegcompressiondistortion": ("ChannelAxisCommon.cuh",),
        # Same-shape validation/dispatch preamble. Deliberately not registered for
        # brightnesscontrast, which only reuses GetCoordForLayout.
        "invert": ("SameShapeCommon.cuh", "PhotometricBound.cuh"),
        "posterize": ("SameShapeCommon.cuh",),
        "solarize": ("SameShapeCommon.cuh", "PhotometricBound.cuh"),
        "adjustsharpness": ("SameShapeCommon.cuh",),
        # Window math, validation and dispatch shared by the two bilateral filters, which are the
        # same operator up to where the range weight is read from.
        "bilateralfilter": ("BilateralFilterCommon.cuh",),
        "jointbilateralfilter": ("BilateralFilterCommon.cuh",),
        "conv2d": ("FilterCommon.cuh",),
        "gaussian": ("FilterCommon.cuh",),
        # The cuOSD rasterizer and its command/context model, shared by the two operators that
        # draw through it.
        "osd": ("OsdRenderer.cu", "OsdRenderer.hpp", "OsdContext.hpp"),
        "bndbox": ("OsdRenderer.cu", "OsdRenderer.hpp", "OsdContext.hpp"),
        "warpaffine": ("WarpCommon.cuh",),
        "warpperspective": ("WarpCommon.cuh",),
        "hqresize": (
            "OpHQResize2D.cu",
            "OpHQResize3D.cu",
            "OpHQResizeBatchWrap.cuh",
            "OpHQResizeDispatch.hpp",
            "OpHQResizeFilter.cuh",
            "OpHQResizeKernel.cuh",
            "OpHQResizePlanar.cuh",
        ),
        "stack": ("OpStackKernels.cu", "OpStackKernels.hpp"),
        "split": ("ChannelCopy.cu", "ChannelCopy.hpp"),
        "merge": ("ChannelCopy.cu", "ChannelCopy.hpp"),
        "averageblur": ("FilterCommon.cuh",),
        "laplacian": ("FilterCommon.cuh",),
    }
)

# Translation units backing more than one operator, which no name rule reaches and the include
# graph cannot answer for (a .cu maps to the operators its kernels implement, not to who
# includes it). Shared HEADERS are deliberately absent: those are derived from the include
# graph at query time, because a hand-maintained set goes stale in both directions.
# Kept separate from SHARED_KERNEL_SOURCES because that table also drives
# review_op / refactor_op / tu_cost LOC accounting, where a shared helper is deliberately not
# booked against its consumers. Here the question is only "whose benchmarks can this change
# move", so the consumer set is what matters. A trailing "/" makes an entry a directory prefix.
# Each set was derived from the including translation units, not guessed.
_HQRESIZE_KERNEL_FILES = (
    "OpHQResize2D.cu",
    "OpHQResize3D.cu",
    "OpHQResizeBatchWrap.cuh",
    "OpHQResizeDispatch.hpp",
    "OpHQResizeFilter.cuh",
    "OpHQResizeKernel.cuh",
    "OpHQResizePlanar.cuh",
)

MULTI_OP_SOURCES = MappingProxyType(
    {
        "bench/cpp/ops/BenchChannelCopy.hpp": ("split", "merge"),
        "src/cvcuda/priv/ChannelCopy.cu": ("split", "merge"),
        # Bodies that name their operator only obliquely, so no suffix rule reaches them.
        "src/cvcuda/priv/OpEraseRegion.cu": ("erase",),
        "src/cvcuda/priv/OpStackKernels.cu": ("stack",),
        "src/cvcuda/priv/OpStackKernels.hpp": ("stack",),
        "src/cvcuda/priv/OpHQResize2D.cu": ("hqresize",),
        "src/cvcuda/priv/OpHQResize3D.cu": ("hqresize",),
        # One kernel split across headers that only include each other, so the include-graph
        # walk never reaches a translation unit it can attribute.
        **{f"src/cvcuda/priv/{name}": ("hqresize",) for name in _HQRESIZE_KERNEL_FILES},
        # Composition: these operators run another operator's implementation on their measured
        # path, so a change there moves their timings too. Re-derive with:
        #   grep -rn "legacy::OSD\\|std::make_unique<Reformat>" src/cvcuda/priv
        "src/cvcuda/priv/OpReformat.cu": ("bndbox", "histogram", "osd", "reformat"),
        "src/cvcuda/priv/OsdRenderer.cu": ("bndbox", "osd"),
        # Reached through OpBndBox.cu / OpOSD.cu -> OsdRenderer.hpp.
        "src/cvcuda/priv/OsdContext.hpp": ("bndbox", "osd"),
        "src/cvcuda/priv/textbackend/": ("bndbox", "osd"),
    }
)

# Directory trees every operator compiles against: the shared CUDA device utilities, the nvcv
# core types, and the benchmark harness itself. A change anywhere under these can move every
# operator's measured time, and naming individual files would rot immediately.
BROAD_PREFIXES = (
    "src/cvcuda/include/cvcuda/cuda_tools/",
    "src/nvcv/",
    "bench/cpp/",
    "bench/python/",
    "bench/config/",
    "bench/_internal/",
)

# Core infrastructure every operator compiles against. A change here resolves to the whole
# operator set, which is deliberately expensive: it runs the advanced sweep for all of them.
BROAD_SOURCES = frozenset(
    {
        "src/cvcuda/priv/Assert.h",
        "src/cvcuda/priv/CudaDeviceUtils.hpp",
        "src/cvcuda/priv/IOperator.cpp",
        "src/cvcuda/priv/IOperator.hpp",
        "src/cvcuda/priv/Nvtx.hpp",
        "src/cvcuda/priv/PerDeviceResource.hpp",
        "src/cvcuda/priv/PlanarTensorView.hpp",
        "src/cvcuda/priv/SafeSize.hpp",
        "src/cvcuda/priv/SymbolVersioning.hpp",
        "src/cvcuda/priv/Types.hpp",
        "src/cvcuda/priv/Version.hpp",
        "src/cvcuda/priv/WorkspaceAllocator.hpp",
        "src/cvcuda/priv/WorkspaceEstimator.hpp",
        "src/cvcuda/priv/WorkspaceUtil.hpp",
        "src/cvcuda/include/cvcuda/Workspace.hpp",
        "bench/run_bench.py",
        "bench/compare_to_baseline.py",
    }
)

# Files in the operator tree that intentionally attribute to no operator: build wiring, the
# operator-agnostic base class, and scheduling gates. Listing them keeps the completeness test
# meaningful - a new unattributable file has to be classified here on purpose rather than
# defaulting to silence. Entries here win over BROAD_SOURCES/BROAD_PREFIXES.
UNOWNED_SOURCES = frozenset(
    {
        "src/cvcuda/Operator.cpp",
        "src/cvcuda/include/cvcuda/Operator.h",
        # Declarations only: ExportOp<Name> prototypes, changed whenever an operator is added.
        "python/mod_cvcuda/operators/Operators.hpp",
        # Decides whether a node may run benchmarks at all, never what they measure. Sweeping
        # every operator would add no signal: a gate that wrongly admits a node fails the leg
        # whatever it measures, and one that wrongly rejects runs nothing. Its neighbours stay
        # broad for two reasons this file meets neither of - run_bench.py imports them, and they
        # consume per-operator data. This one is a standalone script whose only config is the
        # operator-agnostic sku_map.json.
        "bench/_internal/check_bench_sku.py",
    }
)


_NON_OPERATOR_CHARS = re.compile(r"[^a-z0-9]")


def _norm(text):
    """Flatten a filename body to the operator-name alphabet (lowercase alphanumerics)."""
    return _NON_OPERATOR_CHARS.sub("", text.lower())


def _exact_owner(body, all_ops):
    """The operator a filename body names exactly, after stripping one declared suffix.

    `all_ops` may be any container of operator names - callers pass both a set and a sorted
    list - so this uses membership tests rather than set algebra.

    The operator set is fixed and known, so this is an exact set membership test rather than a
    prefix search: `gaussian_noise_util` strips to `gaussiannoise` and matches that operator
    outright, instead of `gaussian` and `gaussiannoise` both being candidate prefixes that some
    tie-break rule has to choose between. A file whose body is not an operator name after one
    suffix strip belongs in an explicit table, and the completeness test says so.
    """
    key = _norm(body)
    if key in all_ops:
        return {key}
    for suffix in OPERATOR_NAME_SUFFIXES:
        if body.endswith(suffix):
            stripped = _norm(body[: -len(suffix)])
            return {stripped} if stripped in all_ops else set()
    return set()


# Directory-prefix entries are rare (one today), so pull them out once rather than re-scanning
# the whole table on every path that misses the exact lookup.
_MULTI_OP_PREFIXES = tuple(
    (key, value) for key, value in MULTI_OP_SOURCES.items() if key.endswith("/")
)


def _multi_op_lookup(path):
    """Operators registered for `path` in MULTI_OP_SOURCES, honoring directory prefixes."""
    exact = MULTI_OP_SOURCES.get(path)
    if exact is not None:
        return set(exact)
    for key, operators in _MULTI_OP_PREFIXES:
        if path.startswith(key):
            return set(operators)
    return set()


# Shared headers under the operator roots: derive consumers from the #include graph at query
# time rather than declaring them. A hand-maintained set is wrong in both directions - it went
# stale twice during review (SameShapeCommon.cuh, VarShapeUtils.hpp), and pinning it with an
# equality test then broke unrelated merge requests whenever main changed who includes what.
_INCLUDE_ROOTS = ("src/cvcuda", "python/mod_cvcuda")
_QUOTED_INCLUDE = re.compile(r'#include\s+"([^"]+)"')


@functools.lru_cache(maxsize=1)
def _include_graph():
    """Repo-relative file -> the repo-relative paths it includes, resolved where possible."""
    graph = {}
    for root in _INCLUDE_ROOTS:
        base = _REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in (".cu", ".cuh", ".hpp", ".cpp", ".h"):
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            resolved = set()
            for include in _QUOTED_INCLUDE.findall(text):
                # Quoted includes resolve against the including file's directory; 74 header
                # basenames collide across these trees, so never key on the basename alone.
                candidate = (path.parent / include).resolve()
                if candidate.is_file():
                    try:
                        resolved.add(str(candidate.relative_to(_REPO)))
                    except ValueError:
                        pass
            graph[str(path.relative_to(_REPO))] = resolved
    return graph


def _transitive_includers(header):
    """Files that reach `header`, stopping at BROAD_SOURCES so one umbrella cannot widen all."""
    graph = _include_graph()
    seen, frontier = set(), {header}
    while frontier:
        nxt = set()
        for path, includes in graph.items():
            if path in seen or path in BROAD_SOURCES:
                continue
            if includes & frontier:
                seen.add(path)
                nxt.add(path)
        frontier = nxt
    return seen


def _derived_header_owners(path, all_ops):
    """Operators whose measured paths compile `path`, from the include graph.

    Consumers are resolved with the derivation disabled. _transitive_includers already returns
    the full closure, so recursing would add nothing - and it would not terminate: a header that
    is reachable from itself (stb_truetype.h is today) would bounce between these two functions
    until the stack blew.
    """
    owners = set()
    for consumer in _transitive_includers(path):
        owners |= operators_for_path(consumer, all_ops, derive_headers=False)
    return owners


def normalize_repo_path(path):
    """Repo-relative POSIX form of a path as git reports it."""
    text = str(path).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def is_broad_source(path):
    """Whether a change to `path` can affect every operator."""
    path = normalize_repo_path(path)
    if path in UNOWNED_SOURCES:
        return False
    return path in BROAD_SOURCES or path.startswith(BROAD_PREFIXES)


def operators_for_path(path, all_ops=None, derive_headers=True):
    """Operators whose benchmarks a change to a single repo path can move.

    Returns the complete operator set for a BROAD_SOURCES file and an empty set for a path
    outside the operator tree (tests, docs, samples, CI - none of which change measured
    performance).
    """
    ops = all_op_names() if all_ops is None else all_ops
    path = normalize_repo_path(path)
    if not path:
        return set()
    if path in UNOWNED_SOURCES:
        return set()

    multi = _multi_op_lookup(path)
    if multi:
        return multi

    stem = posixpath.splitext(posixpath.basename(path))[0]

    # Per-operator artifacts first: these live inside the broad bench trees, and a file that
    # names an operator belongs to that operator whatever tree it sits in.
    if path.startswith("bench/config/operators/") and path.endswith(".json"):
        return _exact_owner(stem, ops)
    if path.startswith("bench/python/ops/") and stem.startswith("bench_"):
        return _exact_owner(stem.removeprefix("bench_"), ops)
    if path.startswith("bench/cpp/ops/") and stem.startswith("Bench"):
        return _exact_owner(stem.removeprefix("Bench"), ops)

    if path in BROAD_SOURCES or path.startswith(BROAD_PREFIXES):
        return set(ops)

    # Op<Name>-prefixed sources: the public API, the C shim, the private impl, the binding.
    if (
        path.startswith("src/cvcuda/include/cvcuda/Op")
        or path.startswith("python/mod_cvcuda/operators/Op")
        or (path.startswith("src/cvcuda/Op") and "/priv/" not in path)
        or (path.startswith("src/cvcuda/priv/Op") and "/legacy/" not in path)
    ):
        return _exact_owner(stem.removeprefix("Op"), ops)

    # Remaining private sources, including the snake_case legacy tree.
    if path.startswith("src/cvcuda/priv/"):
        named = _exact_owner(stem, ops)
        if named:
            return named

    # A shared header nothing else claimed: ask the include graph who compiles it.
    if (
        derive_headers
        and path.endswith((".cuh", ".hpp", ".h"))
        and path.startswith(_INCLUDE_ROOTS)
    ):
        return _derived_header_owners(path, ops)

    if path.startswith("src/cvcuda/priv/"):
        return set()

    return set()


def attribute_paths(paths, all_ops=None):
    """Per-path attribution, for reporting which change pulled in which operators."""
    ops = all_op_names() if all_ops is None else all_ops
    normalized = [path for path in map(normalize_repo_path, paths) if path]
    return {path: operators_for_path(path, ops) for path in normalized}
