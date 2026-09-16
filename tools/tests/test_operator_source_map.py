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
"""Completeness and correctness of changed-path -> operator attribution.

Automation turns a change into the operator list its benchmark job runs. An unattributed
source file does not fail loudly there - it silently shrinks the benchmark scope - so the
completeness test below makes a new file an explicit decision instead of a silent coverage hole.
"""

import functools
import json
import re
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".agents" / "tools"))

from operator_source_map import (  # noqa: E402
    BROAD_PREFIXES,
    BROAD_SOURCES,
    legacy_belongs,
    MULTI_OP_SOURCES,
    OPERATOR_NAME_SUFFIXES,
    SHARED_KERNEL_SOURCES,
    UNOWNED_SOURCES,
    all_op_names,
    attribute_paths,
    is_broad_source,
    operators_for_path,
)

# Paths that are build wiring rather than operator sources.
_IGNORED_NAMES = {"CMakeLists.txt"}

# Every tracked file that must resolve to an operator or be explicitly classified.
_TRACKED_GLOBS = (
    "src/cvcuda/priv",
    "src/cvcuda/include/cvcuda/cuda_tools",
    "bench/cpp/ops",
    "bench/python/ops",
    "bench/config/operators",
    "python/mod_cvcuda/operators",
    "src/cvcuda/Op*.cpp",
    "src/cvcuda/Operator.cpp",
    "src/cvcuda/include/cvcuda/Op*.h",
    "src/cvcuda/include/cvcuda/Op*.hpp",
    "src/cvcuda/include/cvcuda/Operator*.h",
)


@functools.lru_cache(maxsize=1)
def _tracked_files():
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", *_TRACKED_GLOBS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return tuple(
        sorted(
            path
            for path in set(out)
            if Path(path).name not in _IGNORED_NAMES and not path.endswith(".md")
        )
    )


@pytest.fixture(scope="module")
def ops():
    return all_op_names()


def test_every_operator_tree_file_is_attributed(ops):
    """A new source file must map to an operator or be classified broad/unowned."""
    unattributed = [
        path
        for path in _tracked_files()
        if not operators_for_path(path, ops) and path not in UNOWNED_SOURCES
    ]
    assert unattributed == [], (
        "These files attribute to no operator. Add a rule, register them in "
        "MULTI_OP_SOURCES/BROAD_SOURCES, or declare them in UNOWNED_SOURCES:\n  "
        + "\n  ".join(unattributed)
    )


def test_attribution_never_invents_operator_names(ops):
    """Emitted names must be usable as run_bench.py --operator values."""
    manifest = json.loads((REPO / "bench/config/bench_params.json").read_text())
    known = set(manifest["operators"])
    assert ops == known, "public Op*.h headers and the bench manifest disagree"
    emitted = set().union(*attribute_paths(_tracked_files(), ops).values(), set())
    assert emitted <= known


def test_declared_tables_reference_real_operators_and_files(ops):
    """The hand-maintained tables must not rot against the tree."""
    for path, entries in MULTI_OP_SOURCES.items():
        assert set(entries) <= ops, f"{path} names unknown operators"
        target = REPO / path.rstrip("/")
        assert target.exists(), f"MULTI_OP_SOURCES entry no longer exists: {path}"
    for path in BROAD_SOURCES | UNOWNED_SOURCES:
        assert (REPO / path).exists(), f"declared path no longer exists: {path}"


@pytest.mark.parametrize(
    "path",
    [
        "src/cvcuda/priv/legacy/CvCudaUtils.cuh",
        # Whole trees every operator compiles against. These previously attributed to nothing,
        # so a change to the shared device utilities produced an empty scope and a green no-op.
        "src/cvcuda/include/cvcuda/cuda_tools/TypeTraits.hpp",
        "src/nvcv/Tensor.cpp",
        "bench/cpp/CppBenchUtils.hpp",
        "bench/run_bench.py",
    ],
)
def test_broad_source_expands_to_every_operator(path, ops):
    assert is_broad_source(path)
    assert operators_for_path(path, ops) == ops


def test_legacy_umbrella_header_attributes_to_no_operator(ops):
    """CvCudaLegacy.h is declarations only, so it must not drive the scope on its own.

    Every legacy-layer removal deletes one operator's `class <Op>` block from this header. While
    it was broad, that one-operator diff resolved to all 61 operators and ran the advanced sweep
    for every one of them. The operator's own translation unit is what attributes.
    """
    path = "src/cvcuda/priv/legacy/CvCudaLegacy.h"
    assert not is_broad_source(path)
    assert operators_for_path(path, ops) == set()
    # The shared legacy utilities stay broad: they carry inlined device code, not declarations.
    assert operators_for_path("src/cvcuda/priv/legacy/CvCudaUtils.cuh", ops) == ops


def test_broad_prefixes_do_not_swallow_per_operator_artifacts(ops):
    """The per-operator bench files live inside the broad bench trees and must win."""
    assert any(
        "bench/cpp/ops/BenchMedianBlur.cpp".startswith(prefix)
        for prefix in BROAD_PREFIXES
    )
    assert operators_for_path("bench/cpp/ops/BenchMedianBlur.cpp", ops) == {
        "medianblur"
    }
    assert operators_for_path("bench/config/operators/gaussian.json", ops) == {
        "gaussian"
    }


def test_paths_outside_the_operator_tree_attribute_to_nothing(ops):
    """Tests, docs, samples and CI do not move measured performance."""
    outside = [
        "docs/sphinx/operator_list.rst",
        "tests/cvcuda/system/TestOpGaussian.cpp",
        "tests/cvcuda/python/test_opgaussian.py",
        "samples/common/python/nvcodec.py",
        "ci/generate_pipeline.py",
    ]
    assert set().union(*attribute_paths(outside, ops).values(), set()) == set()


@pytest.mark.parametrize(
    "path,expected",
    [
        # The clean case: the legacy kernel names its own operator.
        ("src/cvcuda/priv/legacy/median_blur.cu", {"medianblur"}),
        ("src/cvcuda/priv/legacy/median_blur_var_shape.cu", {"medianblur"}),
        # Names that a prefix search would have contested resolve exactly instead.
        ("src/cvcuda/priv/legacy/gaussian_noise.cu", {"gaussiannoise"}),
        ("src/cvcuda/priv/legacy/gaussian_noise_util.cuh", {"gaussiannoise"}),
        (
            "src/cvcuda/priv/OpResizeCropConvertReformat.cu",
            {"resizecropconvertreformat"},
        ),
        ("src/cvcuda/priv/legacy/adaptive_threshold.cu", {"adaptivethreshold"}),
        ("src/cvcuda/priv/legacy/pad_and_stack.cu", {"padandstack"}),
        # Suffixed helpers still resolve to their operator.
        ("src/cvcuda/priv/OpHQResizeKernel.cuh", {"hqresize"}),
        ("src/cvcuda/priv/OpStackKernels.cu", {"stack"}),
        ("src/cvcuda/priv/OpEraseRegion.cu", {"erase"}),
        ("src/cvcuda/priv/InvertPolicy.hpp", {"invert"}),
        # Shared kernels fan out to every consumer.
        (
            "src/cvcuda/priv/legacy/filter.cu",
            {"gaussian", "averageblur", "laplacian"},
        ),
        (
            "src/cvcuda/priv/legacy/warp.cu",
            {"warpaffine", "warpperspective"},
        ),
        (
            "src/cvcuda/priv/legacy/warp_cubic.cuh",
            {"rotate", "warpaffine", "warpperspective"},
        ),
        ("src/cvcuda/priv/legacy/calc_hist.cu", {"histogram"}),
        ("src/cvcuda/priv/legacy/reduce_kernel_utils.cuh", {"inpaint"}),
        (
            "src/cvcuda/priv/AdjustColorCommon.cuh",
            {"adjustcontrast", "adjustsharpness"},
        ),
        (
            "src/cvcuda/priv/legacy/textbackend/backend.cpp",
            {"bndbox", "boxblur", "osd"},
        ),
        # The rest of an operator's surface.
        ("src/cvcuda/Op MedianBlur.cpp".replace(" ", ""), {"medianblur"}),
        ("src/cvcuda/include/cvcuda/OpMedianBlur.h", {"medianblur"}),
        ("python/mod_cvcuda/operators/OpMedianBlur.cpp", {"medianblur"}),
        ("bench/cpp/ops/BenchMedianBlur.cpp", {"medianblur"}),
        ("bench/python/ops/bench_medianblur.py", {"medianblur"}),
        ("bench/config/operators/medianblur.json", {"medianblur"}),
        ("bench/cpp/ops/BenchGaussian.cpp", {"gaussian"}),
        ("bench/cpp/ops/BenchGaussianNoise.cpp", {"gaussiannoise"}),
    ],
)
def test_known_attributions(path, expected, ops):
    assert operators_for_path(path, ops) == expected


def test_attribution_is_exact_rather_than_a_prefix_search(ops):
    """A body that is not an operator name after one declared suffix resolves to nothing.

    The operator set is fixed, so an unrecognized file must fail the completeness test and be
    declared, never be attached to whichever operator happens to be a prefix of its name.
    """
    assert operators_for_path("src/cvcuda/priv/OpGaussianSomethingNew.cu", ops) == set()
    assert (
        operators_for_path("src/cvcuda/priv/legacy/resize_experimental.cu", ops)
        == set()
    )
    # ... and the near-miss that a longest-prefix search would have silently absorbed.
    assert operators_for_path("src/cvcuda/priv/legacy/flip_or_copy.cu", ops) == set()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/cvcuda/priv/legacy/median_blur_var_shape.cu", {"medianblur"}),
        ("src/cvcuda/priv/legacy/gaussian_noise_util.cu", {"gaussiannoise"}),
        ("src/cvcuda/priv/legacy/gaussian_noise_var_shape.cu", {"gaussiannoise"}),
        ("src/cvcuda/priv/legacy/threshold_util.cuh", {"threshold"}),
        ("src/cvcuda/priv/legacy/gamma_contrast_common.cuh", {"gammacontrast"}),
        ("src/cvcuda/priv/legacy/normalize_planar.cuh", {"normalize"}),
        ("src/cvcuda/priv/legacy/inpaint_utils.cu", {"inpaint"}),
        ("src/cvcuda/priv/InvertPolicy.hpp", {"invert"}),
        ("src/cvcuda/priv/legacy/EraseCopyPolicy.hpp", {"erase"}),
        ("src/cvcuda/priv/legacy/ReformatCopyPolicy.hpp", {"reformat"}),
        ("src/cvcuda/priv/OpHQResizePolicy.hpp", {"hqresize"}),
    ],
)
def test_declared_suffixes_strip_to_an_exact_operator_name(path, expected, ops):
    assert operators_for_path(path, ops) == expected


def test_the_motivating_merge_request_scope(ops):
    """An MR touching Gaussian and MedianBlur kernels benches both, plus filter.cu's peers."""
    changed = [
        "src/cvcuda/priv/OpGaussian.cpp",
        "src/cvcuda/priv/legacy/filter.cu",
        "src/cvcuda/priv/legacy/median_blur.cu",
        "docs/sphinx/operator_list.rst",
    ]
    assert set().union(*attribute_paths(changed, ops).values(), set()) == {
        "gaussian",
        "averageblur",
        "laplacian",
        "medianblur",
    }
    per_path = attribute_paths(changed, ops)
    assert per_path["docs/sphinx/operator_list.rst"] == set()
    assert per_path["src/cvcuda/priv/legacy/median_blur.cu"] == {"medianblur"}


def test_path_normalization(ops):
    for variant in (
        "./src/cvcuda/priv/legacy/median_blur.cu",
        "src\\cvcuda\\priv\\legacy\\median_blur.cu",
        "  src/cvcuda/priv/legacy/median_blur.cu  ",
    ):
        assert operators_for_path(variant, ops) == {"medianblur"}
    assert set().union(*attribute_paths(["", "   "], ops).values(), set()) == set()


def test_suffix_order_is_longest_first_by_construction():
    """A suffix ending in another must be tried first, or the shorter one wins and truncates.

    `EraseCopyPolicy.hpp` is the live case: strip "CopyPolicy" -> `erase`; strip "Policy" first
    -> "erasecopy", which is not an operator, so the file would attribute to nothing at all.
    """
    lengths = [len(suffix) for suffix in OPERATOR_NAME_SUFFIXES]
    assert lengths == sorted(lengths, reverse=True)
    for suffix in OPERATOR_NAME_SUFFIXES:
        for other in OPERATOR_NAME_SUFFIXES:
            if other != suffix and suffix.endswith(other):
                assert OPERATOR_NAME_SUFFIXES.index(
                    suffix
                ) < OPERATOR_NAME_SUFFIXES.index(
                    other
                ), f"{suffix!r} must be tried before {other!r}"


def test_declared_tables_are_immutable():
    """These tables are shared across five tools; a stray write would corrupt all of them."""
    assert isinstance(SHARED_KERNEL_SOURCES, MappingProxyType)
    assert isinstance(MULTI_OP_SOURCES, MappingProxyType)
    assert isinstance(BROAD_SOURCES, frozenset)
    assert isinstance(UNOWNED_SOURCES, frozenset)
    assert isinstance(OPERATOR_NAME_SUFFIXES, tuple)
    # Values too - a read-only mapping over mutable lists is only half immutable.
    assert all(isinstance(v, tuple) for v in SHARED_KERNEL_SOURCES.values())
    assert all(isinstance(v, tuple) for v in MULTI_OP_SOURCES.values())
    for table in (SHARED_KERNEL_SOURCES, MULTI_OP_SOURCES):
        with pytest.raises(TypeError):
            table["scratch"] = ("gaussian",)


def test_the_two_attribution_rules_do_not_contradict_each_other(ops):
    """`legacy_belongs` (LOC accounting) must never claim an operator benchmarks would miss.

    The module deliberately keeps two name rules: `legacy_belongs` is longest-prefix-wins and
    feeds review_op/refactor_op/tu_cost; attribution is exact-after-one-suffix. Nothing forced
    them to agree, so a future edit to either could have the same file name different operators
    in a review than in a benchmark scope, silently.
    """
    disagreements = []
    for path in _tracked_files():
        if not path.startswith("src/cvcuda/priv/"):
            continue
        stem = Path(path).stem
        claimed = {op for op in ops if legacy_belongs(stem, op, ops)}
        attributed = operators_for_path(path, ops)
        if not claimed <= attributed:
            disagreements.append((path, sorted(claimed), sorted(attributed)))
    assert disagreements == []


# CvCudaOSD.hpp's only includer is legacy/CvCudaLegacy.h, which attributes to no operator, so
# the closure stops there and the graph cannot reach its consumers. Its two operators come from
# what the OSD kernels draw, not from who includes the header.
_UNDERIVABLE_HEADERS = frozenset({"src/cvcuda/priv/legacy/CvCudaOSD.hpp"})


def _include_graph():
    """file -> set of basenames it #includes, over the operator source roots."""
    graph = {}
    for root in (REPO / "src/cvcuda", REPO / "python/mod_cvcuda"):
        for path in root.rglob("*"):
            if path.suffix not in (".cu", ".cuh", ".hpp", ".cpp", ".h"):
                continue
            text = path.read_text(errors="replace")
            rel = path.relative_to(REPO)
            resolved = set()
            for inc in re.findall(r'#include\s+"([^"]+)"', text):
                # Quoted includes resolve against the including file's directory first.
                candidate = (path.parent / inc).resolve()
                if candidate.is_file():
                    resolved.add(str(candidate.relative_to(REPO)))
                else:
                    resolved.add(Path(inc).name)
            graph[str(rel)] = resolved
    return graph


def _transitive_includers(header, graph):
    """Every file that reaches `header`, stopping at BROAD_SOURCES and UNOWNED_SOURCES.

    A broad header is already attributed to every operator, so traversing through one would
    expand any set that reaches it to the full operator list. An unowned header attributes to
    no operator on purpose - CvCudaLegacy.h is declarations only - so traversing through one
    would attribute its consumers to a header that deliberately owns nothing.
    """
    stop = BROAD_SOURCES | UNOWNED_SOURCES
    seen, names, changed = set(), {header, Path(header).name}, True
    while changed:
        changed = False
        for path, includes in graph.items():
            if path in seen or path in stop:
                continue
            if includes & names:
                seen.add(path)
                names.update({path, Path(path).name})
                changed = True
    return seen


def test_shared_header_sets_match_the_include_graph(ops):
    """Derive each shared header's consumers instead of trusting the hand-maintained set.

    The completeness test only catches a file that attributes to *nothing*. It cannot catch a
    declared set that is too *small*, which is the failure this table exists to prevent and
    which has occurred twice: SameShapeCommon.cuh and PhotometricBound.cuh omitted direct
    consumers, and VarShapeUtils.hpp omitted nine operators reaching it transitively through
    UnaryElementwiseOp.hpp. Only headers are derivable this way - a .cu entry maps to the
    operators its kernels implement, which the include graph cannot know.
    """
    graph = _include_graph()
    wrong = {}
    for path, declared in MULTI_OP_SOURCES.items():
        if not path.endswith((".cuh", ".hpp")) or path in _UNDERIVABLE_HEADERS:
            continue
        derived = set()
        for consumer in _transitive_includers(path, graph):
            derived |= operators_for_path(consumer, ops)
        if derived != set(declared):
            wrong[path] = {
                "missing": sorted(derived - set(declared)),
                "unexpected": sorted(set(declared) - derived),
            }
    assert wrong == {}, (
        "MULTI_OP_SOURCES disagrees with the #include graph. A too-small set silently shrinks "
        f"the benchmark scope for changes to that header:\n{json.dumps(wrong, indent=2)}"
    )


def test_every_operator_has_an_advanced_config(ops):
    """The scoped leg runs `--tier advanced`, and run_bench exits 1 for an operator with none.

    Without this, adding an operator that declares only basic entries turns the merge-blocking
    benchmark leg red for any MR that touches it, with the cause in a different repo area.
    """
    missing = []
    for operator in sorted(ops):
        config = REPO / f"bench/config/operators/{operator}.json"
        tiers = {
            entry.get("tier")
            for entry in json.loads(config.read_text())["configs"].values()
            if isinstance(entry, dict)
        }
        if "advanced" not in tiers:
            missing.append(operator)
    assert missing == [], (
        "Changed-operator jobs benchmark the advanced tier; these declare "
        f"none, so a merge request touching them would fail: {missing}"
    )


def test_header_derivation_terminates_on_an_include_cycle(ops):
    """A header reachable from itself must not bounce between the two derivation helpers.

    _derived_header_owners resolves consumers, and operators_for_path routes an unclaimed
    header back into _derived_header_owners, so a cycle would recurse until the stack blew.
    src/cvcuda/priv/legacy/textbackend/stb_truetype.h is self-reachable in the real graph today
    and survives only because a MULTI_OP_SOURCES prefix claims it first.
    """
    import operator_source_map as osm

    cyclic = {
        "src/cvcuda/priv/A.cuh": {"src/cvcuda/priv/B.cuh"},
        "src/cvcuda/priv/B.cuh": {"src/cvcuda/priv/A.cuh"},
    }
    with mock.patch.object(osm, "_include_graph", lambda: cyclic):
        # Would raise RecursionError without the derive_headers guard.
        assert osm.operators_for_path("src/cvcuda/priv/A.cuh", ops) == set()


def test_the_real_graph_has_a_self_reachable_header(ops):
    """Pin the premise above: if this stops being true the cycle test still guards, but the
    comment explaining why the guard exists would go stale."""
    import operator_source_map as osm

    header = "src/cvcuda/priv/legacy/textbackend/stb_truetype.h"
    assert header in osm._transitive_includers(header)
