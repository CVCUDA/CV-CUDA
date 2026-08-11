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

import ctypes
import os
import re
from collections import defaultdict
from pathlib import Path

import cupy
import cvcuda
import numpy as np
import pytest

import cvcuda_util as util


ROOT = Path(__file__).resolve().parents[3]
PYTHON_OPERATOR_DIR = ROOT / "python" / "mod_cvcuda" / "operators"
C_API_INCLUDE_DIR = ROOT / "src" / "cvcuda" / "include" / "cvcuda"
C_API_SOURCE_DIR = ROOT / "src" / "cvcuda"

PYTHON_BINDING_RE = re.compile(r'\bm\.def\(\s*"([^"]+)"')
C_API_DECLARATION_RE = re.compile(
    r"\bCVCUDA_PUBLIC\s+NVCVStatus\s+(cvcuda[A-Za-z0-9]+Submit)\s*\("
)
C_API_DEFINITION_RE = re.compile(
    r"CVCUDA_DEFINE_API\(\s*[^,]+,\s*[^,]+,\s*NVCVStatus\s*,\s*"
    r"(cvcuda[A-Za-z0-9]+Submit)\s*,"
)


def _load_probe():
    path = os.environ.get("NVTX_INJECTION64_PATH")
    if not path or not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    lib.CvcudaNvtxProbe_Reset.restype = None
    lib.CvcudaNvtxProbe_Count.restype = ctypes.c_uint
    lib.CvcudaNvtxProbe_Name.restype = ctypes.c_char_p
    lib.CvcudaNvtxProbe_Name.argtypes = [ctypes.c_uint]
    return lib


_PROBE = _load_probe()
requires_probe = pytest.mark.skipif(
    _PROBE is None,
    reason="NVTX injection probe not available (NVTX_INJECTION64_PATH unset)",
)
requires_python_sources = pytest.mark.skipif(
    not PYTHON_OPERATOR_DIR.is_dir(),
    reason="Python operator sources are unavailable in installed-package tests",
)
requires_c_api_sources = pytest.mark.skipif(
    not C_API_INCLUDE_DIR.is_dir() or not C_API_SOURCE_DIR.is_dir(),
    reason="C API sources are unavailable in installed-package tests",
)


def _reset_probe():
    _PROBE.CvcudaNvtxProbe_Reset()


def _captured_ranges():
    count = _PROBE.CvcudaNvtxProbe_Count()
    return [
        _PROBE.CvcudaNvtxProbe_Name(i).decode("utf-8", "replace") for i in range(count)
    ]


RNG = np.random.default_rng(0)


def _tensor(shape=(2, 16, 24, 3), dtype=np.uint8, layout="NHWC", max_random=255):
    return util.create_tensor(shape, dtype, layout, max_random=max_random, rng=RNG)


def _flip():
    cvcuda.flip(_tensor(), 0)


def _channelreorder_tensor():
    cvcuda.channelreorder(_tensor(), [2, 1, 0])


def _public_operator_names():
    names = set(dir(cvcuda))
    return {
        name
        for name in names
        if not name.startswith("_")
        and callable(getattr(cvcuda, name))
        and f"{name}_into" in names
    }


def _source_matches(paths, pattern):
    matches = defaultdict(list)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for match in pattern.finditer(source):
            matches[match.group(1)].append((path, source, match))
    return matches


def _location(path, source, offset):
    line = source.count("\n", 0, offset) + 1
    return f"{path.relative_to(ROOT)}:{line}"


@requires_python_sources
def test_all_python_operator_bindings_are_instrumented():
    """Every overload of every public operator must use its matching Python NVTX range."""
    sources = sorted(PYTHON_OPERATOR_DIR.glob("Op*.cpp"))
    assert sources, f"no Python operator sources found under {PYTHON_OPERATOR_DIR}"
    bindings = _source_matches(sources, PYTHON_BINDING_RE)
    operator_names = _public_operator_names()
    expected_bindings = operator_names | {f"{name}_into" for name in operator_names}

    missing = sorted(expected_bindings - bindings.keys())
    untraced = []
    for name in sorted(expected_bindings & bindings.keys()):
        expected_trace = re.compile(
            rf'\s*,\s*NvtxTrace\(\s*"cvcuda\.{re.escape(name)}"\s*,'
        )
        for path, source, match in bindings[name]:
            if not expected_trace.match(source, match.end()):
                untraced.append(_location(path, source, match.start()))

    assert not missing and not untraced, (
        f"missing operator bindings: {missing}; "
        f"bindings without their matching NvtxTrace: {untraced}"
    )


@requires_c_api_sources
def test_all_c_api_submit_entries_are_instrumented():
    """Every public submit API must push its matching range before executing its body."""
    declaration_sources = sorted(C_API_INCLUDE_DIR.glob("Op*.h"))
    definition_sources = sorted(C_API_SOURCE_DIR.glob("*.cpp"))
    assert declaration_sources, f"no C API headers found under {C_API_INCLUDE_DIR}"
    assert definition_sources, f"no C API sources found under {C_API_SOURCE_DIR}"
    declarations = _source_matches(declaration_sources, C_API_DECLARATION_RE)
    definitions = _source_matches(definition_sources, C_API_DEFINITION_RE)

    missing_definitions = sorted(declarations.keys() - definitions.keys())
    undeclared_definitions = sorted(definitions.keys() - declarations.keys())
    untraced = []
    for name in sorted(declarations.keys() & definitions.keys()):
        for path, source, match in definitions[name]:
            body_start = source.find("{", match.end())
            expected_range = re.compile(
                rf'\s*CVCUDA_NVTX_RANGE\(\s*"{re.escape(name)}"\s*\)\s*;'
            )
            if body_start < 0 or not expected_range.match(source, body_start + 1):
                untraced.append(_location(path, source, match.start()))

    assert not missing_definitions and not undeclared_definitions and not untraced, (
        f"submit declarations without definitions: {missing_definitions}; "
        f"submit definitions without declarations: {undeclared_definitions}; "
        f"submit definitions without an entry range: {untraced}"
    )


@requires_probe
def test_operator_python_range_wraps_submit():
    """An injected probe observes the Python range followed by the C-API submit range."""
    _reset_probe()
    _flip()
    captured = _captured_ranges()
    assert (
        "cvcuda.flip" in captured
    ), f"missing Python operator range. Captured: {captured}"
    assert "cvcudaFlipSubmit" in captured, (
        "NVTX injection probe loaded but did not capture the submit range; NVTX may have "
        f"initialized before the injection library was registered. Captured: {captured}"
    )
    assert captured.index("cvcuda.flip") < captured.index(
        "cvcudaFlipSubmit"
    ), f"Python range must wrap the submit range. Captured: {captured}"


@requires_probe
def test_channelreorder_tensor_submit_marker():
    _reset_probe()
    _channelreorder_tensor()
    assert "cvcudaChannelReorderSubmit" in _captured_ranges()


@requires_probe
def test_stream_methods_emit_markers():
    """Stream methods and functions push their own NVTX ranges."""
    _reset_probe()
    stream = cvcuda.Stream()
    with stream:
        _flip()
    stream.sync()
    cvcuda.Stream.default.wait_stream(stream)
    captured = _captured_ranges()
    for expected in (
        "cvcuda.Stream.__enter__",
        "cvcuda.Stream.__exit__",
        "cvcuda.Stream.sync",
        "cvcuda.Stream.wait_stream",
    ):
        assert (
            expected in captured
        ), f"missing stream range '{expected}'. Captured: {captured}"


@requires_probe
def test_container_and_transfer_markers():
    """Container factories, transfers, and stream creation push their own NVTX ranges."""
    _reset_probe()

    cvcuda.Stream()

    tensor = _tensor()
    tensor.cuda()
    buf = cupy.zeros((2, 16, 24, 3), dtype=cupy.uint8)
    cvcuda.as_tensor(buf, "NHWC")
    cvcuda.as_tensors([buf])
    cvcuda.reshape(tensor, tensor.shape, tensor.layout)

    image = util.create_image((24, 16), cvcuda.Format.RGB8, max_random=255, rng=RNG)
    image.cuda()
    image.cpu()
    cvcuda.as_image(image.cuda(), cvcuda.Format.RGB8)
    cvcuda.as_images([image.cuda()])

    captured = _captured_ranges()
    for expected in (
        "cvcuda.Stream.create",
        "cvcuda.Tensor.cuda",
        "cvcuda.as_tensor",
        "cvcuda.as_tensors",
        "cvcuda.reshape",
        "cvcuda.Image.cuda",
        "cvcuda.Image.cpu",
        "cvcuda.as_image",
        "cvcuda.as_images",
    ):
        assert expected in captured, f"missing range '{expected}'. Captured: {captured}"
