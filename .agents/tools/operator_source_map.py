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
for every operator."""

from pathlib import Path

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
    merely a prefix of the real owner's (e.g. `gaussian` must not claim `gaussian_noise.cu`)."""
    key = file_stem.replace("_", "").lower()
    if not key.startswith(op):
        return False
    return not any(o != op and len(o) > len(op) and key.startswith(o) for o in all_ops)


SHARED_KERNEL_SOURCES = {
    "conv2d": ["legacy/filter_var_shape.cu"],
    "gaussian": ["legacy/filter.cu", "legacy/filter_var_shape.cu"],
    "histogram": ["legacy/calc_hist.cu"],
    "warpaffine": ["legacy/warp.cu", "legacy/warp_var_shape.cu"],
    "warpperspective": ["legacy/warp.cu", "legacy/warp_var_shape.cu"],
    "hqresize": [
        "OpHQResize2D.cu",
        "OpHQResize3D.cu",
        "OpHQResizeBatchWrap.cuh",
        "OpHQResizeDispatch.hpp",
        "OpHQResizeFilter.cuh",
        "OpHQResizeKernel.cuh",
        "OpHQResizePlanar.cuh",
    ],
    "stack": ["OpStackKernels.cu", "OpStackKernels.hpp"],
    "averageblur": ["legacy/filter.cu", "legacy/filter_var_shape.cu"],
    "laplacian": ["legacy/filter.cu", "legacy/filter_var_shape.cu"],
    "reformat": ["legacy/reformat.cu"],
}
