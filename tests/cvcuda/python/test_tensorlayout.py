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

import itertools

import pytest as t

import cvcuda

# The six single-dimension labels. All must be exposed as named constants so
# users never need to reach for strings for the atomic layouts.
SINGLE_LABELS = ["N", "C", "F", "D", "H", "W"]

# Every predefined multi-dimension layout, in the declaration order of
# src/nvcv/src/include/nvcv/TensorLayoutDef.inc. These lists are the single
# source of truth for the parametrized tests below; test_exposed_constants_match
# keeps them in lockstep with the binding, so any layout added to (or dropped
# from) the .inc forces a matching update here.
MULTI_LABELS = [
    "WC", "CW",
    "NW", "NWC", "NCW",
    "HW", "NHW",
    "FHW", "NFHW",
    "CHW", "NCHW", "HWC", "NHWC",
    "CFHW", "FCHW", "FHWC", "NCFHW", "NFCHW", "NFHWC",
    "DHW", "NDHW",
    "CDHW", "DHWC", "NCDHW", "NDHWC",
    "FDHW", "NFDHW",
    "CFDHW", "FCDHW", "FDHWC", "NCFDHW", "NFCDHW", "NFDHWC",
]  # fmt: skip

# Every predefined named layout constant (excludes NONE, which is the empty
# layout and is exercised separately).
ALL_LABELS = SINGLE_LABELS + MULTI_LABELS


def test_exposed_constants_match_expected_set():
    # Exhaustiveness guard: the named layout constants the binding actually
    # exposes must be exactly ALL_LABELS plus NONE. This fails loudly if a
    # layout is ever added to TensorLayoutDef.inc without being covered here, or
    # if one silently stops being exposed to Python.
    exposed = {
        name
        for name in dir(cvcuda.TensorLayout)
        if isinstance(getattr(cvcuda.TensorLayout, name), cvcuda.TensorLayout)
    }
    assert exposed == set(ALL_LABELS) | {"NONE"}


@t.mark.parametrize("label", SINGLE_LABELS)
def test_single_dim_labels_are_exposed(label):
    # Each atomic label must be exposed as a named constant that is value-equal
    # to its string form and round-trips through str().
    layout = getattr(cvcuda.TensorLayout, label)
    assert layout == cvcuda.TensorLayout(label)
    assert str(layout) == label


@t.mark.parametrize("label", ALL_LABELS)
def test_named_constant_matches_string_constructor(label):
    assert getattr(cvcuda.TensorLayout, label) == cvcuda.TensorLayout(label)
    assert str(getattr(cvcuda.TensorLayout, label)) == label


@t.mark.parametrize("label", ALL_LABELS)
def test_string_round_trip(label):
    # A layout survives a full label -> TensorLayout -> str -> TensorLayout trip,
    # from both the named constant and the string constructor.
    named = getattr(cvcuda.TensorLayout, label)
    built = cvcuda.TensorLayout(label)
    assert cvcuda.TensorLayout(str(named)) == named
    assert cvcuda.TensorLayout(str(built)) == built
    assert str(named) == str(built) == label


@t.mark.parametrize("label", ALL_LABELS)
def test_hash_is_value_based(label):
    # Two layouts that compare equal must hash equal, regardless of whether they
    # came from a named constant or a string.
    named = getattr(cvcuda.TensorLayout, label)
    built = cvcuda.TensorLayout(label)
    assert named == built
    assert hash(named) == hash(built)


@t.mark.parametrize("label", ALL_LABELS)
def test_layout_equals_matching_string(label):
    # A string implicitly converts to a TensorLayout, so equality works directly
    # against the string form in either operand order.
    layout = getattr(cvcuda.TensorLayout, label)
    assert layout == label
    assert label == layout
    assert not (layout != label)
    assert not (label != layout)


@t.mark.parametrize("label", ALL_LABELS)
def test_layout_not_equal_to_other_string(label):
    # Comparing against the string of a *different* layout must be unequal in
    # either operand order. "NCHW" is the reference; use "HWC" for it so the
    # counterpart is never the label itself.
    layout = getattr(cvcuda.TensorLayout, label)
    other = "HWC" if label == "NCHW" else "NCHW"
    assert layout != other
    assert other != layout
    assert not (layout == other)
    assert not (other == layout)


@t.mark.parametrize("other", [123, 3.14, None, ("N", "H", "W", "C"), object()])
def test_equality_is_total_against_foreign_types(other):
    # Comparing against a non-string, non-layout object must return False (not
    # raise) so TensorLayout is safe to use in heterogeneous containers.
    assert not (cvcuda.TensorLayout.NHWC == other)
    assert cvcuda.TensorLayout.NHWC != other


@t.mark.parametrize("bad", ["", "ZZZ", "nhwc"])
def test_equality_against_unrelated_strings(bad):
    # Empty, unknown, and wrong-case strings are all valid TensorLayout inputs
    # (labels are case-sensitive), so they simply compare unequal to NHWC.
    assert cvcuda.TensorLayout.NHWC != bad
    assert not (cvcuda.TensorLayout.NHWC == bad)


def test_all_layouts_are_distinct():
    # No two distinct predefined layouts may compare equal or collapse together
    # in a set (value-based hashing must keep them apart).
    layouts = [getattr(cvcuda.TensorLayout, label) for label in ALL_LABELS]
    assert len(set(layouts)) == len(ALL_LABELS)
    for a, b in itertools.combinations(layouts, 2):
        assert a != b
        assert not (a == b)


def test_none_layout_is_exposed():
    assert cvcuda.TensorLayout.NONE == cvcuda.TensorLayout("")
    assert cvcuda.TensorLayout.NONE == ""
    assert cvcuda.TensorLayout.NONE != cvcuda.TensorLayout.N
    assert cvcuda.TensorLayout.NONE != "N"
    # NONE participates in value-based hashing like any other layout.
    assert hash(cvcuda.TensorLayout.NONE) == hash(cvcuda.TensorLayout(""))
    assert len({cvcuda.TensorLayout.NONE, cvcuda.TensorLayout("")}) == 1


def test_usable_as_dict_key():
    d = {cvcuda.TensorLayout.NCHW: "planar", cvcuda.TensorLayout.NHWC: "packed"}
    # Look up with a freshly-constructed, value-equal key.
    assert d[cvcuda.TensorLayout("NCHW")] == "planar"
    assert d[cvcuda.TensorLayout("NHWC")] == "packed"


def test_usable_in_set():
    # Value-equal layouts collapse to a single set element.
    assert len({cvcuda.TensorLayout("NHWC"), cvcuda.TensorLayout.NHWC}) == 1
    assert len({cvcuda.TensorLayout.NCHW, cvcuda.TensorLayout.NHWC}) == 2


@t.mark.parametrize(
    "shape, layout",
    [
        ((10,), cvcuda.TensorLayout.N),
        ((10,), cvcuda.TensorLayout.C),
        ((5, 16, 32, 4), cvcuda.TensorLayout.NHWC),
        ((16, 32, 4), cvcuda.TensorLayout.HWC),
    ],
)
def test_tensor_construction_accepts_layout_enum(shape, layout):
    # The exact case from the request: passing TensorLayout.N (rather than "N")
    # must work now that the atomic labels are exposed.
    tensor = cvcuda.Tensor(shape, cvcuda.Type.U8, layout)
    assert tensor.shape == shape
    assert tensor.layout == layout
    assert tensor.layout == cvcuda.TensorLayout(str(layout))


@t.mark.parametrize("label", ["N", "C", "NHWC", "HWC", "NCHW"])
def test_tensor_construction_accepts_layout_string(label):
    # A string layout implicitly converts, and the resulting tensor's layout is
    # value-equal to the enum-built one -- the round trip works in real usage.
    shape = (10,) if len(label) == 1 else tuple(range(2, 2 + len(label)))
    from_str = cvcuda.Tensor(shape, cvcuda.Type.U8, label)
    from_enum = cvcuda.Tensor(
        shape, cvcuda.Type.U8, getattr(cvcuda.TensorLayout, label)
    )
    assert from_str.layout == from_enum.layout
    assert from_str.layout == label
    assert str(from_str.layout) == label
