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

import enum
import operator
import pickle

import cvcuda
import pytest


def test_enum_alias_uses_canonical_name():
    canonical = cvcuda.ColorConversion.BGR2BGRA
    alias = cvcuda.ColorConversion.RGB2RGBA

    assert alias == canonical
    assert alias is not canonical
    assert alias.name == canonical.name == "BGR2BGRA"
    assert str(alias) == str(canonical) == "ColorConversion.BGR2BGRA"
    assert repr(alias) == repr(canonical) == "<ColorConversion.BGR2BGRA: 0>"


def test_enum_aliases_are_discoverable():
    assert set(cvcuda.ColorConversion.__members__) <= set(dir(cvcuda.ColorConversion))


def test_enum_members_are_returned_as_an_isolated_dictionary():
    members = cvcuda.ThreadScope.__members__

    assert type(members) is dict
    members["EXTRA"] = cvcuda.ThreadScope.GLOBAL
    assert "EXTRA" not in cvcuda.ThreadScope.__members__


def test_enum_docs_include_ordered_members():
    assert cvcuda.AdaptiveThresholdType.__doc__ == (
        "Adaptive threshold computation methods.\n\n"
        "Members:\n\n"
        "  MEAN_C : Threshold is the mean of the neighbourhood area minus C.\n\n"
        "  GAUSSIAN_C : Threshold is the Gaussian-weighted sum of the "
        "neighbourhood area minus C."
    )
    assert cvcuda.ThreadScope.__doc__ == (
        "Members:\n\n" "  GLOBAL : Global scope.\n\n" "  LOCAL : Thread-local scope."
    )


_ARITHMETIC_ENUM_TYPES = [
    cvcuda.AdaptiveThresholdType,
    cvcuda.ConnectivityType,
    cvcuda.LABEL,
    cvcuda.LabelMaskType,
    cvcuda.Matcher,
    cvcuda.Norm,
    cvcuda.Remap,
    cvcuda.SIFT,
    cvcuda.ThresholdType,
]


@pytest.mark.parametrize("enum_type", _ARITHMETIC_ENUM_TYPES)
def test_arithmetic_enums_are_not_integer_subclasses(enum_type):
    member = next(iter(enum_type.__members__.values()))

    assert not issubclass(enum_type, enum.Enum)
    assert not isinstance(member, enum.Enum)
    assert not issubclass(enum_type, int)
    assert not isinstance(member, int)


@pytest.mark.parametrize(
    "enum_type",
    [cvcuda.ColorConversion, cvcuda.ThresholdType, cvcuda.ThreadScope],
)
def test_enum_classes_do_not_expose_python_enum_container_protocol(enum_type):
    with pytest.raises(TypeError):
        iter(enum_type)
    with pytest.raises(TypeError):
        len(enum_type)
    with pytest.raises(TypeError):
        enum_type[next(iter(enum_type.__members__))]


@pytest.mark.parametrize(
    "enum_type, member",
    [
        (cvcuda.ColorConversion, cvcuda.ColorConversion.BGR2BGRA),
        (cvcuda.ThresholdType, cvcuda.ThresholdType.BINARY),
        (cvcuda.ThreadScope, cvcuda.ThreadScope.GLOBAL),
    ],
)
def test_enum_construction_returns_fresh_equal_wrapper(enum_type, member):
    constructed = enum_type(member.value)

    assert isinstance(constructed, enum_type)
    assert constructed == member
    assert constructed is not member
    assert constructed.name == member.name
    assert constructed.value == member.value
    assert repr(constructed) == repr(member)


def test_unknown_enum_value_matches_pybind_wrapper_behavior():
    first = cvcuda.ThreadScope(123)
    second = cvcuda.ThreadScope(123)

    assert first is not second
    assert first == second
    assert first.name == "???"
    assert first.value == 123
    assert str(first) == "ThreadScope.???"
    assert repr(first) == "<ThreadScope.???: 123>"


@pytest.mark.parametrize(
    "member",
    [
        cvcuda.ColorConversion.BGR2BGRA,
        cvcuda.ThresholdType.BINARY,
        cvcuda.ThreadScope.GLOBAL,
    ],
)
def test_enum_pickle_returns_fresh_equal_wrapper(member):
    restored = pickle.loads(pickle.dumps(member))

    assert type(restored) is type(member)
    assert restored == member
    assert restored is not member


def test_arithmetic_enum_matches_pybind_integer_operations():
    binary = cvcuda.ThresholdType.BINARY
    binary_inv = cvcuda.ThresholdType.BINARY_INV

    assert binary == 1
    assert 1 == binary
    assert binary < 2
    assert binary <= 1
    assert binary_inv > 1
    assert binary_inv >= 2
    assert binary & binary_inv == 0
    assert binary_inv & 3 == 2
    assert 3 & binary_inv == 2
    assert binary | binary_inv == 3
    assert binary | 2 == 3
    assert 2 | binary == 3
    assert binary ^ binary_inv == 3
    assert binary ^ 3 == 2
    assert 3 ^ binary == 2
    assert ~binary == -2
    assert operator.index(binary) == 1
    assert hash(binary) == hash(1)
    assert bool(binary)

    with pytest.raises(TypeError):
        binary + 1


@pytest.mark.parametrize(
    "enum_type, member",
    [
        (cvcuda.ColorConversion, cvcuda.ColorConversion.RGB2RGBA),
        (cvcuda.ThresholdType, cvcuda.ThresholdType.BINARY),
        (cvcuda.ThreadScope, cvcuda.ThreadScope.GLOBAL),
    ],
)
def test_enums_keep_native_caster_registration(enum_type, member):
    if hasattr(enum_type, "__nb_enum__"):
        assert type(enum_type.__nb_enum__).__name__ == "PyCapsule"

    assert int(member) == member.value
    assert str(member).startswith(f"{enum_type.__name__}.")


def test_existing_enum_operator_accepts_canonical_and_fresh_wrappers():
    assert cvcuda.Interp.NEAREST | cvcuda.Interp.LINEAR == 1
    assert cvcuda.Interp(0) | cvcuda.Interp(1) == 1


def test_known_and_unknown_enum_wrappers_convert_back_to_native_enum():
    class DerivedThreadScope(cvcuda.ThreadScope):
        pass

    cvcuda.clear_cache(cvcuda.ThreadScope(cvcuda.ThreadScope.GLOBAL.value))
    cvcuda.clear_cache(cvcuda.ThreadScope(123))
    cvcuda.clear_cache(DerivedThreadScope(123))

    roundtrip = getattr(cvcuda._test, "thread_scope_roundtrip", None)
    if roundtrip is not None:
        assert roundtrip(cvcuda.ThreadScope.GLOBAL) is cvcuda.ThreadScope.GLOBAL

        unknown = cvcuda.ThreadScope(123)
        result = roundtrip(unknown)
        assert type(result) is cvcuda.ThreadScope
        assert result is not unknown
        assert result.value == 123
        assert result.name == "???"


@pytest.mark.parametrize(
    "member",
    [
        cvcuda.ColorConversion.BGR2BGRA,
        cvcuda.ConnectivityType.CONNECTIVITY_4_2D,
        cvcuda.LabelMaskType.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY,
        cvcuda.ThreadScope.GLOBAL,
    ],
)
def test_zero_valued_enum_members_are_truthy(member):
    assert member.value == 0
    assert bool(member)


@pytest.mark.parametrize(
    "member",
    [
        cvcuda.ColorConversion.BGR2BGRA,
        cvcuda.ConnectivityType.CONNECTIVITY_4_2D,
        cvcuda.LabelMaskType.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY,
    ],
)
def test_integer_compatible_enums_compare_equal_to_their_value(member):
    assert member == member.value
    assert member.value == member


@pytest.mark.parametrize(
    "member", [cvcuda.ThreadScope.GLOBAL, cvcuda.ThreadScope.LOCAL]
)
def test_scoped_non_arithmetic_enum_remains_distinct_from_integer(member):
    assert member != member.value
    assert member.value != member


@pytest.mark.parametrize(
    "enum_type, member",
    [
        (cvcuda.ColorConversion, cvcuda.ColorConversion.BGR2BGRA),
        (cvcuda.ConnectivityType, cvcuda.ConnectivityType.CONNECTIVITY_4_2D),
        (cvcuda.ThreadScope, cvcuda.ThreadScope.GLOBAL),
    ],
)
def test_enum_name_and_value_are_public_properties(enum_type, member):
    assert "name" in dir(enum_type)
    assert "value" in dir(enum_type)
    assert isinstance(enum_type.name, property)
    assert isinstance(enum_type.value, property)
    assert member.name in enum_type.__members__
    assert member.value == int(member)
