# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle
import weakref

import pytest as t

import cvcuda as cvcuda


FORMAT_MEMBERS = tuple(cvcuda.Format.__members__.items())
CANONICAL_FORMAT_NAMES = {}
for _name, _format in FORMAT_MEMBERS:
    CANONICAL_FORMAT_NAMES.setdefault(_format.value, _name)


@t.mark.parametrize(
    "format,gold_channels",
    [
        (cvcuda.Format.RGBA8, 4),
        (cvcuda.Format.RGB8, 3),
        (cvcuda.Format._2S16, 2),
        (cvcuda.Format.S8, 1),
        (cvcuda.Format.NV12, 3),
        (cvcuda.Format.F16, 1),
    ],
)
def test_imgformat_numchannels(format, gold_channels):
    assert format.channels == gold_channels


@t.mark.parametrize(
    "format,gold_planes",
    [
        (cvcuda.Format.RGBA8, 1),
        (cvcuda.Format.RGB8, 1),
        (cvcuda.Format._2S16, 1),
        (cvcuda.Format.S8, 1),
        (cvcuda.Format.NV12, 2),
        (cvcuda.Format.F16, 1),
    ],
)
def test_imgformat_planes(format, gold_planes):
    assert format.planes == gold_planes


def test_imgformat_has_complete_member_catalog():
    assert len(FORMAT_MEMBERS) == 91


def test_imgformat_doc_lists_registered_members():
    doc_lines = [
        line.strip() for line in cvcuda.Format.__doc__.splitlines() if line.strip()
    ]

    assert doc_lines == ["Members:", *(name for name, _ in FORMAT_MEMBERS)]


@t.mark.parametrize("registered_name,format", FORMAT_MEMBERS)
def test_imgformat_registered_member_contract(registered_name, format):
    canonical_name = CANONICAL_FORMAT_NAMES[format.value]

    assert getattr(cvcuda.Format, registered_name) is format
    if registered_name.startswith("_"):
        assert not hasattr(cvcuda, registered_name)
    else:
        assert getattr(cvcuda, registered_name) is format
    assert format.name == canonical_name
    assert format.value == int(format)
    assert str(format) == f"Format.{canonical_name}"

    reconstructed = cvcuda.Format(format.value)
    assert type(reconstructed) is cvcuda.Format
    assert reconstructed == format
    assert reconstructed.name == canonical_name
    assert str(reconstructed) == str(format)
    assert repr(reconstructed) == repr(format)


@t.mark.parametrize("registered_name,format", FORMAT_MEMBERS)
def test_imgformat_registered_member_pickle(registered_name, format):
    reconstructed = pickle.loads(pickle.dumps(format))

    assert type(reconstructed) is cvcuda.Format
    assert reconstructed == format
    assert reconstructed.value == format.value
    assert reconstructed.name == CANONICAL_FORMAT_NAMES[format.value]
    assert str(reconstructed) == str(format)
    assert repr(reconstructed) == repr(format)
    assert getattr(cvcuda.Format, registered_name) is format


@t.mark.parametrize(
    "format,canonical_name",
    [
        (cvcuda.Format.NV21, "NV12"),
        (cvcuda.Format.NV21_ER, "NV12_ER"),
        (cvcuda.Format._2S16, "_2S16"),
        (cvcuda.Format._2S16_BL, "_2S16_BL"),
        (cvcuda.Format._2F16, "_2F16"),
        (cvcuda.Format._2F32, "_2F32"),
        (cvcuda.Format._2C64, "_2C64"),
        (cvcuda.Format._2C128, "_2C128"),
    ],
)
def test_imgformat_alias_and_numeric_names(format, canonical_name):
    assert format.name == canonical_name
    assert str(format) == f"Format.{canonical_name}"


def test_imgformat_member_supports_weak_references():
    reference = weakref.ref(cvcuda.Format.U8)

    assert reference() is cvcuda.Format.U8
