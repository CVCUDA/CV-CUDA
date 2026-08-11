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

"""Tests for human-readable benchmark axis formatting."""

from __future__ import annotations

from _internal.axes import format_axes, format_axis_name, format_axis_value


def test_axis_names_pass_through():
    assert format_axis_name("inputKind") == "inputKind"
    assert format_axis_name("shape") == "shape"


def test_input_kind_values_are_self_describing():
    # inputKind is a string axis carrying its own labels; nothing to translate.
    assert format_axis_value("inputKind", "Tensor") == "Tensor"
    assert format_axis_value("inputKind", "VarShape") == "VarShape"


def test_format_axes_renders_input_kind():
    axes = (
        ("InOutDataType", "uint8"),
        ("shape", "64x1080x1920"),
        ("inputKind", "VarShape"),
    )

    assert format_axes(axes) == (
        "InOutDataType=uint8, shape=64x1080x1920, inputKind=VarShape"
    )
