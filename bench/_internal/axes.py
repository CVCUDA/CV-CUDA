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

"""Human-readable formatting for benchmark config axes."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def format_axis_name(name: str) -> str:
    """Return the display label for a config axis."""
    return name


def format_axis_value(name: str, value) -> str:
    """Return the display value for a config axis (empty string when missing)."""
    if _is_missing(value):
        return ""
    return str(value)


def format_axes(axes: Iterable[Tuple[str, str]]) -> str:
    """Format row-key axes for reports."""
    return (
        ", ".join(f"{format_axis_name(k)}={format_axis_value(k, v)}" for k, v in axes)
        or "-"
    )


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text in {"", "nan", "NaN", "<NA>", "None"}
