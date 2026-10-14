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
"""Regression coverage for contradictory license notices in NVCV headers."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "header_path",
    [
        "src/nvcv/util/Version.hpp",
        "src/nvcv/util/SymbolVersioning.hpp",
        "src/nvcv/src/include/nvcv/detail/CudaFwd.h",
    ],
)
def test_nvcv_header_license_notice(header_path):
    notice = (REPO / header_path).read_text(encoding="utf-8").split("*/", 1)[0]
    normalized_notice = " ".join(line.strip(" *") for line in notice.splitlines())

    assert "SPDX-License-Identifier: Apache-2.0" in notice
    assert "without an express license agreement" not in normalized_notice
    assert "strictly prohibited" not in normalized_notice
    assert (
        "retain all intellectual property and proprietary rights"
        not in normalized_notice
    )
    assert 'Licensed under the Apache License, Version 2.0 (the "License");' in notice
