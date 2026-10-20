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

"""Unit tests for scoping the benchmark noise gate to the change under test."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import run_bench

NOISY = 31.32
QUIET = 0.20
LIMIT = 10.0


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "Benchmark": operator,
                "GPU Noise (%)": noise,
                "Language": "cpp",
                "InOutDataType": "uchar3",
                "shape": "1x720x1280",
                "inputKind": "Tensor",
            }
            for operator, noise in rows
        ]
    )


def _processor(scope=None):
    return run_bench.ResultsProcessor(LIMIT, LIMIT, 100.0, noise_scope=scope)


def test_no_scope_blocks_on_every_operator():
    """An unscoped run cannot know what the change is, so everything answers."""

    errors, warnings = _processor()._check_noise(
        _frame([("inpaint", NOISY), ("invert", NOISY)])
    )

    assert len(errors) == 2
    assert warnings == []


def test_noise_outside_the_scope_is_reported_but_not_blocking():
    """!1817 failed on inpaint noise; it changes six operators, none of them inpaint."""

    errors, warnings = _processor({"invert", "solarize"})._check_noise(
        _frame([("inpaint", NOISY), ("solarize", QUIET)])
    )

    assert errors == []
    assert len(warnings) == 1
    assert "inpaint" in warnings[0].config


def test_noise_inside_the_scope_still_blocks():
    """The gate must keep catching instability in what the MR actually touched."""

    errors, warnings = _processor({"invert", "solarize"})._check_noise(
        _frame([("inpaint", NOISY), ("invert", NOISY)])
    )

    assert len(errors) == 1
    assert "invert" in errors[0].config
    assert len(warnings) == 1
    assert "inpaint" in warnings[0].config


def test_a_quiet_operator_produces_neither():
    errors, warnings = _processor({"invert"})._check_noise(
        _frame([("invert", QUIET), ("inpaint", QUIET)])
    )

    assert errors == []
    assert warnings == []


def test_validation_result_only_fails_on_blocking_noise():
    result = run_bench.ValidationResult()
    result.noise_warnings = [run_bench.ValidationError("high_noise", "inpaint (cpp)")]

    assert not result.has_errors

    result.noise_errors = [run_bench.ValidationError("high_noise", "invert (cpp)")]
    assert result.has_errors


MANIFEST = {"invert": {}, "solarize": {}, "inpaint": {}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("invert", {"invert"}),
        ("invert,solarize", {"invert", "solarize"}),
        (" invert , solarize ", {"invert", "solarize"}),
    ],
)
def test_scope_parsing_accepts_the_same_syntax_as_operator(raw, expected):
    assert run_bench.parse_noise_scope(raw, MANIFEST) == expected


def test_an_empty_scope_is_refused_rather_than_silently_blocking_nothing():
    """'' must not read as 'no operator blocks' — that would disable the gate."""

    with pytest.raises(ValueError, match="--noise-scope: empty value"):
        run_bench.parse_noise_scope(" , ", MANIFEST)


def test_an_unknown_operator_is_refused_rather_than_matching_no_row():
    """A typo must fail loudly.

    An unrecognised name matches no Benchmark row, so every noisy row would fall
    outside the scope and the gate would go advisory for the whole run while
    reporting green — strictly worse than the empty case above.
    """

    with pytest.raises(ValueError, match="unknown operator"):
        run_bench.parse_noise_scope("inpiant", MANIFEST)


def test_the_scope_matches_operator_names_exactly():
    """One syntax across selectors: --operator rejects a miscased name, so does this."""

    with pytest.raises(ValueError, match="did you mean inpaint"):
        run_bench.parse_noise_scope("Inpaint", MANIFEST)


def test_every_operator_selector_shares_one_resolver():
    """--operator, --benchmarks and --noise-scope must not drift apart again."""

    for option in ("--operator", "--benchmarks", "--noise-scope"):
        with pytest.raises(ValueError, match=f"{option}: unknown operator"):
            run_bench._resolve_operator_selector("inpiant", MANIFEST, option)


def test_an_empty_scope_fails_argument_parsing_not_the_bench_run(monkeypatch):
    """The gate must reject a bad scope before any benchmark work is done."""

    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--noise-scope", " , "])
    with pytest.raises(SystemExit) as excinfo:
        run_bench.parse_args()
    assert excinfo.value.code == 2
