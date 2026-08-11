# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for cross-language benchmark state ordering."""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

from config.axis_order import order_axis_names
from config.load_config import ConfigLoader, register_axes_from_config


BENCH_DIR = Path(__file__).resolve().parent.parent
OPERATORS_DIR = BENCH_DIR / "config/operators"


class _AxisRecorder:
    def __init__(self):
        self.axes = []

    def add_string_axis(self, name, values):
        self.axes.append((name, list(values)))

    def add_int64_axis(self, name, values):
        self.axes.append((name, list(values)))

    def add_float64_axis(self, name, values):
        self.axes.append((name, list(values)))


def _operator_documents():
    for path in sorted(OPERATORS_DIR.glob("*.json")):
        document = json.loads(path.read_text())
        yield path, document["benchmark"], document["configs"]


def _cpp_registered_axes(tmp_path, documents):
    generated_headers = []
    invocations = []
    for path, operator, _ in documents:
        header = tmp_path / f"Bench{operator.title()}Config.hpp"
        subprocess.run(
            [
                "cmake",
                f"-DJSON_FILE={path}",
                f"-DOPERATOR={operator}",
                f"-DOPERATOR_CAMEL={operator.title()}",
                f"-DOUTPUT_FILE={header}",
                "-P",
                str(BENCH_DIR / "cpp/GenerateBenchConfig.cmake"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        generated_headers.append(header.name)
        invocations.append(
            f'    Recorder recorder_{operator}("{operator}");\n'
            f"    recorder_{operator} BENCH_{operator.upper()}_AXES;"
        )

    fake_include = tmp_path / "include/nvbench"
    fake_include.mkdir(parents=True)
    (fake_include / "nvbench.cuh").write_text("\n")
    source = tmp_path / "record_axes.cpp"
    includes = "\n".join(f'#include "{header}"' for header in generated_headers)
    source.write_text(
        f"""\
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>

{includes}

struct Recorder
{{
    explicit Recorder(const char *operator_name)
        : operator_name(operator_name)
    {{
    }}

    Recorder &set_type_axes_names(std::initializer_list<const char *> names)
    {{
        for (const char *name : names)
            std::cout << operator_name << '\\t' << name << '\\n';
        return *this;
    }}

    Recorder &add_string_axis(const char *name, std::initializer_list<const char *>)
    {{
        std::cout << operator_name << '\\t' << name << '\\n';
        return *this;
    }}

    Recorder &add_int64_axis(const char *name, std::initializer_list<int64_t>)
    {{
        std::cout << operator_name << '\\t' << name << '\\n';
        return *this;
    }}

    Recorder &add_float64_axis(const char *name, std::initializer_list<double>)
    {{
        std::cout << operator_name << '\\t' << name << '\\n';
        return *this;
    }}

    const char *operator_name;
}};

int main()
{{
{chr(10).join(invocations)}
}}
"""
    )
    executable = tmp_path / "record_axes"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            f"-I{tmp_path / 'include'}",
            f"-I{tmp_path}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    ).stdout
    registered = {}
    for line in output.splitlines():
        operator, name = line.split("\t")
        registered.setdefault(operator, []).append(name)
    return registered


def _effective_cpp_axes(registered_axes, config):
    values_by_name = {
        "InOutDataType": config.dtypes,
        **config.string_axes,
        **config.int64_axes,
        **config.float64_axes,
    }
    return [
        (name, values_by_name[name])
        for name in registered_axes
        if name in values_by_name
    ]


def _state_order(axes):
    names = [name for name, _ in axes]
    reversed_values = [values for _, values in reversed(axes)]
    return [
        dict(zip(names, reversed(values)))
        for values in itertools.product(*reversed_values)
    ]


def test_future_axes_are_automatically_sorted_by_utf8_bytes():
    assert order_axis_names(
        ({"zFuture": [], "AFuture": [], "äFuture": [], "aFuture": []},)
    ) == [
        "AFuture",
        "aFuture",
        "zFuture",
        "äFuture",
    ]


def test_all_cartesian_state_orders_match_cpp_and_python(tmp_path):
    documents = list(_operator_documents())
    cpp_axes = _cpp_registered_axes(tmp_path, documents)
    checked_configs = 0

    for path, operator, entries in documents:
        loader = ConfigLoader(str(path))
        for config_key in entries:
            config = loader.get_operator_config(config_key)
            python_recorder = _AxisRecorder()
            python_recorder.add_string_axis("InOutDataType", config.dtypes)
            register_axes_from_config(python_recorder, config)

            assert [name for name, _ in python_recorder.axes] == sorted(
                (name for name, _ in python_recorder.axes),
                key=lambda name: name.encode("utf-8"),
            )

            cpp_effective_axes = _effective_cpp_axes(cpp_axes[operator], config)
            assert [name for name, _ in cpp_effective_axes] == [
                name for name, _ in python_recorder.axes
            ], f"{operator}: {config_key}"

            cpp_states = _state_order(cpp_effective_axes)
            python_states = _state_order(python_recorder.axes)
            assert cpp_states == python_states, f"{operator}: {config_key}"
            checked_configs += 1

    assert checked_configs == sum(len(entries) for _, _, entries in documents)
