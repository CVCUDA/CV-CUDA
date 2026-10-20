# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for TvResize benchmark batch patterns."""

import ast
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_SIZES = {
    ("ordinary", False): [(1080, 1920), (2160, 3840), (7, 7)],
    ("mixed-singleton", False): [(1080, 1920), (2160, 3840), (1, 7)],
    ("ordinary", True): [(720, 1280), (1080, 1920), (4, 3)],
    ("mixed-singleton", True): [(720, 1280), (1080, 1920), (4, 3)],
}


@pytest.fixture(scope="module")
def python_helpers():
    path = BENCH_DIR / "python/ops/bench_tvresize.py"
    module = ast.parse(path.read_text())
    module.body = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_sizes", "_ramp")
    ]
    namespace = {"cp": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def cpp_helpers(tmp_path_factory):
    original = (BENCH_DIR / "cpp/ops/BenchTvResize.cpp").read_text()
    start = original.index("template<typename T>\nT TvResizeRamp(")
    end = original.index("template<typename T>\nvoid FillTvResizeTensor(", start)
    directory = tmp_path_factory.mktemp("tvresize-host")
    source = directory / "tvresize.cpp"
    executable = directory / "tvresize"
    source.write_text(
        """
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

struct int2 { int x; int y; };
"""
        + original[start:end]
        + r"""
int main(int, char **arguments)
{
    if (std::string_view(arguments[1]) == "ramp")
    {
        const auto sample = std::stol(arguments[2]);
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (long index = 0; index < 42; ++index)
        {
            std::cout << static_cast<int>(TvResizeRamp<uint8_t>(index, sample)) << ' '
                      << TvResizeRamp<float>(index, sample) << '\n';
        }
        return 0;
    }
    try
    {
        const auto sizes = TvResizeSizes(std::stoi(arguments[2]), std::stoi(arguments[3]),
                                        std::stoi(arguments[4]), arguments[5],
                                        std::stoi(arguments[6]) != 0);
        for (const auto &size : sizes)
        {
            std::cout << size.x << ' ' << size.y << '\n';
        }
    }
    catch (const std::invalid_argument &error)
    {
        std::cerr << error.what();
        return 2;
    }
}
"""
    )
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


@pytest.fixture(params=["python", "cpp"])
def sizes(request):
    if request.param == "python":
        return request.getfixturevalue("python_helpers")["_sizes"]
    executable = request.getfixturevalue("cpp_helpers")

    def run(count, height, width, batch_case, output):
        result = subprocess.run(
            [
                str(executable),
                "sizes",
                str(count),
                str(height),
                str(width),
                batch_case,
                str(int(output)),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 2:
            raise ValueError(result.stderr)
        result.check_returncode()
        return [tuple(map(int, line.split())) for line in result.stdout.splitlines()]

    return run


@pytest.mark.parametrize("count", [3, 6, 12, 24])
@pytest.mark.parametrize("batch_case,output", ORIGINAL_SIZES)
def test_heterogeneous_repeats_original_three_sizes(sizes, count, batch_case, output):
    expected = ORIGINAL_SIZES[batch_case, output] * (count // 3)
    assert sizes(count, 17, 29, batch_case, output) == expected


@pytest.mark.parametrize("count", [-6, -3, -1, 0, 1, 2, 4, 5, 7])
@pytest.mark.parametrize("batch_case,output", ORIGINAL_SIZES)
def test_heterogeneous_rejects_invalid_counts(sizes, count, batch_case, output):
    with pytest.raises(ValueError, match="Invalid TvResize batchCase/count"):
        sizes(count, 17, 29, batch_case, output)


@pytest.mark.parametrize("count", [0, 3, 6])
@pytest.mark.parametrize("batch_case", ["", "unknown"])
@pytest.mark.parametrize("output", [False, True])
def test_rejects_unknown_case(sizes, count, batch_case, output):
    with pytest.raises(ValueError, match="Invalid TvResize batchCase/count"):
        sizes(count, 17, 29, batch_case, output)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 6, 12, 24])
@pytest.mark.parametrize("output", [False, True])
def test_uniform_sizes_unchanged(sizes, count, output):
    assert sizes(count, 17, 29, "uniform", output) == [(17, 29)] * count


@pytest.mark.parametrize("count", [-6, -3, -1])
@pytest.mark.parametrize("output", [False, True])
def test_uniform_rejects_negative_counts(sizes, count, output):
    with pytest.raises(ValueError, match="Invalid TvResize batchCase/count"):
        sizes(count, 17, 29, "uniform", output)


@pytest.mark.parametrize("language", ["python", "cpp"])
@pytest.mark.parametrize("sample", [0, 1, 2, 3, 5, 11, 23])
def test_ramp_preserves_absolute_sample_index(request, language, sample):
    expected = np.array([(index * 17 + sample * 13) % 251 for index in range(42)])
    if language == "python":
        ramp = request.getfixturevalue("python_helpers")["_ramp"]
        integers = ramp(2, 7, 3, sample, np.uint8).reshape(-1)
        floats = ramp(2, 7, 3, sample, np.float32).reshape(-1)
    else:
        executable = request.getfixturevalue("cpp_helpers")
        result = subprocess.run(
            [str(executable), "ramp", str(sample)],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [line.split() for line in result.stdout.splitlines()]
        integers = np.array([int(value[0]) for value in values], dtype=np.uint8)
        floats = np.array([float(value[1]) for value in values], dtype=np.float32)
    np.testing.assert_array_equal(integers, expected.astype(np.uint8))
    np.testing.assert_array_equal(floats, expected.astype(np.float32) / np.float32(251))
