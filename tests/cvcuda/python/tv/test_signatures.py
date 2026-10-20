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

"""Surface + signature parity for the whole cvcuda.tv catalog.

Pure introspection (no tensors, no GPU), so it runs on a CPU-only runner and never
requests the GPU-gated fixtures.
"""

import ast
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from cvcuda import tv as cvcuda_tv

from conftest import param_spec

# The mirrored functional surface is exactly the public, lowercase names:
# integration entry points are underscore-prefixed and classes such as Compose
# have no torchvision functional counterpart to compare against.
_CATALOG = sorted(name for name in cvcuda_tv.__all__ if name[0].islower())


def test_declarative_parity_catalog():
    module = ast.parse(Path(__file__).with_name("test_parity.py").read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert [node.name for node in functions] == [f"test_{name}" for name in _CATALOG]
    for function in functions:
        image_marks = [
            mark
            for mark in function.decorator_list
            if isinstance(mark, ast.Call)
            and isinstance(mark.func, ast.Name)
            and mark.func.id == "images"
        ]
        assert len(image_marks) == 1, function.name
        assert {keyword.arg for keyword in image_marks[0].keywords} >= {
            "dtypes",
            "channels",
        }


@pytest.mark.parametrize("name", _CATALOG)
def test_signature_parity(tvf, name):
    function = getattr(cvcuda_tv, name)
    assert hasattr(tvf, name), f"torchvision.transforms.v2.functional has no {name}"
    ours = inspect.signature(function)
    theirs = inspect.signature(getattr(tvf, name))
    assert set(get_type_hints(function)) == set(ours.parameters) | {"return"}

    assert "stream" in ours.parameters, f"cvcuda.tv.{name} is missing the stream kwarg"
    stream = ours.parameters["stream"]
    assert stream.kind == inspect.Parameter.KEYWORD_ONLY
    assert stream.default is None

    if name == "gaussian_noise":
        seed = ours.parameters["seed"]
        assert seed.kind == inspect.Parameter.KEYWORD_ONLY
        assert seed.default is None
    else:
        assert "seed" not in ours.parameters

    extensions = {"stream", "seed"} if name == "gaussian_noise" else {"stream"}
    ours_spec = param_spec(ours, extensions)
    theirs_spec = param_spec(theirs)
    assert ours_spec == theirs_spec, (
        f"cvcuda.tv.{name} parameters {ours_spec} != "
        f"torchvision {name} parameters {theirs_spec}"
    )
