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

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


class _NetworkCreated(Exception):
    pass


def _find_sample(relative_path):
    install_or_source_root = Path(__file__).resolve().parents[3]
    candidates = (
        install_or_source_root / "samples" / relative_path,
        install_or_source_root / "bin" / relative_path,
    )
    sample_path = next((path for path in candidates if path.is_file()), None)
    if sample_path is None:
        pytest.skip(f"{relative_path} is not available in this test installation")
    return sample_path


def _load_samples_common(monkeypatch):
    common_path = _find_sample("common.py")

    cvcuda = ModuleType("cvcuda")
    cuda = ModuleType("cuda")
    cuda.__path__ = []
    cuda_bindings = ModuleType("cuda.bindings")
    cuda_bindings.__path__ = []
    cudart = ModuleType("cuda.bindings.runtime")
    cuda.bindings = cuda_bindings
    cuda_bindings.runtime = cudart
    nvidia = ModuleType("nvidia")
    nvidia.__path__ = []
    nvimgcodec = ModuleType("nvidia.nvimgcodec")
    nvidia.nvimgcodec = nvimgcodec

    for name, module in (
        ("cvcuda", cvcuda),
        ("cuda", cuda),
        ("cuda.bindings", cuda_bindings),
        ("cuda.bindings.runtime", cudart),
        ("nvidia", nvidia),
        ("nvidia.nvimgcodec", nvimgcodec),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "_samples_common_under_test", common_path
    )
    assert spec is not None
    assert spec.loader is not None
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    return common


def test_engine_from_onnx_does_not_use_removed_explicit_batch_flag(monkeypatch):
    common = _load_samples_common(monkeypatch)

    class FakeLogger:
        WARNING = object()

        def __init__(self, severity):
            self.severity = severity

    class FakeBuilder:
        def __init__(self):
            self.network_args = None

        def create_network(self, *args):
            self.network_args = args
            raise _NetworkCreated

    builder = FakeBuilder()
    tensorrt = ModuleType("tensorrt")
    tensorrt.init_libnvinfer_plugins = lambda *_args: None
    tensorrt.Logger = FakeLogger
    tensorrt.Builder = lambda _logger: builder
    monkeypatch.setitem(sys.modules, "tensorrt", tensorrt)

    with pytest.raises(_NetworkCreated):
        common.engine_from_onnx(Path("unused.onnx"), Path("unused.engine"))

    assert builder.network_args == ()


def test_pynvvideocodec_sample_skips_missing_dependency_on_python314(
    monkeypatch, capsys
):
    sample_path = _find_sample("interoperability/pynvvideocodec_interop.py")
    monkeypatch.setitem(sys.modules, "cvcuda", ModuleType("cvcuda"))
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    monkeypatch.setattr(sys, "version_info", (3, 14, 0, "final", 0))

    runpy.run_path(str(sample_path), run_name="__main__")

    assert "Skipping PyNvVideoCodec interoperability sample" in capsys.readouterr().out


def test_pynvvideocodec_sample_requires_dependency_on_supported_python(monkeypatch):
    sample_path = _find_sample("interoperability/pynvvideocodec_interop.py")
    monkeypatch.setitem(sys.modules, "cvcuda", ModuleType("cvcuda"))
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    monkeypatch.setattr(sys, "version_info", (3, 12, 0, "final", 0))

    with pytest.raises(ModuleNotFoundError) as error:
        runpy.run_path(str(sample_path), run_name="__main__")

    assert error.value.name == "PyNvVideoCodec"
