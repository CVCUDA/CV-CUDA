# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the CvtColor benchmark."""

import importlib.util
import sys
import types
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_cvtcolor_benchmark(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Type = types.SimpleNamespace(U8="U8", F16="F16", F32="F32")
    fake_cvcuda.Format = types.SimpleNamespace(
        **{
            name: name
            for name in (
                "RGB8",
                "RGB8p",
                "BGR8",
                "BGR8p",
                "RGBA8",
                "RGBA8p",
                "Y8",
                "HSV8",
                "LAB8",
                "LAB8p",
                "LABf16",
                "LABf16p",
                "LABf32",
                "LABf32p",
                "YUV8p",
                "NV12",
                "BGRf16",
                "BGRf16p",
                "BGRf32",
                "BGRf32p",
                "RGBf16",
                "RGBf16p",
                "RGBf32",
                "RGBf32p",
            )
        }
    )
    fake_cvcuda.ColorConversion = types.SimpleNamespace(
        **{
            name: name
            for name in (
                "RGB2BGR",
                "RGB2RGBA",
                "RGBA2RGB",
                "RGB2GRAY",
                "GRAY2RGB",
                "RGB2HSV",
                "HSV2RGB",
                "BGR2Lab",
                "RGB2Lab",
                "Lab2BGR",
                "Lab2RGB",
                "LBGR2Lab",
                "LRGB2Lab",
                "Lab2LBGR",
                "Lab2LRGB",
                "RGB2YUV",
                "YUV2RGB",
                "RGB2YUV_NV12",
                "YUV2RGB_NV12",
            )
        }
    )
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    tensor_calls = []
    image_calls = []
    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda value: value
    fake_utils.parse_shape = lambda _shape: (2, 3, 4)
    fake_utils.get_dtype = lambda dtype: {
        "uint8": fake_cvcuda.Type.U8,
        "float16": fake_cvcuda.Type.F16,
        "float32": fake_cvcuda.Type.F32,
    }[dtype]
    fake_utils.get_dtype_size = lambda dtype: {
        fake_cvcuda.Type.U8: 1,
        fake_cvcuda.Type.F16: 2,
        fake_cvcuda.Type.F32: 4,
    }[dtype]
    fake_utils.get_format_from_dtype = lambda dtype, channels, planar=False: (
        f"{fake_utils.get_dtype(dtype)}_C{channels}{'_PLANAR' if planar else ''}"
    )
    fake_utils.create_tensor = lambda *args, **kwargs: tensor_calls.append(
        (args, kwargs)
    )
    fake_utils.create_image_batch_varshape = lambda *args, **kwargs: image_calls.append(
        (args, kwargs)
    )
    fake_utils.create_stream_cache = lambda: lambda _launch: None
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "python_bench_utils", fake_utils)

    spec = importlib.util.spec_from_file_location(
        "bench_cvtcolor_test", BENCH_DIR / "python" / "ops" / "bench_cvtcolor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, tensor_calls, image_calls


def test_f16_allocation_formats_channels_and_accounting(monkeypatch):
    bench_cvtcolor, tensor_calls, image_calls = _load_cvtcolor_benchmark(monkeypatch)

    class State:
        def __init__(self, code, input_kind, dtype="float16", layout="NHWC"):
            self.values = {
                "shape": "2x3x4",
                "InOutDataType": dtype,
                "code": code,
                "inputKind": input_kind,
                "layout": layout,
            }
            self.reads = []
            self.writes = []

        def get_string(self, name):
            return self.values[name]

        def get_device(self):
            return 0

        def add_global_memory_reads(self, value):
            self.reads.append(value)

        def add_global_memory_writes(self, value):
            self.writes.append(value)

        def skip(self, message):
            self.skipped = message

    tensor_state = State("RGB2GRAY", "Tensor")
    bench_cvtcolor.cvtcolor(tensor_state)
    assert [(args[0], args[1]) for args, _kwargs in tensor_calls] == [
        ((2, 3, 4, 3), "F16"),
        ((2, 3, 4, 1), "F16"),
    ]
    assert (tensor_state.reads, tensor_state.writes) == ([144], [48])

    varshape_state = State("RGB2RGBA", "VarShape")
    bench_cvtcolor.cvtcolor(varshape_state)
    assert [(args[0], args[2], args[3]) for args, _kwargs in image_calls] == [
        ((2, 3, 4, 3), "F16_C3", "F16"),
        ((2, 3, 4, 4), "F16_C4", "F16"),
    ]
    assert (varshape_state.reads, varshape_state.writes) == ([144], [192])

    tensor_calls.clear()
    nv12_f16_state = State("RGB2YUV_NV12", "Tensor")
    bench_cvtcolor.cvtcolor(nv12_f16_state)
    assert nv12_f16_state.skipped == "NV12 CvtColor benchmarks support only U8"
    assert tensor_calls == []

    nv12_state = State("RGB2YUV_NV12", "Tensor", dtype="uint8")
    bench_cvtcolor.cvtcolor(nv12_state)
    assert [args[1] for args, _kwargs in tensor_calls] == ["U8", "U8"]
    assert (nv12_state.reads, nv12_state.writes) == ([72], [36])

    image_calls.clear()
    bgr_state = State("RGB2BGR", "VarShape")
    bench_cvtcolor.cvtcolor(bgr_state)
    assert [args[2] for args, _kwargs in image_calls] == ["F16_C3", "BGRf16"]

    image_calls.clear()
    lab_state = State("RGB2Lab", "VarShape")
    bench_cvtcolor.cvtcolor(lab_state)
    assert [args[2] for args, _kwargs in image_calls] == ["F16_C3", "LABf16"]

    image_calls.clear()
    lab_f32_state = State("Lab2RGB", "VarShape", dtype="float32", layout="NCHW")
    bench_cvtcolor.cvtcolor(lab_f32_state)
    assert [args[2] for args, _kwargs in image_calls] == ["LABf32p", "F32_C3_PLANAR"]

    image_calls.clear()
    linear_bgr_state = State("BGR2Lab", "VarShape", layout="NCHW")
    bench_cvtcolor.cvtcolor(linear_bgr_state)
    assert [args[2] for args, _kwargs in image_calls] == ["BGRf16p", "LABf16p"]

    image_calls.clear()
    lab_linear_bgr_state = State("Lab2LBGR", "VarShape", dtype="uint8", layout="NCHW")
    bench_cvtcolor.cvtcolor(lab_linear_bgr_state)
    assert [args[2] for args, _kwargs in image_calls] == ["LAB8p", "BGR8p"]


def test_cpp_varshape_fill_is_typed_and_shared():
    source = (BENCH_DIR / "cpp" / "ops" / "BenchCvtColor.cpp").read_text()
    entrypoint = source.partition("inline void cvtcolor")[2]

    assert "std::vector<uint8_t>" not in source
    assert "nvcv::cuda::MakeType<BaseT, Channels>" in source
    assert "benchutils::FillImageBatch<PixelT>" in source
    assert "benchutils::FillPlanarImageBatch<PixelT>" in source
    assert "RunVarShapeBenchmark<BaseT>" in source
    assert "NVCV_IMAGE_FORMAT_RGBf32" in source
    assert "NVCV_IMAGE_FORMAT_LABf32" in source
    assert "nvcv::FMT_LABf32p" in source
    remap = entrypoint.index("FormatForBaseType<BaseT>")
    assert (
        entrypoint.index("HasSubsampledFormat(nvcv::ImageFormat{inFormatValue}") < remap
    )
    assert entrypoint.index("inFormatValue == NVCV_IMAGE_FORMAT_YUV8") < remap
