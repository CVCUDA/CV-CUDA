# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-backed storage parity coverage for the CustomCrop benchmark."""

import importlib.util
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent


class _State:
    def __init__(self, dtype, layout):
        self.dtype = dtype
        self.layout = layout

    def get_string(self, name):
        return {
            "shape": "1x6x19",
            "InOutDataType": self.dtype,
            "cropMode": "center_half",
            "inputKind": "Tensor",
            "layout": self.layout,
        }[name]

    def get_device(self):
        return 0

    def add_global_memory_reads(self, _value):
        pass

    def add_global_memory_writes(self, _value):
        pass


@pytest.mark.parametrize("dtype", ["ulonglong3", "float16"])
@pytest.mark.parametrize("layout", ["NHWC", "NCHW", "NCHW_FAKE"])
def test_customcrop_tensors_match_native_allocation_strides(monkeypatch, dtype, layout):
    cvcuda = pytest.importorskip("cvcuda")
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if not device_count:
        pytest.skip("A CUDA device is required")

    monkeypatch.syspath_prepend(str(BENCH_DIR / "python"))
    spec = importlib.util.spec_from_file_location(
        "bench_customcrop_storage_test",
        BENCH_DIR / "python" / "ops" / "bench_customcrop.py",
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    calls = []

    def capture(dst, src, *_args, **_kwargs):
        calls.append((dst, src))
        return dst

    monkeypatch.setattr(cvcuda, "customcrop_into", capture)
    monkeypatch.setattr(cvcuda, "reformat_into", capture)
    monkeypatch.setattr(benchmark, "create_stream_cache", lambda: lambda _launch: None)

    with cp.cuda.Device(0):
        run = benchmark.customcrop(_State(dtype, layout))
        run(None)
        assert len(calls) == (3 if layout == "NCHW_FAKE" else 1)
        for dst, src in calls:
            for tensor in (src, dst):
                native = cvcuda.Tensor(tensor.shape, tensor.dtype, tensor.layout)
                assert tensor.cuda().strides == native.cuda().strides, (
                    f"{dtype} {layout}: shape={tensor.shape}, layout={tensor.layout}, "
                    f"benchmark strides={tensor.cuda().strides}, "
                    f"native strides={native.cuda().strides}"
                )


def test_native_tensor_fill_ready_on_another_stream(monkeypatch):
    cvcuda = pytest.importorskip("cvcuda")
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("A CUDA device is required")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    monkeypatch.syspath_prepend(str(BENCH_DIR / "python"))
    from python_bench_utils import create_tensor

    delay = cp.RawKernel(
        r"""
        extern "C" __global__ void delay_copy() {
            unsigned long long start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            } while (now - start < 250000000ULL);
        }
        """,
        "delay_copy",
    )
    with cp.cuda.Device(0):
        delay.compile()
        shape = (1, 3, 19, 1)
        producer = cp.cuda.Stream(non_blocking=True)
        consumer = cp.cuda.Stream(non_blocking=True)
        native = cvcuda.Tensor(shape, cvcuda.Type.S32, "NHWC")
        view = cp.asarray(native.cuda())
        source = cp.full(shape, 37, dtype=cp.int32)
        output = cp.empty(shape, dtype=cp.int32)
        copyto = cp.copyto
        copyto(view, source)
        copyto(output, view)
        cp.cuda.get_current_stream().synchronize()
        retained = []

        def delayed_copyto(dst, src, *args, **kwargs):
            if not isinstance(src, cp.ndarray):
                return copyto(dst, src, *args, **kwargs)
            # Keep the temporary source alive so releasing it cannot hide missing ordering.
            retained.extend((dst, src))
            dst.fill(-91)
            producer.synchronize()
            delay((1,), (1,), ())
            copyto(dst, src, *args, **kwargs)

        monkeypatch.setattr(cp, "copyto", delayed_copyto)
        try:
            with producer:
                tensor = create_tensor(
                    shape, cvcuda.Type.S32, fill_mode=37, native_allocation=True
                )
            assert tensor.shape == shape
            with consumer:
                copyto(output, retained[0])
            consumer.synchronize()
            assert (cp.asnumpy(output) == 37).all()
        finally:
            producer.synchronize()
