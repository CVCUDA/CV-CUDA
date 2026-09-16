# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest as t

import cvcuda
import cupy


def test_stream_gcbag_vs_streamsync_race_condition():
    inputImage = cupy.asarray(
        np.random.randint(0, 256, (100, 1500, 1500, 3), dtype=np.uint8)
    )
    cvcudaInputTensor = cvcuda.as_tensor(inputImage, "NHWC")
    inputmap = cupy.asarray(
        np.random.randint(0, 256, (100, 1500, 1500, 2), dtype=np.uint8).astype(
            np.float32
        )
    )
    cvcudaInputMap = cvcuda.as_tensor(inputmap, "NHWC")

    cvcuda_stream = cvcuda.Stream()
    with cvcuda_stream:
        cvcudaResizeTensor = cvcuda.remap(cvcudaInputTensor, cvcudaInputMap)
    del cvcudaResizeTensor


def test_current_stream():
    assert cvcuda.Stream.current is cvcuda.Stream.default
    assert type(cvcuda.Stream.current) is cvcuda.Stream


def test_current_stream_is_thread_local():
    contexts_active = threading.Barrier(2)
    streams_observed = threading.Barrier(2)

    def use_stream():
        default_before = cvcuda.Stream.current
        stream = cvcuda.Stream()
        with stream:
            contexts_active.wait(timeout=10)
            current = cvcuda.Stream.current
            streams_observed.wait(timeout=10)
        return stream, default_before, current, cvcuda.Stream.current

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(use_stream) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]

    for stream, default_before, current, default_after in results:
        assert default_before is cvcuda.Stream.default
        assert current is stream
        assert default_after is cvcuda.Stream.default


def test_stream_context_rejects_cross_thread_exit():
    stream = cvcuda.Stream()
    stream_active = threading.Event()
    allow_stream_exit = threading.Event()

    def use_stream():
        stream.__enter__()
        stream_active.set()
        assert allow_stream_exit.wait(timeout=10)
        assert cvcuda.Stream.current is stream
        stream.__exit__(None, None, None)
        assert cvcuda.Stream.current is cvcuda.Stream.default

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(use_stream)
        assert stream_active.wait(timeout=10)
        try:
            with t.raises(RuntimeError, match="not the current stream on this thread"):
                stream.__exit__(None, None, None)
            assert cvcuda.Stream.current is cvcuda.Stream.default
        finally:
            allow_stream_exit.set()
        future.result(timeout=10)


def test_stream_context_survives_expired_inner_context():
    # A manual __enter__ whose Stream is then destroyed leaves a stack entry that
    # nothing else can pop: the object whose __exit__ would have popped it is
    # gone. It must not hide the enclosing context, nor break its __exit__.
    #
    # as_stream is required here: streams from cvcuda.Stream() are held by the
    # cache, so dropping the caller's reference would never expire them.
    def leak_inner_context():
        outer = cvcuda.as_stream(cupy.cuda.Stream())
        outer.__enter__()
        try:
            inner = cvcuda.as_stream(cupy.cuda.Stream())
            inner.__enter__()
            inner_ref = weakref.ref(inner)
            del inner
            # Assert the precondition rather than assume it, so that a deferred
            # deallocation fails as itself instead of as a wrong current stream.
            assert inner_ref() is None, "inner Stream outlived `del`"
            assert cvcuda.Stream.current is outer
        finally:
            outer.__exit__(None, None, None)
        assert cvcuda.Stream.current is cvcuda.Stream.default

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(leak_inner_context).result(timeout=10)


# Repeats of the off-main-thread import probe below. The failure it guards is
# timing-dependent: one attempt usually passes, ten usually catch it. Ten is
# therefore a hunting configuration, not a gating one -- defaulting to it would
# convert a rare failure into a near-certain one for every pipeline in the
# project. Default to a single attempt and let a job that is looking for the
# crash ask for more.
def _probe_attempts():
    """Attempts per run, floored at one.

    Zero or a negative value would skip the loop entirely and let this test
    pass without running the probe at all. A bad value is clamped rather than
    raised on: this is a hunting amplifier, not a correctness switch, and a
    typo in it should not take out collection of every other test in the file.
    """
    raw = os.environ.get("CVCUDA_STREAM_PROBE_ATTEMPTS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


_OFF_MAIN_THREAD_IMPORT_ATTEMPTS = _probe_attempts()


def _native_backtrace(program):
    """Native backtrace for a crash faulthandler cannot see.

    faulthandler uninstalls its handlers in Py_FinalizeEx, so a SIGSEGV during
    late finalization -- which is where this one lands, since it reports
    nothing -- leaves no Python-level trace at all. Re-running the child under
    gdb is the only way to name the frame. The re-run races the same way the
    original did, so it may well exit cleanly; that is reported rather than
    hidden.
    """
    gdb = shutil.which("gdb")
    if gdb is None:
        return "gdb not installed; no native backtrace available"
    try:
        rerun = subprocess.run(
            [
                gdb,
                "--batch",
                "-ex",
                "run",
                "-ex",
                "bt full",
                "-ex",
                "info threads",
                "--args",
                sys.executable,
                "-X",
                "faulthandler",
                "-c",
                program,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return "gdb re-run timed out"
    return (
        f"gdb exited {rerun.returncode}\n{rerun.stdout[-6000:]}\n{rerun.stderr[-2000:]}"
    )


def test_no_leak_warning_when_imported_off_main_thread():
    # The importing thread and the interpreter's cleanup thread need not be the
    # same, so cleanup must not assume the importer's stack state.
    program = textwrap.dedent(
        """
        import threading

        def work():
            import cvcuda

            assert cvcuda.Stream.current is cvcuda.Stream.default

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        """
    )
    # This has segfaulted in CI roughly once in a hundred and fifty runs, in
    # cache teardown racing interpreter finalization. One attempt per run made
    # it a lottery nobody could reproduce, and the crash carried no stack:
    # -X faulthandler turns SIGSEGV into a native traceback on stderr, which
    # the assertion below reports.
    for attempt in range(_OFF_MAIN_THREAD_IMPORT_ATTEMPTS):
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", program],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            t.fail(
                f"attempt {attempt} exited {result.returncode}\n"
                f"child stderr:\n{result.stderr}\n"
                f"native backtrace from a gdb re-run:\n{_native_backtrace(program)}"
            )
        assert "Stream stack leak detected" not in result.stderr


def test_stream_default_is_read_only():
    # Assigning it used to succeed and change nothing: Stream.current resolves
    # the default through the binding, not through this attribute, so the two
    # names silently disagreed from then on.
    with t.raises(AttributeError):
        cvcuda.Stream.default = cvcuda.Stream()
    assert cvcuda.Stream.current is cvcuda.Stream.default
    assert cvcuda.Stream.default.handle == 0


def test_stream_context_exit_does_not_mask_body_exception():
    # An out-of-order __exit__ is a real error, but reporting it while the body
    # is already unwinding buries the failure the caller needs to see.
    def use_streams():
        outer = cvcuda.Stream()
        inner = cvcuda.Stream()
        with t.raises(ValueError, match="body failure"):
            with outer:
                inner.__enter__()  # deliberately left on the stack
                raise ValueError("body failure")

        # The exit was suppressed rather than applied, so both are still active.
        assert cvcuda.Stream.current is inner
        inner.__exit__(None, None, None)
        outer.__exit__(None, None, None)
        assert cvcuda.Stream.current is cvcuda.Stream.default

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(use_streams).result(timeout=10)


def test_user_stream():
    with cvcuda.Stream():
        assert cvcuda.Stream.current is not cvcuda.Stream.default
    stream = cvcuda.Stream()
    with stream:
        assert stream is cvcuda.Stream.current
        assert stream is not cvcuda.Stream.default
    assert stream is not cvcuda.Stream.default
    assert stream is not cvcuda.Stream.current


def test_nested_streams():
    stream1 = cvcuda.Stream()
    stream2 = cvcuda.Stream()
    assert stream1 is not stream2
    with stream1:
        with stream2:
            assert stream2 is cvcuda.Stream.current
            assert stream1 is not cvcuda.Stream.current
        assert stream2 is not cvcuda.Stream.current
        assert stream1 is cvcuda.Stream.current


def test_wrap_stream_voidp():
    stream = cupy.cuda.Stream()

    extStream = ctypes.c_void_p(stream.ptr)

    cvcudaStream = cvcuda.as_stream(extStream)

    assert extStream.value == cvcudaStream.handle


def test_wrap_stream_int():
    stream = cupy.cuda.Stream()

    extStream = int(stream.ptr)

    cvcudaStream = cvcuda.as_stream(extStream)

    assert extStream == cvcudaStream.handle


def test_stream_conv_to_int():
    stream = cvcuda.Stream()

    assert stream.handle == int(stream)


class MockStream:
    def __init__(self, cuda_stream=None):
        if cuda_stream:
            self.m_stream = cupy.cuda.ExternalStream(cuda_stream)
        else:
            self.m_stream = cupy.cuda.Stream()

    def cuda_stream(self):
        return self.m_stream.ptr

    def stream(self):
        return self.m_stream


@t.mark.parametrize(
    "stream_type",
    [
        MockStream,
    ],
)
def test_wrap_stream_external(stream_type):
    extstream = stream_type()

    # Keep the underlying cupy stream alive across the del below.
    # cupy.cuda.Stream eagerly destroys the CUDA stream in __del__,
    # so we must prevent GC from reclaiming it.
    underlying = extstream.stream()

    stream = cvcuda.as_stream(underlying.ptr)

    assert extstream.cuda_stream() == stream.handle

    del extstream

    extstream = stream_type(stream.handle)
    stream = cvcuda.as_stream(extstream.stream().ptr)

    assert extstream.cuda_stream() == stream.handle

    del underlying


def test_as_stream_cupy_object():
    """cvcuda.as_stream() must accept a cupy.cuda.Stream object directly, not just
    an integer handle.  Without a dedicated type_caster this raises TypeError."""
    stream = cupy.cuda.Stream()
    cvcuda_stream = cvcuda.as_stream(stream)
    assert cvcuda_stream.handle == stream.ptr


def test_as_stream_cupy_object_keeps_stream_alive():
    """When wrapping a cupy stream *by object*, cvcuda must keep the stream alive
    for as long as the cvcuda wrapper exists.
    If the wrapper stores only the integer (m_wrappedObj = int), the cupy stream
    is destroyed the moment the caller drops their reference, leaving a dead handle."""
    import gc

    cupy_stream = cupy.cuda.Stream()
    handle = cupy_stream.ptr

    cvcuda_stream = cvcuda.as_stream(cupy_stream)

    # Drop caller's reference to the cupy stream.
    del cupy_stream
    gc.collect()

    # cvcuda_stream must still hold the cupy stream alive via m_wrappedObj.
    # If the stream was destroyed, streamSynchronize will raise.
    cupy.cuda.runtime.streamSynchronize(cvcuda_stream.handle)
    assert cvcuda_stream.handle == handle


def test_as_stream_cupy_stream_switch():
    """A resource submitted on a cupy stream (via as_stream(cupy_stream)) can be
    safely used on a different stream even after the caller drops their cupy reference.

    as_stream(cupy_stream) keeps the cupy stream alive via m_wrappedObj, so the
    CUDA handle remains valid when submitSync synchronizes against it.
    The chain is: out_nv -> Resource -> m_lastStream -> cvcuda Stream -> cupy_stream.
    """
    import gc

    src = cupy.full((1, 4, 4, 3), fill_value=100, dtype=cupy.uint8)
    src_nv = cvcuda.as_tensor(src, "NHWC")

    # Wrap by object (not .ptr) so cvcuda holds a strong ref to the cupy stream.
    cupy_stream = cupy.cuda.Stream()
    cvcuda_stream = cvcuda.as_stream(cupy_stream)
    with cvcuda_stream:
        out_nv = cvcuda.cvtcolor(
            src_nv, cvcuda.ColorConversion.BGR2GRAY, stream=cvcuda_stream
        )
    cupy_stream.synchronize()

    # Drop caller's references.  The cupy stream stays alive via the ref chain above.
    del cupy_stream
    del cvcuda_stream
    gc.collect()

    # Use out_nv on a fresh native stream.  submitSync synchronizes against the
    # still-valid cupy stream handle held in m_lastStream.
    stream2 = cvcuda.Stream()
    with stream2:
        out2 = cvcuda.cvtcolor(out_nv, cvcuda.ColorConversion.GRAY2BGR, stream=stream2)
    stream2.sync()

    result = cupy.asarray(out2.cuda())
    assert result.shape == (1, 4, 4, 3)


def test_stream_default_is_zero():
    assert cvcuda.Stream.default.handle == 0


def test_stream_size_in_bytes():
    """
    Checks if the computation of the Stream size in bytes is correct
    """
    stream = cvcuda.Stream()
    assert cvcuda.internal.nbytes_in_cache(stream) == 0
