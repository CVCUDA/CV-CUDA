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

"""Regression tests for ResourceGuard lifetime and exception safety.

Failure scenarios are driven by the private ``cvcuda._test`` failure-injection
toggles, which make the corresponding C API callback fail through its
production error path.  Every scenario runs in a subprocess so a crash on
unfixed code (SIGABRT from a throwing noexcept destructor) does not kill
pytest.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np

import cvcuda

_ABORT_REPRO_SCRIPT = textwrap.dedent(
    """\
    import cvcuda

    try:
        cvcuda._test.resourceguard_destructor_error()
    except SystemError as exc:
        # The destructor preserves the injected pending error instead of
        # clearing it; CPython surfaces the non-raising binding that returned
        # with an error set as a SystemError chaining the original.
        assert isinstance(exc.__cause__, RuntimeError), repr(exc.__cause__)
        assert "injected ResourceGuard commit failure" in str(exc.__cause__)

    print("PASS", flush=True)
    """
)


_POST_SUBMIT_HOLD_FAILURE_SCRIPT = textwrap.dedent(
    """\
    import numpy as np
    import cupy as cp
    import cvcuda

    stream = cvcuda.Stream()
    from cvcuda_test_utils import as_cupy_stream

    external_stream = as_cupy_stream(stream.handle)
    src = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    dst = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")

    # Warm the operator and drain its resource-release callback so the only
    # outstanding work below is the deliberately delayed submission.
    cvcuda.flip_into(src=src, dst=dst, flipCode=0, stream=stream)
    stream.sync()
    cvcuda.internal.drain_pending_releases()

    delay = cp.RawKernel(
        r'''\\
        extern "C" __global__ void delay(unsigned long long cycles)
        {
            unsigned long long start = clock64();
            while (clock64() - start < cycles)
            {
            }
        }
        ''',
        "delay",
    )
    delay((1,), (1,), (np.uint64(500_000_000),), stream=external_stream)
    assert not external_stream.done, "delay kernel completed before failure injection"

    # Fail the post-submission hold: flip's kernel is queued behind the delay
    # kernel, then Stream_HoldResources rejects the lifetime hold.
    cvcuda._test.fail_hold_resources(True)
    error = None
    try:
        cvcuda.flip_into(src=src, dst=dst, flipCode=0, stream=stream)
    except Exception as exc:
        error = exc
    finally:
        cvcuda._test.fail_hold_resources(False)

    # Query before cleanup: ResourceGuard itself must have drained the stream
    # while it still owned every resource reference.
    drained = external_stream.done
    stream.sync()

    assert isinstance(error, ValueError), repr(error)
    assert "injected Stream_HoldResources failure" in str(error), str(error)
    assert drained, "ResourceGuard released references while GPU work was active"
    print("PASS", flush=True)
    """
)


_LEGACY_HOLD_FAILURE_SCRIPT = textwrap.dedent(
    """\
    import sys
    import numpy as np
    import cvcuda

    stream = cvcuda.Stream()
    resource = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    refs_before = tuple(sys.getrefcount(obj) for obj in (stream, resource))
    cvcuda._test.fail_hold_resources(True)
    errors = []
    try:
        for _ in range(5):
            try:
                cvcuda._test.resourceguard_legacy_commit(stream, resource)
            except Exception as exc:
                errors.append(exc)
    finally:
        cvcuda._test.fail_hold_resources(False)

    refs_after = tuple(sys.getrefcount(obj) for obj in (stream, resource))
    assert len(errors) == 5, errors
    assert all(isinstance(error, ValueError) for error in errors), errors
    assert all("injected Stream_HoldResources failure" in str(error) for error in errors)
    assert refs_after == refs_before, (refs_before, refs_after)
    print("PASS", flush=True)
    """
)


_PRESYNC_FAILURE_SCRIPT = textwrap.dedent(
    """\
    import sys
    import numpy as np
    import cvcuda

    stream = cvcuda.Stream()
    src = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    dst = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")

    # Fail the pre-submission sync; also arm the legacy combined callback so a
    # destructor retry on unfixed code fails loudly instead of silently
    # re-submitting the sync it already reported as failed.
    cvcuda._test.fail_submit_sync_only(True)
    cvcuda._test.fail_sync_and_hold(True)
    refs_before = tuple(sys.getrefcount(obj) for obj in (stream, src, dst))
    errors = []
    try:
        for _ in range(5):
            try:
                cvcuda.flip_into(src=src, dst=dst, flipCode=0, stream=stream)
            except Exception as exc:
                errors.append(exc)
    finally:
        cvcuda._test.fail_submit_sync_only(False)
        cvcuda._test.fail_sync_and_hold(False)

    refs_after = tuple(sys.getrefcount(obj) for obj in (stream, src, dst))
    assert len(errors) == 5, errors
    assert all(isinstance(error, ValueError) for error in errors), errors
    assert all(
        "injected Resources_SubmitSyncOnly failure" in str(error) for error in errors
    ), errors
    assert refs_after == refs_before, (refs_before, refs_after)
    print("PASS", flush=True)
    """
)


_SUBMISSION_ERROR_WITH_CLEANUP_FAILURE_SCRIPT = textwrap.dedent(
    """\
    import numpy as np
    import cvcuda

    stream = cvcuda.Stream()
    src = cvcuda.Tensor((1, 8, 8, 1), np.int16, "NHWC")
    dst = cvcuda.Tensor((1, 8, 8, 1), np.int16, "NHWC")

    cvcuda._test.fail_hold_resources(True)
    error = None
    try:
        # Flip rejects S16 inside the submission callable.  ResourceGuard then
        # encounters the injected secondary failure while finalizing the hold
        # for any work the callable might have queued before throwing.
        cvcuda.flip_into(src=src, dst=dst, flipCode=0, stream=stream)
    except Exception as exc:
        error = exc
    finally:
        cvcuda._test.fail_hold_resources(False)

    assert isinstance(error, RuntimeError), repr(error)
    # The status name is the contract; the message body is the operator's own text.
    assert "NVCV_ERROR_INVALID_ARGUMENT" in str(error), str(error)
    assert "data type" in str(error), str(error)
    print("PASS", flush=True)
    """
)


_NATIVE_STREAM_CAPTURE_REJECTION_SCRIPT = textwrap.dedent(
    r"""
    import numpy as np
    import cupy as cp
    import cvcuda

    cupy_stream = cp.cuda.Stream(non_blocking=True)
    stream = cvcuda.as_stream(cupy_stream)
    src = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    dst = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    scratch = cp.zeros((1,), dtype=cp.uint32)
    kernel = cp.RawKernel(
        r'''
        extern "C" __global__ void increment(unsigned int *value)
        {
            ++value[0];
        }
        ''',
        "increment",
    )

    # Warm every lazy path before capture; CUDA capture forbids allocations.
    kernel((1,), (1,), (scratch,), stream=cupy_stream)
    cvcuda.convertto_into(src=src, dst=dst, stream=stream)
    stream.sync()

    cupy_stream.begin_capture()
    kernel((1,), (1,), (scratch,), stream=cupy_stream)
    error = None
    try:
        cvcuda.convertto_into(src=src, dst=dst, stream=stream)
    except Exception as exc:
        error = exc

    # Rejection must happen before any CV-CUDA submission invalidates the
    # capture. The caller can still finish and launch the work already added.
    graph = cupy_stream.end_capture()
    graph.launch(stream=cupy_stream)
    cupy_stream.synchronize()

    assert isinstance(error, ValueError), repr(error)
    assert "not supported during CUDA stream capture" in str(error), str(error)
    assert int(scratch.get()[0]) == 2
    print("PASS", flush=True)
    """
)


_LEGACY_STREAM_CAPTURE_REJECTION_SCRIPT = textwrap.dedent(
    r"""
    import numpy as np
    import cupy as cp
    import cvcuda

    cupy_stream = cp.cuda.Stream(non_blocking=True)
    stream = cvcuda.as_stream(cupy_stream)
    tensor = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    scratch = cp.zeros((1,), dtype=cp.uint32)
    kernel = cp.RawKernel(
        r'''
        extern "C" __global__ void increment(unsigned int *value)
        {
            ++value[0];
        }
        ''',
        "increment",
    )

    # Warm both the kernel and the public ResourceGuard/C-API adapter used by
    # out-of-tree bindings before capture.
    kernel((1,), (1,), (scratch,), stream=cupy_stream)
    cvcuda._test.legacy_resourceguard_noop(stream, tensor)
    stream.sync()

    cupy_stream.begin_capture()
    kernel((1,), (1,), (scratch,), stream=cupy_stream)
    error = None
    try:
        cvcuda._test.legacy_resourceguard_noop(stream, tensor)
    except Exception as exc:
        error = exc

    graph = cupy_stream.end_capture()
    graph.launch(stream=cupy_stream)
    cupy_stream.synchronize()

    assert isinstance(error, ValueError), repr(error)
    assert "not supported during CUDA stream capture" in str(error), str(error)
    assert int(scratch.get()[0]) == 2
    print("PASS", flush=True)
    """
)


def _run_repro(script):
    script = (
        f"import sys; sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        + script
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_resourceguard_destructor_no_abort_on_commit_error():
    """ResourceGuard destructor must not call std::terminate() when commit() raises.

    Regression: the destructor was implicitly noexcept; CheckCAPIError() inside
    commit() could throw a Python binding exception, which caused
    std::terminate() (SIGABRT) rather than a recoverable Python exception.

    The internal test hook constructs a real ResourceGuard and sets a Python
    exception immediately before scope exit.  The destructor must neither
    terminate nor discard that error: it is fetched before finalization and
    re-instated afterwards, so it propagates to the caller.

    Runs in a subprocess so a crash on unfixed code does not kill pytest.
    """
    result = _run_repro(_ABORT_REPRO_SCRIPT)
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"Process aborted or did not reach PASS "
        f"(returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def _assert_capture_rejection(script, path):
    result = _run_repro(script)
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"{path} capture rejection corrupted the graph "
        f"(returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def test_native_resourceguard_rejects_capture_before_invalidating_graph():
    """Unsupported capture must be rejected before the operator submits.

    ConvertTo uses the direct native sentinel path. A post-submit lifetime
    check is too late: CUDA has already admitted the operator kernel into the
    graph and a failing event/callback operation can invalidate the capture.
    Run this in a subprocess because ending an invalidated capture has
    triggered driver teardown crashes in the original failure mode.
    """
    _assert_capture_rejection(_NATIVE_STREAM_CAPTURE_REJECTION_SCRIPT, "native")


def test_legacy_resourceguard_rejects_capture_before_invalidating_graph():
    """The public ResourceGuard/C-API path must reject capture before run()."""
    _assert_capture_rejection(_LEGACY_STREAM_CAPTURE_REJECTION_SCRIPT, "legacy")


def test_resourceguard_post_submit_hold_failure_drains_and_propagates():
    """A failed post-submit hold must drain before references are released.

    The hold is what keeps resources alive until the submitted kernel
    completes.  When it fails, releasing the references immediately would free
    GPU memory the kernel may still be reading or writing; the guard must
    first prove completion by draining the stream, and the failure must reach
    Python instead of being swallowed by the destructor.
    """
    result = _run_repro(_POST_SUBMIT_HOLD_FAILURE_SCRIPT)
    assert (
        result.returncode == 0
        and "PASS" in result.stdout
        and "~ResourceGuard" not in result.stderr
    ), (
        f"Post-submit recovery failed (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def test_resourceguard_legacy_hold_failure_does_not_leak_references():
    """A failed hold after successful legacy sync may release after draining."""
    result = _run_repro(_LEGACY_HOLD_FAILURE_SCRIPT)
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"Legacy hold recovery leaked references (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def test_resourceguard_presync_failure_is_not_retried():
    """A failed pre-submit synchronization must not be retried in teardown.

    When run() fails before the submission callable executes, no work was
    queued on behalf of the guard, so there is nothing to hold.  The destructor
    must not fall back to the legacy sync-and-hold call (a retry of the exact
    operation that just failed), and no resource references may leak.
    """
    result = _run_repro(_PRESYNC_FAILURE_SCRIPT)
    assert (
        result.returncode == 0
        and "PASS" in result.stdout
        and "~ResourceGuard" not in result.stderr
    ), (
        f"Pre-submit failure was retried (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def test_resourceguard_submission_error_remains_primary_during_cleanup_failure():
    """A hold failure during unwind must not replace the submission error.

    The callable's own exception is what the user needs to see; the secondary
    cleanup failure is logged to stderr by the destructor instead.
    """
    result = _run_repro(_SUBMISSION_ERROR_WITH_CLEANUP_FAILURE_SCRIPT)
    assert (
        result.returncode == 0
        and "PASS" in result.stdout
        and "~ResourceGuard: commit() threw:" in result.stderr
        and "injected Stream_HoldResources failure" in result.stderr
    ), (
        f"Cleanup replaced the submission error (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def test_deferred_legacy_python_owner_is_released_after_off_gil_retirement():
    stream = cvcuda.Stream()
    resource = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    refs_before = sys.getrefcount(resource)

    cvcuda._test.legacy_resourceguard_noop(stream, resource)
    stream.sync()
    cvcuda._test.retire_pending_releases_off_gil(stream)

    assert sys.getrefcount(resource) == refs_before


def test_external_stream_owner_is_deferred_during_off_gil_retirement():
    result = _run_repro(
        textwrap.dedent(
            """\
            import gc
            import weakref

            import cupy as cp
            import cvcuda

            producer = cp.cuda.Stream(non_blocking=True)
            reference = weakref.ref(producer)
            stream = cvcuda.as_stream(producer)
            cvcuda._test.enqueue_native_noop(stream)
            producer.synchronize()

            del stream, producer
            gc.collect()
            assert reference() is not None

            cvcuda._test.retire_pending_releases_off_gil(cvcuda.Stream())
            gc.collect()
            assert reference() is None
            print("PASS", flush=True)
            """
        )
    )
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"Off-GIL stream-owner retirement failed (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def test_external_stream_owner_is_not_displaced_off_gil():
    result = _run_repro(
        textwrap.dedent(
            """\
            import ctypes
            import gc
            import weakref

            import cupy as cp
            import cvcuda
            import numpy as np

            producer = cp.cuda.Stream(non_blocking=True)
            delay = cp.RawKernel(
                r'''\
                extern "C" __global__ void delay(unsigned long long cycles)
                {
                    unsigned long long start = clock64();
                    while (clock64() - start < cycles)
                    {
                    }
                }
                ''',
                "delay",
            )
            delay((1,), (1,), (np.uint64(1),), stream=producer)
            producer.synchronize()

            first_owner = ctypes.c_void_p(producer.ptr)
            second_owner = ctypes.c_void_p(producer.ptr)
            third_owner = ctypes.c_void_p(producer.ptr)
            first = cvcuda.as_stream(first_owner)
            second = cvcuda.as_stream(second_owner)
            third = cvcuda.as_stream(third_owner)

            finalized_with_gil = []
            ctypes.pythonapi.PyGILState_Check.restype = ctypes.c_int
            second_reference = weakref.ref(second_owner)
            weakref.finalize(
                second_owner,
                lambda: finalized_with_gil.append(ctypes.pythonapi.PyGILState_Check()),
            )

            cvcuda.internal.drain_pending_releases()
            delay((1,), (1,), (np.uint64(10_000_000_000),), stream=producer)
            cvcuda._test.enqueue_native_noop(first)
            cvcuda._test.enqueue_native_noop(second)
            assert cvcuda.internal._pending_release_batch_count() == 2

            del second, second_owner
            gc.collect()
            assert second_reference() is not None

            assert not producer.done
            cvcuda._test.retire_pending_releases_off_gil(third)
            gc.collect()

            assert finalized_with_gil == [1], finalized_with_gil
            assert second_reference() is None
            print("PASS", flush=True)
            """
        )
    )
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"Off-GIL stream-owner publication failed (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )


def test_off_gil_retirement_setup_failure_preserves_stream_capture():
    result = _run_repro(
        textwrap.dedent(
            """\
            import cupy as cp
            import cvcuda

            cupy_stream = cp.cuda.Stream(non_blocking=True)
            stream = cvcuda.as_stream(cupy_stream)
            cupy_stream.begin_capture()
            error = None
            try:
                cvcuda._test.retire_pending_releases_off_gil(stream)
            except Exception as exc:
                error = exc
            finally:
                graph = cupy_stream.end_capture()

            assert isinstance(error, ValueError), repr(error)
            assert "not supported during CUDA stream capture" in str(error), str(error)
            graph.launch(stream=cupy_stream)
            stream.sync()
            cvcuda._test.retire_pending_releases_off_gil(stream)
            print("PASS", flush=True)
            """
        )
    )
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"Detached setup failure corrupted cleanup (returncode={result.returncode}).\n"
        f"stdout: {result.stdout[:1000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )
