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

import subprocess
import sys
import textwrap

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
    external_stream = cp.cuda.ExternalStream(stream.handle)
    src = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
    dst = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")

    # Warm the operator and drain its resource-release callback so the only
    # outstanding work below is the deliberately delayed submission.
    cvcuda.flip_into(src=src, dst=dst, flipCode=0, stream=stream)
    stream.sync()
    cvcuda.internal.syncAuxStream()

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
    assert "INVALID_DATA_TYPE" in str(error), str(error)
    print("PASS", flush=True)
    """
)


def _run_repro(script):
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_resourceguard_destructor_no_abort_on_commit_error():
    """ResourceGuard destructor must not call std::terminate() when commit() raises.

    Regression: the destructor was implicitly noexcept; CheckCAPIError() inside
    commit() could throw pybind11::error_already_set, which caused
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
