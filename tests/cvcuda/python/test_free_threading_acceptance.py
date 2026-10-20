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

import os
import subprocess
import sys
import sysconfig
import textwrap
from functools import cache

import pytest

import cvcuda_test_utils as test_util

# Free-threaded ABI acceptance coverage, including explicit GIL-enabled runs.
# The ordinary ABI retains main's behavior and is covered by the existing suite.
pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires the free-threaded ABI",
)


# The probe is preloaded before Python starts; bind its C entry points once per child.
_PROBE_SETUP = """
import ctypes
import threading
import cvcuda
probe = ctypes.CDLL(None)
for _name, _arguments, _result in [
    ("cvcudaTestArmBatchMutation", [ctypes.c_int], None),
    ("cvcudaTestArmDetachedFlipGate", None, None),
    ("cvcudaTestArmDeviceSynchronizeGate", None, None),
    ("cvcudaTestArmEventRecordEntryGate", None, None),
    ("cvcudaTestArmEventRecordGate", None, None),
    ("cvcudaTestArmFlipGate", None, None),
    ("cvcudaTestArmGpuGate", None, None),
    ("cvcudaTestArmGpuSignalGate", None, None),
    ("cvcudaTestArmImageBatchMetadataGate", [ctypes.c_int], None),
    ("cvcudaTestArmImageCpuCopyFailure", None, None),
    ("cvcudaTestArmLifetimeGate", None, None),
    ("cvcudaTestArmNextBusyMutexAttempt", None, None),
    ("cvcudaTestArmReaderEntryGate", [ctypes.c_int], None),
    ("cvcudaTestDelayedFlipReleaseReached", None, ctypes.c_int),
    ("cvcudaTestDeleteDLPackManagerOnNativeThread", [ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("cvcudaTestDestroyedTensorCount", None, ctypes.c_int),
    ("cvcudaTestFlipOverlapObserved", None, ctypes.c_int),
    ("cvcudaTestImageCpuCopyApi", None, ctypes.c_int),
    ("cvcudaTestImageCpuCopyPending", None, ctypes.c_int),
    ("cvcudaTestLifetimeWorkPending", None, ctypes.c_int),
    ("cvcudaTestNativeDLPackDeleteThreadWasAttached", None, ctypes.c_int),
    ("cvcudaTestQueueGpuGate", [ctypes.c_void_p], ctypes.c_int),
    ("cvcudaTestQueueGpuSignal", [ctypes.c_void_p], ctypes.c_int),
    ("cvcudaTestReachReaderEntry", None, None),
    ("cvcudaTestReleaseBusyMutexAttempt", None, None),
    ("cvcudaTestReleaseConcurrentStackAndObserveSecond", None, None),
    ("cvcudaTestReleaseDeviceSynchronizeGate", None, None),
    ("cvcudaTestReleaseEventRecordGate", None, None),
    ("cvcudaTestReleaseFindHomographyAndObserveSecond", None, None),
    ("cvcudaTestReleaseFlip", None, None),
    ("cvcudaTestReleaseFlipAfterDelay", [ctypes.c_int], None),
    ("cvcudaTestReleaseFlipAndObserveSecond", None, None),
    ("cvcudaTestReleaseGpuGate", None, None),
    ("cvcudaTestReleaseGpuSignalGate", None, None),
    ("cvcudaTestReleaseImageBatchMetadataAndObserveSecond", None, None),
    ("cvcudaTestReleaseImageCpuFirstCopy", None, None),
    ("cvcudaTestReleaseLifetimeGate", None, None),
    ("cvcudaTestReleaseMorphologyAndObserveSecond", None, None),
    ("cvcudaTestReleaseReaderEntryGate", None, None),
    ("cvcudaTestStopResourceDestructionTracking", None, None),
    ("cvcudaTestTensorDestroyedBeforeLifetimeRelease", None, ctypes.c_int),
    ("cvcudaTestTrackedTensorCount", None, ctypes.c_int),
    ("cvcudaTestWaitForBatchPush", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForBusyMutexAttempt", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForCriticalSectionHold", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForDelayedFlipRelease", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForDeviceSynchronize", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForDeviceSynchronizeCalls", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForEventRecordCount", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForFirstConcurrentStack", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForFirstFindHomography", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForFirstFlip", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForFirstImageBatchMetadata", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForFirstMorphology", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForFlipReturn", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForGpuGate", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForGpuSignal", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForImageCpuCleanupSync", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForImageCpuCopyFailure", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForImageCpuFirstCopy", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForLifetimeGate", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForLifetimeSubmit", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForReaderEntry", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForSecondConcurrentStack", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForSecondFindHomography", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForSecondFlip", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForSecondImageBatchMetadata", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForSecondMorphology", [ctypes.c_int], ctypes.c_int),
    ("cvcudaTestWaitForStackSubmit", [ctypes.c_int], ctypes.c_int),
]:
    _function = getattr(probe, _name)
    _function.argtypes = _arguments
    _function.restype = _result
"""


def _subprocess_output_text(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output or ""


def _captured_subprocess(
    command: list[str], environment: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    scaled_timeout = timeout * test_util.subprocess_timeout_scale()
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=scaled_timeout,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"subprocess timed out after {scaled_timeout:g} seconds\n"
            f"stdout:\n{_subprocess_output_text(error.stdout)}\n"
            f"stderr:\n{_subprocess_output_text(error.stderr)}",
            pytrace=False,
        )


@cache
def _cvcuda_library_path() -> str:
    # Wheel RPATHs do not apply to LD_PRELOAD. Use the installed libraries
    # already mapped by cvcuda, including auditwheel's hashed filenames.
    try:
        import cvcuda  # noqa: F401
    except (ImportError, OSError):
        return ""

    with open("/proc/self/maps", encoding="utf-8") as maps:
        directories = {
            os.path.dirname(path)
            for line in maps
            if (path := line.rstrip().rsplit(" ", 1)[-1]).startswith("/")
            and ".so" in os.path.basename(path)
            and (
                "cvcuda" in os.path.basename(path)
                or "nvcv_types" in os.path.basename(path)
            )
        }
    return os.pathsep.join(sorted(directories))


def _child_environment(extra_env: dict[str, str] | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CVCUDA_TEST_TIMEOUT_SCALE"] = str(test_util.subprocess_timeout_scale())
    if library_path := _cvcuda_library_path():
        inherited = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{library_path}{os.pathsep}{inherited}" if inherited else library_path
        )
    if extra_env:
        environment.update(extra_env)
    return environment


def _cleanup_module_globals_script(script: str, filename: str) -> str:
    test_script = textwrap.dedent(script)
    return textwrap.dedent(
        f"""\
        import gc as _cvcuda_test_gc

        _cvcuda_test_namespace = {{"__name__": "__main__"}}
        try:
            exec(
                compile({test_script!r}, {filename!r}, "exec"),
                _cvcuda_test_namespace,
            )
        finally:
            _cvcuda_test_namespace.clear()
            _cvcuda_test_gc.collect()
        """
    )


def _run_free_threaded(
    script: str,
    *,
    gil: int | None,
    timeout: int = 30,
    probe: bool = False,
) -> None:
    expect_gil = gil == 1
    preamble = textwrap.dedent(
        f"""
        import sys as _cvcuda_test_sys
        import sysconfig as _cvcuda_test_sysconfig

        _cvcuda_test_sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})

        assert _cvcuda_test_sysconfig.get_config_var("Py_GIL_DISABLED") == 1
        assert _cvcuda_test_sys._is_gil_enabled() is {expect_gil!r}
        """
    )
    test_script = textwrap.dedent(script)
    if probe:
        test_script = _PROBE_SETUP + test_script
    test_script = (
        "from cvcuda_test_utils import as_cupy_stream\n"
        f"_cvcuda_test_timeout_scale = {test_util.subprocess_timeout_scale()!r}\n"
        "def _cvcuda_test_seconds(value):\n"
        "    return value * _cvcuda_test_timeout_scale\n"
        "def _cvcuda_test_milliseconds(value):\n"
        "    return int(value * _cvcuda_test_timeout_scale)\n" + test_script
    )
    # These scripts test live-runtime behavior; release their owners before shutdown.
    test_script = _cleanup_module_globals_script(
        test_script, "<cvcuda-free-threading-repro>"
    )
    epilogue = f"\nassert _cvcuda_test_sys._is_gil_enabled() is {expect_gil!r}\n"
    wrapped_script = preamble + test_script + epilogue
    command = [
        sys.executable,
        "-u",
        "-W",
        "always",
        "-W",
        "error:ExternalStream is deprecated:DeprecationWarning",
        "-c",
        wrapped_script,
    ]
    if gil is not None:
        command[2:2] = ["-X", f"gil={gil}"]
    environment = _child_environment(_probe_environment() if probe else None)
    environment.pop("PYTHON_GIL", None)
    result = _captured_subprocess(
        command,
        environment=environment,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"free-threaded subprocess failed with return code {result.returncode}\n"
        f"command: {' '.join(command[:-1])} <script>\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@cache
def _dlopen_error(library: str) -> str:
    result = _captured_subprocess(
        [sys.executable, "-c", "import ctypes, sys; ctypes.CDLL(sys.argv[1])", library],
        environment=_child_environment(None),
        timeout=10,
    )
    return result.stderr.strip() if result.returncode else ""


def _require_cvcuda_soname() -> None:
    if not (load_error := _dlopen_error("libcvcuda.so.0")):
        return
    if not _dlopen_error("libnvcv_types.so.0"):
        pytest.fail(
            "libcvcuda.so.0 is not loadable despite libnvcv_types.so.0 being "
            f"available: {load_error}"
        )
    pytest.skip(f"libcvcuda.so.0 is not loadable in this install: {load_error}")


def _probe_environment() -> dict[str, str]:
    probe = os.environ.get("CVCUDA_FREE_THREADING_PROBE")
    if not probe:
        pytest.fail("free-threading concurrency probe is required but is not available")
    if not os.path.isfile(probe):
        pytest.fail(f"configured concurrency probe is not a file: {probe}")
    if load_error := _dlopen_error(probe):
        if not _dlopen_error("libnvcv_types.so.0"):
            pytest.fail(
                "concurrency probe is not loadable despite its CV-CUDA dependency "
                f"being available: {load_error}"
            )
        pytest.skip(f"concurrency probe is not loadable in this install: {load_error}")
    preload = probe
    if inherited_preload := os.environ.get("LD_PRELOAD"):
        preload = f"{probe}:{inherited_preload}"
    return {"LD_PRELOAD": preload}


def test_python_and_core_nvtx_first_use_are_independent():
    _require_cvcuda_soname()
    _run_free_threaded(
        """
        import ctypes
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import cvcuda

        thread_count = 16
        library = ctypes.CDLL("libcvcuda.so.0")
        flip_submit = library.cvcudaFlipSubmit
        flip_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        flip_submit.restype = ctypes.c_int
        barrier = threading.Barrier(thread_count)
        results = [None] * thread_count

        def first_use(index):
            barrier.wait()
            if index % 2 == 0:
                results[index] = cvcuda.Stream()
            else:
                results[index] = flip_submit(None, None, None, None, 0)

        with ThreadPoolExecutor(max_workers=thread_count) as workers:
            list(workers.map(first_use, range(thread_count)))
        assert all(results[index] is not None for index in range(0, thread_count, 2))
        assert all(
            isinstance(results[index], int) and results[index] != 0
            for index in range(1, thread_count, 2)
        )
        """,
        gil=0,
    )


@pytest.mark.parametrize(
    "consumer",
    [
        "getter",
        "setter",
        "repr",
        "customcrop",
        "customcrop_into",
        "resize_tensor",
        "resize_varshape",
    ],
)
def test_recti_access_and_operator_consumers_use_object_critical_section(consumer):
    _run_free_threaded(
        f"""
        import numpy as np
        attached_probe = ctypes.PyDLL(None)
        attached_probe.cvcudaTestHoldCriticalSection.argtypes = [ctypes.c_void_p]
        attached_probe.cvcudaTestHoldCriticalSection.restype = None

        consumer = {consumer!r}
        rect = cvcuda.RectI(0, 0, 4, 4)
        stream = cvcuda.Stream()
        tensor = cvcuda.Tensor((1, 8, 8, 3), np.uint8, "NHWC")
        output = cvcuda.Tensor(tensor.shape, tensor.dtype, tensor.layout)
        batch = cvcuda.ImageBatchVarShape(1)
        batch.pushback(cvcuda.Image((8, 8), cvcuda.Format.RGB8))

        def consume(value):
            if consumer == "getter":
                assert value.x == 0
            elif consumer == "setter":
                value.x = 7
            elif consumer == "repr":
                assert repr(value) == "RectI(x=0,y=0,width=4,height=4)"
            elif consumer == "customcrop":
                cvcuda.customcrop(tensor, value, stream=stream)
            elif consumer == "customcrop_into":
                cvcuda.customcrop_into(output, tensor, value, stream=stream)
            elif consumer == "resize_tensor":
                cvcuda.resize_crop_convert_reformat(
                    tensor,
                    (8, 8),
                    cvcuda.Interp.NEAREST,
                    value,
                    layout="NHWC",
                    data_type=cvcuda.Type.U8,
                    stream=stream,
                )
            else:
                cvcuda.resize_crop_convert_reformat(
                    batch,
                    (8, 8),
                    cvcuda.Interp.NEAREST,
                    value,
                    layout="NHWC",
                    data_type=cvcuda.Type.U8,
                    stream=stream,
                )

        # Warm lazy operator and cache initialization so it cannot land inside
        # the critical section the consumer threads below are ordered around.
        if consumer not in ("getter", "setter", "repr"):
            consume(cvcuda.RectI(0, 0, 4, 4))
            stream.sync()

        failures = [None, None]
        consumer_done = threading.Event()

        def hold_rect():
            try:
                attached_probe.cvcudaTestHoldCriticalSection(
                    ctypes.c_void_p(id(rect))
                )
            except BaseException as error:
                failures[0] = error

        def consume_rect():
            try:
                probe.cvcudaTestReachReaderEntry()
                consume(rect)
            except BaseException as error:
                failures[1] = error
            finally:
                consumer_done.set()

        probe.cvcudaTestArmCriticalSectionHold()
        holder = threading.Thread(target=hold_rect)
        consumer_thread = threading.Thread(target=consume_rect)
        holder.start()
        try:
            assert probe.cvcudaTestWaitForCriticalSectionHold(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            consumer_thread.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            consumer_bypassed_critical_section = consumer_done.wait(
                _cvcuda_test_timeout_scale
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseCriticalSectionHold()

        assert consumer_done.wait(_cvcuda_test_seconds(10))
        holder.join(_cvcuda_test_seconds(15))
        if consumer_thread.ident is not None:
            consumer_thread.join(_cvcuda_test_seconds(15))
        assert not holder.is_alive()
        assert not consumer_thread.is_alive()
        assert failures == [None, None], failures
        stream.sync()
        assert not consumer_bypassed_critical_section, (
            consumer + " bypassed the RectI critical section"
        )
        """,
        gil=0,
        timeout=30,
        probe=True,
    )


def test_shared_resource_sync_and_kernel_submission_are_atomic():
    _run_free_threaded(
        """
        import numpy as np
        import cupy as cp

        expected = np.arange(1 * 32 * 32 * 4, dtype=np.uint8).reshape(1, 32, 32, 4)
        src = cvcuda.as_tensor(cp.asarray(expected), "NHWC")
        dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]

        def submit(thread_index):
            try:
                if thread_index == 1:
                    probe.cvcudaTestReachReaderEntry()
                cvcuda.flip_into(
                    src=src,
                    dst=dst,
                    flipCode=0,
                    stream=streams[thread_index],
                )
            except BaseException as error:
                failures[thread_index] = error

        probe.cvcudaTestArmFlipGate()
        first = threading.Thread(target=submit, args=(0,))
        first.start()
        assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestArmReaderEntryGate(1)
        second = threading.Thread(target=submit, args=(1,))
        second.start()
        assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestReleaseReaderEntryGate()
        second_reached_submit = (
            probe.cvcudaTestWaitForSecondFlip(
                int(1_000 * _cvcuda_test_timeout_scale)
            )
            == 1
        )
        probe.cvcudaTestReleaseFlipAndObserveSecond()
        assert probe.cvcudaTestWaitForSecondFlip(_cvcuda_test_milliseconds(10_000)) == 1
        first.join(_cvcuda_test_seconds(15))
        second.join(_cvcuda_test_seconds(15))
        assert not first.is_alive()
        assert not second.is_alive()
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()
        actual = cp.asarray(dst.cuda()).get()
        assert np.array_equal(actual, np.flip(expected, axis=1))
        assert not second_reached_submit
        assert probe.cvcudaTestFlipOverlapObserved() == 0
        """,
        gil=0,
        probe=True,
    )


def test_resource_guard_preserves_parallelism_for_compatible_resources():
    _run_free_threaded(
        """
        import numpy as np
        import cupy as cp
        def run_pair(sources, destinations, flip_codes):
            streams = [cvcuda.Stream(), cvcuda.Stream()]
            failures = [None, None]

            def submit(index):
                try:
                    cvcuda.flip_into(
                        src=sources[index],
                        dst=destinations[index],
                        flipCode=flip_codes[index],
                        stream=streams[index],
                    )
                except BaseException as error:
                    failures[index] = error

            probe.cvcudaTestArmFlipGate()
            first = threading.Thread(target=submit, args=(0,))
            first.start()
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            second = threading.Thread(target=submit, args=(1,))
            second.start()
            second_reached_submit = probe.cvcudaTestWaitForSecondFlip(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseFlip()
            first.join(_cvcuda_test_seconds(15))
            second.join(_cvcuda_test_seconds(15))
            assert not first.is_alive()
            assert not second.is_alive()
            assert failures == [None, None], failures
            assert second_reached_submit, "compatible resource sets were globally serialized"
            assert probe.cvcudaTestFlipOverlapObserved() == 1
            for stream in streams:
                stream.sync()

        host_a = np.arange(1 * 16 * 16, dtype=np.uint8).reshape(1, 16, 16, 1)
        host_b = np.bitwise_xor(host_a, np.uint8(0x5A))

        # Completely disjoint reads and writes must not be serialized by a
        # process-wide fallback lock.
        disjoint_sources = [
            cvcuda.as_tensor(cp.asarray(host_a), "NHWC"),
            cvcuda.as_tensor(cp.asarray(host_b), "NHWC"),
        ]
        disjoint_destinations = [
            cvcuda.Tensor(disjoint_sources[0].shape, np.uint8, "NHWC"),
            cvcuda.Tensor(disjoint_sources[1].shape, np.uint8, "NHWC"),
        ]
        run_pair(disjoint_sources, disjoint_destinations, [0, 1])
        assert np.array_equal(
            cp.asarray(disjoint_destinations[0].cuda()).get(), np.flip(host_a, axis=1)
        )
        assert np.array_equal(
            cp.asarray(disjoint_destinations[1].cuda()).get(), np.flip(host_b, axis=2)
        )

        # Two readers of the same input with disjoint outputs are compatible
        # and must retain the same parallelism.
        shared_source = cvcuda.as_tensor(cp.asarray(host_a), "NHWC")
        shared_read_destinations = [
            cvcuda.Tensor(shared_source.shape, np.uint8, "NHWC"),
            cvcuda.Tensor(shared_source.shape, np.uint8, "NHWC"),
        ]
        run_pair([shared_source, shared_source], shared_read_destinations, [0, 1])
        assert np.array_equal(
            cp.asarray(shared_read_destinations[0].cuda()).get(), np.flip(host_a, axis=1)
        )
        assert np.array_equal(
            cp.asarray(shared_read_destinations[1].cuda()).get(), np.flip(host_a, axis=2)
        )
        """,
        gil=0,
        probe=True,
    )


def test_same_handle_reader_aliases_remain_tracked_until_each_submission_retires():
    _run_free_threaded(
        """
        import gc
        import numpy as np
        import cupy as cp

        delay_kernel = cp.RawKernel(
            (
                'extern "C" __global__ void cvcuda_test_reader_alias_delay('
                "unsigned long long cycles) {"
                "unsigned long long start = clock64();"
                "while (clock64() - start < cycles) {}"
                "}"
            ),
            "cvcuda_test_reader_alias_delay",
        )

        old_value = np.uint8(17)
        new_value = np.uint8(91)
        shape = (1, 64, 64, 1)
        source = cvcuda.Tensor(shape, np.uint8, "NHWC")
        replacement = cvcuda.Tensor(shape, np.uint8, "NHWC")
        first_output = cvcuda.Tensor(shape, np.uint8, "NHWC")
        second_output = cvcuda.Tensor(shape, np.uint8, "NHWC")
        cp.asarray(source.cuda()).fill(old_value)
        cp.asarray(replacement.cuda()).fill(new_value)
        cp.cuda.runtime.deviceSynchronize()

        underlying = cp.cuda.Stream(non_blocking=True)
        streams = [cvcuda.as_stream(underlying), cvcuda.as_stream(underlying)]
        assert streams[0] is not streams[1]
        assert streams[0].handle == streams[1].handle == underlying.ptr

        # Compile the delay before either submission reaches the probe.
        with underlying:
            delay_kernel((1,), (1,), (1,))
        underlying.synchronize()

        failures = [None, None]

        def read(index):
            try:
                cvcuda.flip_into(
                    dst=first_output if index == 0 else second_output,
                    src=source,
                    flipCode=0,
                    stream=streams[index],
                )
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmDetachedFlipGate()
        first = threading.Thread(target=read, args=(0,))
        second = threading.Thread(target=read, args=(1,))
        first.start()
        try:
            # The first reader has registered its stream owner and still holds
            # the shared submission lock, but its kernel is not enqueued yet.
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            second.start()
            assert probe.cvcudaTestWaitForSecondFlip(_cvcuda_test_milliseconds(10_000)) == 1
            second.join(_cvcuda_test_seconds(15))
            assert not second.is_alive()
            assert failures[1] is None, failures[1]

            # The second alias has enqueued and retired before the first alias.
            # Dropping it must not make the still-live first reader disappear
            # from the resource's synchronization state.
            underlying.synchronize()
            streams[1] = None
            second_output = None
            cvcuda.internal.drain_pending_releases()
            cvcuda.clear_cache()
            gc.collect()

            # Queue the first read behind a delay. A conflicting writer on a
            # different stream must wait for that read rather than overwrite
            # its source while the delay is pending.
            with underlying:
                delay_kernel((1,), (1,), (300_000_000,))
        finally:
            probe.cvcudaTestReleaseFlip()

        first.join(_cvcuda_test_seconds(15))
        assert not first.is_alive()
        assert failures[0] is None, failures[0]

        writer = cvcuda.Stream()
        cvcuda.flip_into(
            dst=source,
            src=replacement,
            flipCode=0,
            stream=writer,
        )
        writer.sync()
        streams[0].sync()
        cvcuda.internal.drain_pending_releases()

        observed = cp.asarray(first_output.cuda()).get()
        assert np.all(observed == old_value), (
            "a same-handle reader alias replaced the still-live owner and let "
            "a conflicting writer bypass its pending read"
        )
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def _allocating_batch_metadata_prelude_script(detach_first_waiter: bool) -> str:
    return f"""

        batch = cvcuda.ImageBatchVarShape(1)
        batch.pushback(cvcuda.Image((16, 16), cvcuda.Format.RGB8))
        stream = cvcuda.Stream()
        failures = [None, None]

        def allocate_and_submit():
            try:
                cvcuda.stack(batch, stream=stream)
            except BaseException as error:
                failures[0] = error

        def mutate():
            try:
                probe.cvcudaTestArmNextBusyMutexAttempt()
                batch.clear()
            except BaseException as error:
                failures[1] = error

        probe.cvcudaTestArmImageBatchMetadataGate({int(detach_first_waiter)})
        probe.cvcudaTestArmBatchMutation(6)
        submitter = threading.Thread(target=allocate_and_submit)
        mutator = threading.Thread(target=mutate)
        submitter.start()
        try:
            assert probe.cvcudaTestWaitForFirstImageBatchMetadata(_cvcuda_test_milliseconds(10_000)) == 1
            mutator.start()
            assert probe.cvcudaTestWaitForBusyMutexAttempt(_cvcuda_test_milliseconds(10_000)) == 1
            mutation_reached_during_prelude = bool(
                probe.cvcudaTestWaitForBatchPush(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseBusyMutexAttempt()
            probe.cvcudaTestReleaseImageBatchMetadata()
            probe.cvcudaTestReleaseBatchPush()

        submitter.join(_cvcuda_test_seconds(15))
        if mutator.ident is not None:
            mutator.join(_cvcuda_test_seconds(15))
        assert not submitter.is_alive()
        assert not mutator.is_alive()
        assert not mutation_reached_during_prelude, (
            "batch mutation changed native metadata while an allocating operator "
            "was deriving its output"
        )
        assert failures == [None, None], failures
        stream.sync()
        """


def test_allocating_batch_operator_holds_structure_during_metadata_prelude():
    _run_free_threaded(
        _allocating_batch_metadata_prelude_script(False),
        gil=0,
        timeout=45,
        probe=True,
    )


@pytest.mark.parametrize("metadata", ["uniqueformat", "maxsize"])
def test_image_batch_lazy_metadata_cache_reads_are_exclusive(metadata):
    _run_free_threaded(
        f"""

        metadata = {metadata!r}
        batch = cvcuda.ImageBatchVarShape(2)
        batch.pushback(
            [
                cvcuda.Image((16, 16), cvcuda.Format.RGB8),
                cvcuda.Image((8, 8), cvcuda.Format.RGB8),
            ]
        )
        if metadata == "uniqueformat":
            batch.clear()
        else:
            batch.popback()

        failures = [None, None]
        unobserved = object()
        observations = [unobserved, unobserved]

        def read(index):
            try:
                if index == 1:
                    probe.cvcudaTestReachReaderEntry()
                observations[index] = getattr(batch, metadata)
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmImageBatchMetadataGate(0)
        first = threading.Thread(target=read, args=(0,))
        second = threading.Thread(target=read, args=(1,))
        first.start()
        try:
            assert probe.cvcudaTestWaitForFirstImageBatchMetadata(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            second.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            second_reached_cache = (
                probe.cvcudaTestWaitForSecondImageBatchMetadata(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
                == 1
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseImageBatchMetadataAndObserveSecond()

        assert probe.cvcudaTestWaitForSecondImageBatchMetadata(_cvcuda_test_milliseconds(10_000)) == 1

        first.join(_cvcuda_test_seconds(15))
        if second.ident is not None:
            second.join(_cvcuda_test_seconds(15))
        assert not first.is_alive()
        assert not second.is_alive()
        assert failures == [None, None], failures
        if metadata == "uniqueformat":
            assert observations == [None, None]
        else:
            assert observations == [(16, 16), (16, 16)]
        assert not second_reached_cache, "lazy image-batch metadata cache was read concurrently"
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_resource_guard_conflicts_aliases_and_error_paths_do_not_deadlock():
    _run_free_threaded(
        """
        import numpy as np
        import cupy as cp

        first_tensor = cvcuda.Tensor((1, 16, 16, 1), np.uint8, "NHWC")
        second_tensor = cvcuda.Tensor(first_tensor.shape, first_tensor.dtype, first_tensor.layout)
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]

        def submit(index):
            try:
                if index == 1:
                    probe.cvcudaTestReachReaderEntry()
                source, destination = (
                    (first_tensor, second_tensor)
                    if index == 0
                    else (second_tensor, first_tensor)
                )
                cvcuda.flip_into(
                    src=source,
                    dst=destination,
                    flipCode=0,
                    stream=streams[index],
                )
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmFlipGate()
        first = threading.Thread(target=submit, args=(0,))
        first.start()
        assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestArmReaderEntryGate(1)
        second = threading.Thread(target=submit, args=(1,))
        second.start()
        assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestReleaseReaderEntryGate()
        second_reached_submit = (
            probe.cvcudaTestWaitForSecondFlip(
                int(1_000 * _cvcuda_test_timeout_scale)
            )
            == 1
        )
        probe.cvcudaTestReleaseFlipAndObserveSecond()
        assert probe.cvcudaTestWaitForSecondFlip(_cvcuda_test_milliseconds(10_000)) == 1
        first.join(_cvcuda_test_seconds(15))
        second.join(_cvcuda_test_seconds(15))
        assert not first.is_alive()
        assert not second.is_alive()
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()

        # The region overload explicitly supports in-place erase. Registering
        # one object as both input and output must coalesce its lock mode rather
        # than trying to acquire the same lock twice.
        alias = cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
        values = cvcuda.Tensor((1, 2, 3), np.float32, "CHW")
        cp.asarray(alias.cuda()).fill(4)
        cp.asarray(values.cuda()).fill(7)
        cp.cuda.get_current_stream().synchronize()
        alias_stream = cvcuda.Stream()
        alias_failure = []

        def submit_alias():
            try:
                cvcuda.erase_into(
                    alias,
                    alias,
                    2,
                    3,
                    2,
                    3,
                    values,
                    stream=alias_stream,
                )
                alias_stream.sync()
            except BaseException as error:
                alias_failure.append(error)

        alias_worker = threading.Thread(target=submit_alias)
        alias_worker.start()
        alias_worker.join(_cvcuda_test_seconds(15))
        assert not alias_worker.is_alive(), "aliased resource acquisition deadlocked"
        assert alias_failure == [], alias_failure
        alias_result = cp.asarray(alias.cuda()).get()
        alias_expected = np.full_like(alias_result, 4)
        alias_expected[:, 2:4, 3:6, :] = 7
        assert np.array_equal(alias_result, alias_expected)

        # A submission exception must release all acquired host-side locks.
        invalid_src = cvcuda.Tensor((1, 8, 8, 1), np.int16, "NHWC")
        invalid_dst = cvcuda.Tensor(invalid_src.shape, invalid_src.dtype, invalid_src.layout)
        errors = []

        def submit_invalid():
            try:
                cvcuda.flip_into(src=invalid_src, dst=invalid_dst, flipCode=0)
            except BaseException as error:
                errors.append(error)

        for _ in range(2):
            worker = threading.Thread(target=submit_invalid)
            worker.start()
            worker.join(_cvcuda_test_seconds(15))
            assert not worker.is_alive(), "submission error leaked a resource lock"
        assert len(errors) == 2
        assert all(
            "NVCV_ERROR_INVALID_ARGUMENT" in str(error)
            and "Invalid data type" in str(error)
            for error in errors
        ), errors

        assert not second_reached_submit, "opposite write acquisition orders overlapped"
        """,
        gil=0,
        probe=True,
    )


def test_dlpack_writer_snapshot_and_consumer_wait_are_atomic():
    script = """


        tensor = cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC")
        producer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        tensor.submitStreamSync(producer)
        producer.sync()
        exported = tensor.cuda()
        cvcuda.internal.drain_pending_releases()

        capsules = []
        failures = []

        def export_dlpack():
            try:
                capsules.append(exported.__dlpack__(stream=int(consumer.handle)))
            except BaseException as error:
                failures.append(error)

        exporter = threading.Thread(target=export_dlpack)
        probe.cvcudaTestArmEventRecordGate()
        try:
            exporter.start()
            assert probe.cvcudaTestWaitForEventRecordCount(1, _cvcuda_test_milliseconds(10_000)) == 1
            export_still_owns_submission = not cvcuda._test.try_lock_submission(
                tensor
            )
        finally:
            probe.cvcudaTestReleaseEventRecordGate()

        exporter.join(_cvcuda_test_seconds(10))
        assert not exporter.is_alive()
        assert failures == [], failures
        assert len(capsules) == 1
        assert export_still_owns_submission, (
            "DLPack released the producer submission lock before installing "
            "the consumer wait"
        )
        consumer.sync()
        capsules.clear()
    """
    _run_free_threaded(script, gil=0, timeout=35, probe=True)


def test_exception_records_per_thread_completion_before_unlocking_submission():
    _run_free_threaded(
        """
        tensor = cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC")
        failures = []

        def fail_after_enqueue():
            probe.cvcudaTestArmEventRecordGate()
            raise RuntimeError("submission failed")

        def submit():
            try:
                stream = cvcuda.as_stream(2)
                try:
                    cvcuda._test.native_resourceguard_run(tensor, stream, fail_after_enqueue)
                except RuntimeError as error:
                    assert str(error) == "submission failed"
                else:
                    raise AssertionError("submission error was lost")
                stream.sync()
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=submit)
        worker.start()
        try:
            assert probe.cvcudaTestWaitForEventRecordCount(1, _cvcuda_test_milliseconds(10_000))
            locked_at_completion = not cvcuda._test.try_lock_submission(tensor)
        finally:
            probe.cvcudaTestReleaseEventRecordGate()
        worker.join(_cvcuda_test_seconds(10))
        assert not worker.is_alive()
        assert failures == [], failures
        assert locked_at_completion, "exception released submission before recording PTDS completion"
        """,
        gil=0,
        timeout=30,
        probe=True,
    )


def test_per_thread_import_preserves_pending_alias_dependencies():
    _run_free_threaded(
        """
        import cupy as cp
        for name in ("cvcudaTestQueueGpuGate", "cvcudaTestQueueGpuSignal"):
            getattr(probe, name).argtypes = [ctypes.c_void_p]
            getattr(probe, name).restype = ctypes.c_int
        for name in ("cvcudaTestWaitForGpuGate", "cvcudaTestWaitForGpuSignal"):
            getattr(probe, name).argtypes = [ctypes.c_int]
            getattr(probe, name).restype = ctypes.c_int

        class Producer:
            def __init__(self, array, stream):
                self.array, self.stream = array, stream

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["stream"] = self.stream
                return interface

        array = cp.zeros((1, 8, 8, 1), dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        original = cvcuda.as_tensor(Producer(array, -1), "NHWC")
        writer, consumer = cvcuda.Stream(), cvcuda.Stream()
        original.submitStreamSync(writer)
        writer.sync()
        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestArmGpuSignalGate()
        try:
            alias = cvcuda.as_tensor(Producer(array, 2), "NHWC")
            alias.submitStreamSync(consumer)
            assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            early = probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(1_000))
        finally:
            probe.cvcudaTestReleaseGpuGate()
        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()
        writer.sync()
        consumer.sync()
        cp.cuda.Stream.ptds.synchronize()
        assert not early, "PTDS import lost the allocation's earlier writer dependency"
        """,
        gil=0,
        timeout=35,
        probe=True,
    )


def test_import_retains_completion_after_external_producer_stream_is_destroyed():
    _run_free_threaded(
        """
        import cupy as cp
        import cvcuda

        handle = cp.cuda.runtime.streamCreateWithFlags(1)

        class Producer:
            def __init__(self, array):
                self.array = array

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["stream"] = handle
                return interface

        try:
            with as_cupy_stream(handle):
                array = cp.full((1, 16, 16, 1), 37, dtype=cp.uint8)
                tensor = cvcuda.as_tensor(Producer(array), "NHWC")
        finally:
            cp.cuda.runtime.streamDestroy(handle)
        consumer = cvcuda.Stream()
        output = cvcuda.flip(tensor, flipCode=0, stream=consumer)
        consumer.sync()
        assert cp.all(cp.asarray(output.cuda()) == 37).item()
        # Re-exporting the imported source also needs a live advertised stream.
        assert cp.all(cp.asarray(tensor.cuda()) == 37).item()
        """,
        gil=0,
        timeout=30,
    )


def test_dlpack_export_keeps_reader_stream_alive_until_event_record_returns():
    _run_free_threaded(
        """
        import gc

        import cupy as cp

        producer = cvcuda.Stream()
        owner = cp.cuda.Stream(non_blocking=True)
        reader = cvcuda.as_stream(owner)
        src = cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC")
        dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
        src.submitStreamSync(producer)
        cvcuda.flip_into(dst, src, flipCode=0, stream=reader)
        # Remove the output's writer reference, leaving only src's weak reader.
        dst.submitStreamSync(producer)
        producer.sync()
        reader.sync()
        cvcuda.internal.drain_pending_releases()
        exported = src.cuda()
        capsules = []
        failures = []

        def export_dlpack():
            try:
                # The writer is already on producer, so the gated record is
                # the dependency on reader, not the previous writer.
                capsules.append(exported.__dlpack__(stream=int(producer.handle)))
            except BaseException as error:
                failures.append(error)

        exporter = threading.Thread(target=export_dlpack)
        probe.cvcudaTestArmEventRecordEntryGate()
        try:
            exporter.start()
            assert probe.cvcudaTestWaitForEventRecordCount(1, _cvcuda_test_milliseconds(10_000)) == 1
            del reader, owner
            gc.collect()
        finally:
            probe.cvcudaTestReleaseEventRecordGate()
        exporter.join(_cvcuda_test_seconds(10))
        assert not exporter.is_alive()
        assert failures == [], failures
        assert len(capsules) == 1
        producer.sync()
        capsules.clear()
        """,
        gil=0,
        timeout=35,
        probe=True,
    )


@pytest.mark.parametrize(
    "owner_kind,precreated",
    [("tensor", False), ("tensor", True), ("image", True), ("image_batch", False)],
)
@pytest.mark.parametrize("export_kind", ["cai", "dlpack"])
def test_exports_wait_for_writer_kernel_enqueue(owner_kind, precreated, export_kind):
    _run_free_threaded(
        f"""
        import numpy as np
        import cupy as cp

        owner_kind = {owner_kind!r}
        host = np.arange(16 * 16 * 4, dtype=np.uint8).reshape(16, 16, 4)
        source = cvcuda.as_tensor(cp.asarray(host), "HWC")
        if owner_kind == "tensor":
            owner = cvcuda.Tensor(source.shape, source.dtype, source.layout)
            write_target = owner
        else:
            owner = cvcuda.Image((16, 16), cvcuda.Format.RGBA8)
            write_target = cvcuda.as_tensor(owner)
        if owner_kind == "image":
            source = source.reshape(write_target.shape, write_target.layout)
        cp.asarray(write_target.cuda()).fill(0)
        if owner_kind == "image_batch":
            source_image = cvcuda.as_image(source.cuda(), cvcuda.Format.RGBA8)
            source = cvcuda.ImageBatchVarShape(1)
            source.pushback(source_image)
            write_target = cvcuda.ImageBatchVarShape(1)
            write_target.pushback(owner)
            flip_code = cvcuda.as_tensor(cp.zeros((1,), dtype=cp.int32), "N")
        else:
            flip_code = 0
        cp.cuda.get_current_stream().synchronize()

        # Exercise buffers captured both before and during the writer's enqueue.
        exported = owner.cuda() if {precreated!r} else None
        producer_stream = cvcuda.Stream()
        consumer_stream = cvcuda.Stream()
        consumed = cp.empty_like(cp.asarray(host))
        failures = [None, None]
        consumer_done = threading.Event()

        def produce():
            try:
                cvcuda.flip_into(
                    dst=write_target, src=source, flipCode=flip_code, stream=producer_stream
                )
            except BaseException as error:
                failures[0] = error

        def consume():
            try:
                probe.cvcudaTestReachReaderEntry()
                with as_cupy_stream(consumer_stream.handle):
                    buffer = exported if {precreated!r} else owner.cuda()
                    view = cp.asarray(buffer) if {export_kind!r} == "cai" else cp.from_dlpack(buffer)
                    cp.copyto(consumed, view)
                consumer_stream.sync()
            except BaseException as error:
                failures[1] = error
            finally:
                consumer_done.set()

        probe.cvcudaTestArmFlipGate()
        producer = threading.Thread(target=produce)
        consumer = threading.Thread(target=consume)
        producer.start()
        try:
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            consumer.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            bypassed_enqueue = consumer_done.wait(_cvcuda_test_seconds(1))
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseFlip()

        producer.join(_cvcuda_test_seconds(15))
        consumer.join(_cvcuda_test_seconds(15))
        assert not producer.is_alive()
        assert not consumer.is_alive()
        assert failures == [None, None], failures
        producer_stream.sync()
        consumer_stream.sync()
        assert not bypassed_enqueue, "export bypassed the writer's kernel enqueue"
        assert np.array_equal(consumed.get(), np.flip(host, axis=0))
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_multiplane_image_waits_for_every_cai_producer_stream():
    _run_free_threaded(
        """
        import cupy as cp

        class ProducerPlane:
            def __init__(self, array, stream):
                self.array = array
                self.stream = stream

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["stream"] = int(self.stream.ptr)
                return interface

        planes = [
            cp.zeros((6, 8), dtype=cp.uint8),
            cp.zeros((3, 4, 2), dtype=cp.uint8),
        ]
        cp.cuda.get_current_stream().synchronize()
        producers = [
            cp.cuda.Stream(non_blocking=True),
            cp.cuda.Stream(non_blocking=True),
        ]
        wrapped_planes = [
            ProducerPlane(array, stream)
            for array, stream in zip(planes, producers)
        ]
        consumer = cvcuda.Stream()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(producers[1].ptr) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        signal_reached_before_release = False
        probe.cvcudaTestArmGpuSignalGate()
        try:
            image = cvcuda.as_image(wrapped_planes, cvcuda.Format.NV12_ER)
            image.submitStreamSync(consumer)
            assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        for stream in producers:
            stream.synchronize()
        consumer.sync()
        assert not signal_reached_before_release, (
            "multi-plane Image became ready before every advertised CAI "
            "producer stream completed"
        )
        """,
        gil=0,
        timeout=30,
        probe=True,
    )


def test_external_buffer_preserves_readonly_cuda_array_interface_flag():
    _run_free_threaded(
        """
        import cupy as cp
        import cvcuda

        class ReadOnlyProducer:
            def __init__(self, array):
                self.array = array

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["data"] = (interface["data"][0], True)
                return interface

        producer = ReadOnlyProducer(cp.zeros((8, 8, 4), dtype=cp.uint8))
        image = cvcuda.as_image(producer, cvcuda.Format.RGBA8)
        exported = image.cuda()
        assert exported.__cuda_array_interface__["data"][1] is True, (
            "wrapping and re-exporting a read-only CAI buffer made it writable"
        )
        """,
        gil=0,
    )


def test_direct_external_buffer_rewrap_waits_for_latest_writer_stream():
    _run_free_threaded(
        """

        output = cvcuda.Tensor((1, 8, 8, 1), cvcuda.Type.U8, "NHWC")
        writer = cvcuda.Stream()
        consumer = cvcuda.Stream()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        signal_reached_before_release = False
        probe.cvcudaTestArmGpuSignalGate()
        try:
            output.submitStreamSync(writer)
            exported = output.cuda()
            alias = cvcuda.as_tensor(exported, "NHWC")
            alias.submitStreamSync(consumer)
            assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        writer.sync()
        consumer.sync()
        assert not signal_reached_before_release, (
            "a direct ExternalBuffer rewrap ignored its latest writer stream"
        )
        """,
        gil=0,
        timeout=30,
        probe=True,
    )


def test_per_thread_cai_producer_does_not_synchronize_unrelated_streams():
    script = """

        import cupy as cp

        class Producer:
            def __init__(self, array, sentinel):
                self.array = array
                self.sentinel = sentinel

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                if self.sentinel is None:
                    interface.pop("stream", None)
                else:
                    interface["stream"] = self.sentinel
                return interface

        array = cp.zeros((1, 8, 8, 1), dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()

        # Only the per-thread sentinel still avoids a device-wide wait. An
        # absent field, 0 and 1 all map to cudaStreamLegacy, and
        # bindSyncDomains deliberately synchronizes the producer device for
        # that sentinel: a non-blocking producer has no implicit ordering
        # against the legacy stream, so an event recorded on it would not
        # capture the producer's work. That behaviour is asserted by
        # test_advertised_legacy_sentinel_synchronizes_non_blocking_producer
        # in test_cai_input_stream_race.py.
        for sentinel in (2,):
            unrelated = cp.cuda.Stream(non_blocking=True)
            finished = threading.Event()
            failures = []
            wrapped = []

            def wrap():
                try:
                    wrapped.append(cvcuda.as_tensor(Producer(array, sentinel), "NHWC"))
                except BaseException as error:
                    failures.append(error)
                finally:
                    finished.set()

            probe.cvcudaTestArmGpuGate()
            assert probe.cvcudaTestQueueGpuGate(unrelated.ptr) == 1
            assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
            worker = threading.Thread(target=wrap)
            try:
                worker.start()
                wrapped_without_device_sync = finished.wait(_cvcuda_test_seconds(15))
            finally:
                probe.cvcudaTestReleaseGpuGate()

            worker.join(_cvcuda_test_seconds(10))
            assert not worker.is_alive()
            assert failures == [], failures
            assert wrapped_without_device_sync, (
                f"CAI stream sentinel {sentinel!r} synchronized an unrelated "
                "non-blocking stream on the host"
            )
            unrelated.synchronize()
    """
    _run_free_threaded(
        script,
        gil=0,
        timeout=40,
        probe=True,
    )


def test_bind_sync_domains_does_not_hold_state_lock_during_device_sync():
    script = """

        import cupy as cp

        class LegacyProducer:
            def __init__(self, array):
                self.array = array

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["stream"] = 1
                return interface

        array = cp.zeros((1, 8, 8, 1), dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        buffer = cvcuda._test.load_external_buffer(LegacyProducer(array))
        wrapped = []
        exported_states = []
        failures = []
        reader_done = threading.Event()

        def bind_domains():
            try:
                wrapped.append(cvcuda.as_tensor(buffer, "NHWC"))
            except BaseException as error:
                failures.append(("bind", error))

        def read_state():
            try:
                exported_states.append(buffer.__cuda_array_interface__)
            except BaseException as error:
                failures.append(("read", error))
            finally:
                reader_done.set()

        binder = threading.Thread(target=bind_domains)
        reader = threading.Thread(target=read_state)
        probe.cvcudaTestArmDeviceSynchronizeGate()
        binder.start()
        try:
            assert probe.cvcudaTestWaitForDeviceSynchronize(_cvcuda_test_milliseconds(10_000)) == 1
            reader.start()
            state_read_while_bind_waited = reader_done.wait(_cvcuda_test_seconds(15))
        finally:
            probe.cvcudaTestReleaseDeviceSynchronizeGate()

        binder.join(_cvcuda_test_seconds(15))
        reader.join(_cvcuda_test_seconds(15))
        assert not binder.is_alive()
        assert not reader.is_alive()
        assert failures == [], failures
        assert state_read_while_bind_waited, (
            "bindSyncDomains held the buffer state lock across device synchronization"
        )
        assert len(wrapped) == 1
        assert len(exported_states) == 1
    """
    _run_free_threaded(
        script,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_concurrent_image_wrappers_share_the_published_buffer_domain():
    script = """

        import cupy as cp

        class LegacyProducer:
            def __init__(self, array):
                self.array = array

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["stream"] = 1
                return interface

        array = cp.zeros((8, 8, 4), dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        buffer = cvcuda._test.load_external_buffer(LegacyProducer(array))
        images = [None, None]
        failures = []

        def wrap(index):
            try:
                images[index] = cvcuda.as_image(buffer, cvcuda.Format.RGBA8)
            except BaseException as error:
                failures.append((index, error))

        first = threading.Thread(target=wrap, args=(0,))
        second = threading.Thread(target=wrap, args=(1,))
        probe.cvcudaTestArmDeviceSynchronizeGate()
        first.start()
        try:
            assert probe.cvcudaTestWaitForDeviceSynchronize(_cvcuda_test_milliseconds(10_000)) == 1
            second.start()
            assert probe.cvcudaTestWaitForDeviceSynchronizeCalls(
                2, _cvcuda_test_milliseconds(10_000)
            ) == 1
        finally:
            probe.cvcudaTestReleaseBusyMutexAttempt()
            probe.cvcudaTestReleaseDeviceSynchronizeGate()

        first.join(_cvcuda_test_seconds(15))
        second.join(_cvcuda_test_seconds(15))
        assert not first.is_alive()
        assert not second.is_alive()
        assert failures == [], failures
        assert all(image is not None for image in images)

        writer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        probe.cvcudaTestArmGpuSignalGate()
        try:
            images[0].submitStreamSync(writer)
            images[1].submitStreamSync(consumer)
            assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        writer.sync()
        consumer.sync()
        assert not signal_reached_before_release, (
            "concurrent wrappers retained disjoint synchronization domains"
        )
    """
    _run_free_threaded(
        script,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_fresh_external_producer_is_merged_with_existing_writer():
    script = """

        import cupy as cp

        class Producer:
            def __init__(self, array, stream):
                self.array = array
                self.stream = stream

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["stream"] = self.stream
                return interface

        array = cp.zeros((1, 8, 8, 1), dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        first = cvcuda.as_tensor(Producer(array, -1), "NHWC")
        prior_writer = cvcuda.Stream()
        first.submitStreamSync(prior_writer)
        prior_writer.sync()

        fresh_producer = cp.cuda.Stream(non_blocking=True)
        consumer = cvcuda.Stream()
        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(fresh_producer.ptr) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        signal_reached_before_release = False
        probe.cvcudaTestArmGpuSignalGate()
        try:
            fresh = cvcuda.as_tensor(
                Producer(array, int(fresh_producer.ptr)), "NHWC"
            )
            fresh.submitStreamSync(consumer)
            assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        fresh_producer.synchronize()
        consumer.sync()
        assert not signal_reached_before_release, (
            "a fresh external producer was dropped because the allocation "
            "already retained a CV-CUDA writer"
        )
    """
    _run_free_threaded(
        script,
        gil=0,
        timeout=35,
        probe=True,
    )


@pytest.mark.parametrize(
    "source_kind",
    ["tensor", "multiplane_image", "independent_multiplane_tensors"],
)
def test_image_rewrap_observes_future_writes_through_exported_buffers(
    source_kind,
):
    _run_free_threaded(
        f"""

        source_kind = {source_kind!r}
        if source_kind == "tensor":
            source = cvcuda.Tensor((8, 8, 4), cvcuda.Type.U8, "HWC")
            exported = source.cuda()
            alias = cvcuda.as_image(exported, cvcuda.Format.RGBA8)
        elif source_kind == "multiplane_image":
            source = cvcuda.Image((8, 6), cvcuda.Format.NV12_ER)
            exported = source.cuda()
            assert isinstance(exported, list) and len(exported) == 2
            alias = cvcuda.as_image(exported, cvcuda.Format.NV12_ER)
        else:
            source = cvcuda.Tensor((6, 8, 1), cvcuda.Type.U8, "HWC")
            chroma = cvcuda.Tensor((3, 4, 2), cvcuda.Type.U8, "HWC")
            exported = [source.cuda(), chroma.cuda()]
            alias = cvcuda.as_image(exported, cvcuda.Format.NV12_ER)

        writer = cvcuda.Stream()
        consumer = cvcuda.Stream()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        signal_reached_before_release = False
        probe.cvcudaTestArmGpuSignalGate()
        try:
            source.submitStreamSync(writer)
            alias.submitStreamSync(consumer)
            assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        writer.sync()
        consumer.sync()
        assert not signal_reached_before_release, (
            source_kind
            + " -> Image rewrap ignored a write submitted after export"
        )
        """,
        gil=0,
        timeout=30,
        probe=True,
    )


def test_image_rewrap_joins_existing_multiplane_alias_domains():
    script = """
        import numpy as np

        planes = [
            cvcuda.Tensor((6, 8, 1), np.uint8, "HWC"),
            cvcuda.Tensor((3, 4, 2), np.uint8, "HWC"),
        ]
        multiplane = cvcuda.as_image(
            [plane.cuda() for plane in planes], cvcuda.Format.NV12_ER
        )
        exported = multiplane.cuda("HWC")
        assert isinstance(exported, list) and len(exported) == 2

        alias = cvcuda.as_image(exported[0], cvcuda.Format.U8)
        writer = cvcuda.Stream()
        consumer = cvcuda.Stream()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        signal_reached_before_release = False
        probe.cvcudaTestArmGpuSignalGate()
        try:
            planes[1].submitStreamSync(writer)
            alias.submitStreamSync(consumer)
            assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        writer.sync()
        consumer.sync()
        assert not signal_reached_before_release, (
            "Image rewrap discarded synchronization domains already attached "
            "to an exported plane"
        )
    """
    _run_free_threaded(script, gil=0, timeout=30, probe=True)


@pytest.mark.parametrize("plane_index", [0, 1])
@pytest.mark.parametrize("writer_timing", ["before_export", "after_export"])
@pytest.mark.parametrize("export_kind", ["cai", "dlpack"])
def test_explicit_layout_multiplane_image_export_preserves_plane_ordering(
    plane_index,
    writer_timing,
    export_kind,
):
    _run_free_threaded(
        f"""
        import cupy as cp

        plane_index = {plane_index}
        writer_timing = {writer_timing!r}
        export_kind = {export_kind!r}
        source_planes = [
            cvcuda.Tensor((6, 8, 1), cvcuda.Type.U8, "HWC"),
            cvcuda.Tensor((3, 4, 2), cvcuda.Type.U8, "HWC"),
        ]
        image = cvcuda.as_image(
            [plane.cuda() for plane in source_planes],
            cvcuda.Format.NV12_ER,
        )
        writer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        consumer_external_stream = as_cupy_stream(consumer.handle)

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        if writer_timing == "before_export":
            source_planes[plane_index].submitStreamSync(writer)
        exported_planes = image.cuda("HWC")
        assert isinstance(exported_planes, list) and len(exported_planes) == 2
        if writer_timing == "after_export":
            source_planes[plane_index].submitStreamSync(writer)

        failures = []
        consumer_done = threading.Event()

        def consume():
            try:
                if export_kind == "cai":
                    # CAI consumers may synchronize the producer on the host.
                    # Probe the advertised stream so that behavior cannot make
                    # the ordering oracle vacuous.
                    advertised_stream = exported_planes[
                        plane_index
                    ].__cuda_array_interface__["stream"]
                    assert advertised_stream not in (None, 0, 1, 2, -1)
                    assert probe.cvcudaTestQueueGpuSignal(advertised_stream) == 1
                else:
                    with consumer_external_stream:
                        cp.from_dlpack(exported_planes[plane_index])
                    assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            except BaseException as error:
                failures.append(error)
            finally:
                consumer_done.set()

        probe.cvcudaTestArmGpuSignalGate()
        consumer_thread = threading.Thread(target=consume)
        consumer_thread.start()
        try:
            assert consumer_done.wait(_cvcuda_test_seconds(10)), (
                "consumer did not reach the ordering probe before producer release"
            )
            signal_reached_before_release = (
                probe.cvcudaTestWaitForGpuSignal(
                    int(100 * _cvcuda_test_timeout_scale)
                )
                == 1
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            consumer_thread.join(_cvcuda_test_seconds(10))
            assert not consumer_thread.is_alive()
            assert consumer_done.is_set()
            assert failures == [], failures
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        writer.sync()
        consumer.sync()
        assert not signal_reached_before_release, (
            f"plane {{plane_index}} {{export_kind}} consumer ignored a "
            f"{{writer_timing.replace('_', '-')}} producer when exported "
            "through Image.cuda(layout=...)"
        )
        """,
        gil=0,
        timeout=30,
        probe=True,
    )


@pytest.mark.parametrize(
    "layout,plane_index", [("HWC", 0), ("HWC", 1), ("CHW", 1), ("CHW", 2)]
)
@pytest.mark.parametrize("export_kind", ["cai", "dlpack"])
def test_planar_image_export_preserves_raw_cai_producers(
    layout, plane_index, export_kind
):
    _run_free_threaded(
        f"""
        import cupy as cp

        class ProducerPlane:
            def __init__(self, array, stream):
                self.array = array
                self.stream = stream

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.array.__cuda_array_interface__)
                interface["stream"] = int(self.stream.ptr)
                return interface

        if {layout!r} == "CHW":
            storage = cp.zeros((3, 6, 8), dtype=cp.uint8)
            arrays = list(storage)
            format = cvcuda.Format.RGB8p
        else:
            arrays = [cp.zeros((6, 8), dtype=cp.uint8), cp.zeros((3, 4, 2), dtype=cp.uint8)]
            format = cvcuda.Format.NV12_ER
        cp.cuda.get_current_stream().synchronize()
        producers = [cp.cuda.Stream(non_blocking=True) for _ in arrays]
        wrapped_planes = [ProducerPlane(array, stream) for array, stream in zip(arrays, producers)]
        consumer = cp.cuda.Stream(non_blocking=True)

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(producers[{plane_index}].ptr) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        image = cvcuda.as_image(wrapped_planes, format)
        exported = image.cuda({layout!r})
        if {layout!r} == "CHW":
            assert not isinstance(exported, list)
            assert tuple(exported.shape) == (3, 6, 8)
        else:
            assert isinstance(exported, list) and len(exported) == 2
            exported = exported[{plane_index}]

        failures = []
        consumer_done = threading.Event()
        def consume():
            try:
                if {export_kind!r} == "cai":
                    # Probe the advertised dependency, avoiding a CuPy host sync
                    # that would hide an incorrectly advertised producer stream.
                    advertised_stream = exported.__cuda_array_interface__["stream"]
                    assert advertised_stream not in (None, 0, 1, 2, -1)
                    assert probe.cvcudaTestQueueGpuSignal(advertised_stream) == 1
                else:
                    with consumer:
                        cp.from_dlpack(exported)
                    assert probe.cvcudaTestQueueGpuSignal(consumer.ptr) == 1
            except BaseException as error:
                failures.append(error)
            finally:
                consumer_done.set()

        probe.cvcudaTestArmGpuSignalGate()
        worker = threading.Thread(target=consume)
        worker.start()
        try:
            assert consumer_done.wait(_cvcuda_test_seconds(10)), (
                "consumer did not reach the ordering probe before producer release"
            )
            bypassed_producer = bool(probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(100)))
        finally:
            probe.cvcudaTestReleaseGpuGate()
        try:
            worker.join(_cvcuda_test_seconds(10))
            assert not worker.is_alive()
            assert failures == [], failures
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        for producer in producers:
            producer.synchronize()
        consumer.synchronize()
        assert not bypassed_producer, "image export ignored a plane's independent CAI producer"
        """,
        gil=0,
        timeout=35,
        probe=True,
    )


def _operator_retirement_finalizer_reentry_script() -> str:
    return """
        import weakref
        import cupy
        import cvcuda

        stream = cvcuda.Stream()
        shape = (1, 16, 16, 1)
        source = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        retired_output = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        outer_output = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        nested_source = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        nested_output = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        reentries = []

        def reenter_same_operator():
            cvcuda.gaussian_into(
                nested_output,
                nested_source,
                (3, 3),
                (0.0, 0.0),
                stream=stream,
            )
            reentries.append(True)

        finalizer = weakref.finalize(retired_output, reenter_same_operator)
        cvcuda.gaussian_into(
            retired_output,
            source,
            (3, 3),
            (0.0, 0.0),
            stream=stream,
        )
        # Complete the first submission without asking CV-CUDA to retire its
        # queued owner closure; the next submission must perform that retirement.
        cupy.cuda.runtime.streamSynchronize(stream.handle)
        del retired_output
        assert finalizer.alive, "pending release did not retain the submitted output"

        cvcuda.gaussian_into(
            outer_output,
            source,
            (3, 3),
            (0.0, 0.0),
            stream=stream,
        )
        assert reentries == [True]
        assert not finalizer.alive
        stream.sync()
        cvcuda.internal.drain_pending_releases()
    """


def test_operator_submission_unlocks_before_retirement_finalizer_reentry():
    _run_free_threaded(
        _operator_retirement_finalizer_reentry_script(),
        gil=0,
        timeout=15,
    )


def test_same_stream_owner_upgrade_defers_finalizer_until_submission_unlock():
    _run_free_threaded(
        r"""
        import gc
        import cupy as cp
        import cvcuda

        underlying = cp.cuda.Stream(non_blocking=True)
        source = cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC")
        destination = cvcuda.Tensor(source.shape, source.dtype, source.layout)
        reentries = []

        owning_stream = None

        class ReentrantHandle(int):
            def __del__(self):
                cvcuda.flip_into(
                    destination,
                    source,
                    0,
                    stream=owning_stream,
                )
                reentries.append(True)

        raw_owner = ReentrantHandle(underlying.ptr)
        raw_stream = cvcuda.as_stream(raw_owner)
        cvcuda.flip_into(destination, source, 0, stream=raw_stream)
        underlying.synchronize()
        cvcuda.internal.drain_pending_releases()

        # Submission history now holds the only remaining references to the
        # non-owning wrapper. Replacing that history must not invoke Python
        # while the outer submission still owns its resource reservation.
        owning_stream = cvcuda.as_stream(underlying)
        del raw_stream, raw_owner
        gc.collect()
        assert reentries == []

        cvcuda.flip_into(destination, source, 0, stream=owning_stream)
        assert reentries == [True]
        underlying.synchronize()
        cvcuda.internal.drain_pending_releases()
        """,
        gil=0,
        timeout=15,
    )


def test_submitted_resources_remain_held_after_the_submitting_thread_exits():
    _run_free_threaded(
        r"""
        import gc
        import os
        import weakref
        import numpy as np
        import cupy as cp

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        resources = []
        resource_refs = []
        failure = []

        def create_resources():
            try:
                src = cvcuda.Tensor((1, 4096, 4096, 4), np.uint8, "NHWC")
                dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
                cp.asarray(src.cuda()).fill(9)
                cp.cuda.get_current_stream().synchronize()
                resource_refs.extend((weakref.ref(src), weakref.ref(dst)))
                resources.extend((src, dst))
            except BaseException as error:
                failure.append(error)

        creator = threading.Thread(target=create_resources)
        creator.start()
        creator.join(_cvcuda_test_seconds(15))
        assert not creator.is_alive()
        assert failure == [], failure
        assert len(resources) == 2

        def submit_and_release_last_wrapper_references():
            try:
                src, dst = resources
                resources.clear()
                cvcuda.flip_into(src=src, dst=dst, flipCode=0, stream=stream)
            except BaseException as error:
                failure.append(error)

        probe.cvcudaTestArmLifetimeGate()
        submitter = threading.Thread(target=submit_and_release_last_wrapper_references)
        submitter.start()
        try:
            assert probe.cvcudaTestWaitForLifetimeGate(_cvcuda_test_milliseconds(10_000)) == 1
            assert probe.cvcudaTestWaitForLifetimeSubmit(_cvcuda_test_milliseconds(10_000)) == 1
            submitter.join(_cvcuda_test_seconds(15))
            assert not submitter.is_alive()
            probe.cvcudaTestStopResourceDestructionTracking()
            assert failure == [], failure
            assert probe.cvcudaTestTrackedTensorCount() == 2
            assert probe.cvcudaTestLifetimeWorkPending() == 1
            gc.collect()
            assert resources == []
            assert all(ref() is not None for ref in resource_refs)
            assert probe.cvcudaTestDestroyedTensorCount() == 0
            assert probe.cvcudaTestTensorDestroyedBeforeLifetimeRelease() == 0
        finally:
            probe.cvcudaTestReleaseLifetimeGate()
            stream.sync()

        cvcuda.internal.drain_pending_releases()
        cvcuda.clear_cache()
        gc.collect()
        assert all(ref() is None for ref in resource_refs)
        assert probe.cvcudaTestTensorDestroyedBeforeLifetimeRelease() == 0
        assert probe.cvcudaTestDestroyedTensorCount() == 2
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_same_stream_shared_resource_fast_paths_are_thread_safe():
    _run_free_threaded(
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        import cupy as cp
        import cvcuda

        thread_count = 8
        iterations = 25
        host = np.arange(1 * 16 * 16, dtype=np.uint8).reshape(1, 16, 16, 1)
        source = cvcuda.as_tensor(cp.asarray(host), "NHWC")
        destination = cvcuda.Tensor(source.shape, source.dtype, source.layout)
        stream = cvcuda.Stream()

        # Publish this stream as both domains' writer before the concurrent
        # phase, so every source read and destination write takes the exact
        # same-writer fast path under contention.
        cvcuda.flip_into(
            src=source, dst=destination, flipCode=0, stream=stream
        )
        stream.sync()
        cvcuda.internal.drain_pending_releases()

        barrier = threading.Barrier(thread_count)

        def submit(index):
            barrier.wait()
            for _ in range(iterations):
                result = cvcuda.flip_into(
                    src=source,
                    dst=destination,
                    flipCode=0,
                    stream=stream,
                )
                assert result is destination

        with ThreadPoolExecutor(max_workers=thread_count) as workers:
            list(workers.map(submit, range(thread_count)))
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        assert np.array_equal(
            cp.asarray(destination.cuda()).get(), np.flip(host, axis=1)
        )
        """,
        gil=0,
        timeout=60,
    )


def test_tensor_batch_mutation_and_operator_submission_are_atomic():
    _run_free_threaded(
        """
        import numpy as np
        import cupy as cp

        tensors = [
            cvcuda.Tensor((8, 8, 1), np.uint8, "HWC"),
            cvcuda.Tensor((8, 8, 1), np.uint8, "HWC"),
        ]
        replacement = cvcuda.Tensor((8, 8, 1), np.uint8, "HWC")
        cp.asarray(tensors[0].cuda())[...].fill(3)
        cp.asarray(tensors[1].cuda())[...].fill(7)
        cp.asarray(replacement.cuda())[...].fill(11)
        cp.cuda.get_current_stream().synchronize()

        batch = cvcuda.TensorBatch(2)
        batch.pushback(tensors)
        iterator = iter(batch)
        assert next(iterator) is tensors[0]
        stream = cvcuda.Stream()
        failures = [None, None]
        outputs = []

        def mutate():
            try:
                batch[1] = replacement
            except BaseException as error:
                failures[0] = error

        def submit():
            try:
                probe.cvcudaTestReachReaderEntry()
                outputs.append(cvcuda.stack(batch, stream=stream))
            except BaseException as error:
                failures[1] = error

        iterator_failure = []
        iterator_tail = []
        iterator_done = threading.Event()

        def consume_existing_iterator():
            try:
                probe.cvcudaTestReachReaderEntry()
                iterator_tail.extend(iterator)
            except BaseException as error:
                iterator_failure.append(error)
            finally:
                iterator_done.set()

        probe.cvcudaTestArmBatchMutation(4)
        mutator = threading.Thread(target=mutate)
        mutator.start()
        assert probe.cvcudaTestWaitForBatchPush(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestArmReaderEntryGate(2)
        consumer = threading.Thread(target=submit)
        iterator_reader = threading.Thread(target=consume_existing_iterator)
        consumer.start()
        iterator_reader.start()
        assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestReleaseReaderEntryGate()
        observation_budget = _cvcuda_test_seconds(0.5)
        submit_reached_during_mutation = (
            probe.cvcudaTestWaitForStackSubmit(
                _cvcuda_test_milliseconds(500)
            )
            == 1
        )
        iterator_completed_during_mutation = iterator_done.wait(observation_budget)
        probe.cvcudaTestReleaseBatchPush()
        assert probe.cvcudaTestWaitForStackSubmit(_cvcuda_test_milliseconds(10_000)) == 1
        assert iterator_done.wait(_cvcuda_test_seconds(10))
        mutator.join(_cvcuda_test_seconds(15))
        consumer.join(_cvcuda_test_seconds(15))
        iterator_reader.join(_cvcuda_test_seconds(15))
        assert not mutator.is_alive()
        assert not consumer.is_alive()
        assert not iterator_reader.is_alive()
        assert failures == [None, None], failures
        assert len(iterator_failure) == 1, iterator_failure
        assert isinstance(iterator_failure[0], RuntimeError), iterator_failure
        assert "structurally modified during iteration" in str(
            iterator_failure[0]
        )
        assert not iterator_completed_during_mutation
        assert iterator_tail == []
        assert list(batch) == [tensors[0], replacement]
        assert len(outputs) == 1
        stream.sync()
        actual = cp.asarray(outputs[0].cuda()).get()
        assert np.all(actual[0] == 3)
        assert np.all(actual[1] == 11)
        assert not submit_reached_during_mutation
        """,
        gil=0,
        probe=True,
    )


def test_dirty_batch_preserves_the_descriptor_of_pending_readers():
    _run_free_threaded(
        """
        import cupy as cp
        import numpy as np

        original = cvcuda.as_tensor(cp.full((16, 16, 1), 7, dtype=cp.uint8), "HWC")
        replacement = cvcuda.as_tensor(cp.full((16, 16, 1), 19, dtype=cp.uint8), "HWC")
        cp.cuda.get_current_stream().synchronize()
        batch = cvcuda.TensorBatch(1)
        batch.pushback(original)
        outputs = [
            cvcuda.Tensor((1, 16, 16, 1), np.uint8, "NHWC")
            for _ in range(3)
        ]
        first = cvcuda.Stream()
        second = cvcuda.Stream()

        # Populate the descriptor first. The stalled submission below then has
        # no H2D copy of its own, leaving its kernel as the only pending user of
        # the shared descriptor when the batch is made dirty.
        cvcuda.stack_into(outputs[0], batch, stream=first)
        first.sync()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(first.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        cvcuda.stack_into(outputs[1], batch, stream=first)
        batch[0] = replacement

        probe.cvcudaTestArmGpuSignalGate()
        try:
            cvcuda.stack_into(outputs[2], batch, stream=second)
            assert probe.cvcudaTestQueueGpuSignal(second.handle) == 1
            bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        first.sync()
        second.sync()
        # Both waiting to reuse the descriptor and copying it on mutation are
        # valid. Earlier submissions must still see the original contents.
        assert np.all(cp.asarray(outputs[1].cuda()).get() == 7)
        assert np.all(cp.asarray(outputs[2].cuda()).get() == 19)
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_writable_exports_join_pending_reader_streams():
    _run_free_threaded(
        """
        import cupy as cp
        import cvcuda

        gate_kernel = cp.RawKernel(
            (
                'extern "C" __global__ void cvcuda_test_export_reader_gate('
                "unsigned int *release, unsigned long long timeout_cycles) {"
                "unsigned long long start = clock64();"
                "while (atomicAdd(release, 0U) == 0U"
                " && clock64() - start < timeout_cycles) {}"
                "}"
            ),
            "cvcuda_test_export_reader_gate",
        )
        set_gate_kernel = cp.RawKernel(
            (
                'extern "C" __global__ void cvcuda_test_set_export_reader_gate('
                "unsigned int *release, unsigned int value) {"
                "atomicExch(release, value);"
                "}"
            ),
            "cvcuda_test_set_export_reader_gate",
        )

        def set_gate(control_stream, release, value):
            set_gate_kernel(
                (1,),
                (1,),
                (release, cp.uint32(value)),
                stream=control_stream,
            )
            control_stream.synchronize()

        def enqueue_gate(stream_handle, release, timeout_cycles):
            with as_cupy_stream(int(stream_handle)):
                gate_kernel(
                    (1,),
                    (1,),
                    (release, cp.uint64(timeout_cycles)),
                )

        gate_timeout_cycles = int(
            10_000_000_000 * _cvcuda_test_timeout_scale
        )

        def assert_export_waits(export_kind):
            source = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
            output = cvcuda.Tensor(source.shape, source.dtype, source.layout)
            producer = cvcuda.Stream()
            reader = cvcuda.Stream()
            consumer = cvcuda.Stream()
            release = cp.empty(1, dtype=cp.uint32)
            control_stream = cp.cuda.Stream(non_blocking=True)

            # Compile both gate paths and the operator before creating the
            # pending-reader state.
            set_gate(control_stream, release, 1)
            enqueue_gate(reader.handle, release, 1)
            cvcuda.flip_into(
                dst=output,
                src=source,
                flipCode=0,
                stream=reader,
            )
            reader.sync()

            source.submitStreamSync(producer)
            producer.sync()

            set_gate(control_stream, release, 0)
            enqueue_gate(reader.handle, release, gate_timeout_cycles)
            cvcuda.flip_into(
                dst=output,
                src=source,
                flipCode=0,
                stream=reader,
            )
            exported = source.cuda()

            capsule = None
            try:
                if export_kind == "cai":
                    ordered_stream = exported.__cuda_array_interface__["stream"]
                else:
                    capsule = exported.__dlpack__(stream=int(consumer.handle))
                    ordered_stream = consumer.handle
                completed_while_reader_blocked = (
                    cp.cuda.runtime.streamQuery(int(ordered_stream)) == 0
                )
            finally:
                set_gate(control_stream, release, 1)

            cp.cuda.runtime.streamSynchronize(int(ordered_stream))
            try:
                reader.sync()
                consumer.sync()
            finally:
                del capsule
            assert not completed_while_reader_blocked, (
                export_kind + " export did not join the pending reader stream"
            )

        for export_kind in ("cai", "dlpack"):
            assert_export_waits(export_kind)

        source = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        output = cvcuda.Tensor(source.shape, source.dtype, source.layout)
        producer = cvcuda.Stream()
        reader = cvcuda.Stream()
        release = cp.empty(1, dtype=cp.uint32)
        control_stream = cp.cuda.Stream(non_blocking=True)
        source.submitStreamSync(producer)
        producer.sync()
        set_gate(control_stream, release, 0)
        enqueue_gate(reader.handle, release, gate_timeout_cycles)
        cvcuda.flip_into(
            dst=output,
            src=source,
            flipCode=0,
            stream=reader,
        )

        try:
            exported = source.cuda()
            advertised = exported.__cuda_array_interface__["stream"]
            assert advertised not in (None, 0, 1, 2, -1)
            handoff_completed_while_reader_blocked = (
                cp.cuda.runtime.streamQuery(int(advertised)) == 0
            )
            del exported

            # Deferred CAI consumers still own the advertised handoff indirectly.
            cvcuda.clear_cache()
        finally:
            set_gate(control_stream, release, 1)

        cp.cuda.runtime.streamSynchronize(int(advertised))
        reader.sync()
        assert not handoff_completed_while_reader_blocked
        """,
        gil=0,
        timeout=45,
    )


def test_multi_batch_operator_sees_a_consistent_view_of_every_batch():
    _run_free_threaded(
        """
        import numpy as np
        import cupy as cp

        def make_batch(tensor):
            batch = cvcuda.TensorBatch(1)
            batch.pushback(tensor)
            return batch

        src_tensor = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        src = make_batch(src_tensor)
        dst_tensor = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        dst = make_batch(dst_tensor)
        src_replacement = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        cp.asarray(src_tensor.cuda()).fill(3)
        cp.asarray(src_replacement.cuda()).fill(11)
        cp.cuda.get_current_stream().synchronize()
        stream = cvcuda.Stream()
        failures = [None, None]

        def mutate_src():
            try:
                src[0] = src_replacement
            except BaseException as error:
                failures[0] = error

        def resize():
            try:
                probe.cvcudaTestReachReaderEntry()
                cvcuda.hq_resize_into(
                    dst,
                    src,
                    min_interpolation=cvcuda.Interp.NEAREST,
                    mag_interpolation=cvcuda.Interp.NEAREST,
                    antialias=False,
                    stream=stream,
                )
            except BaseException as error:
                failures[1] = error

        # Gate the mutating thread inside pushBack so it is provably mid-flight
        # while the operator runs. The operator must still complete against one
        # coherent membership for both batches, whichever side wins the race.
        probe.cvcudaTestArmBatchMutation(4)
        probe.cvcudaTestArmReaderEntryGate(1)
        inner_mutator = threading.Thread(target=mutate_src)
        operator = threading.Thread(target=resize)
        inner_mutator.start()
        try:
            assert probe.cvcudaTestWaitForBatchPush(_cvcuda_test_milliseconds(10_000)) == 1
            operator.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseBatchPush()

        for thread in (inner_mutator, operator):
            if thread.ident is not None:
                thread.join(_cvcuda_test_seconds(15))
                assert not thread.is_alive(), "batch structure acquisition deadlocked"
        assert failures == [None, None], failures

        assert list(src) == [src_replacement]
        assert list(dst) == [dst_tensor]
        stream.sync()
        observed = cp.asarray(dst_tensor.cuda()).get()
        assert np.all(observed == 3) or np.all(observed == 11), (
            "operator observed a torn source-batch membership"
        )
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_swapped_batch_arguments_do_not_deadlock():
    # Two threads running one operator with its batch arguments swapped. The
    # structure locks must be taken in a global order, not in argument order,
    # or each thread holds one batch and blocks on the other.
    _run_free_threaded(
        """
        import threading
        import numpy as np
        import cvcuda

        def make_batch():
            image = cvcuda.Image((16, 16), cvcuda.Format.RGBA8)
            batch = cvcuda.ImageBatchVarShape(1)
            batch.pushback(image)
            return batch

        first = make_batch()
        second = make_batch()
        flip_code = cvcuda.Tensor((1,), np.int32, "N")
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]
        ready = threading.Barrier(2)

        def submit(index):
            try:
                source, destination = (
                    (first, second) if index == 0 else (second, first)
                )
                ready.wait(10 * _cvcuda_test_timeout_scale)
                for _ in range(50):
                    cvcuda.flip_into(
                        dst=destination,
                        src=source,
                        flipCode=flip_code,
                        stream=streams[index],
                    )
            except BaseException as error:
                failures[index] = error

        workers = [threading.Thread(target=submit, args=(i,)) for i in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(_cvcuda_test_seconds(30))
            assert not worker.is_alive(), "swapped batch arguments deadlocked"
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()
        """,
        gil=0,
        timeout=60,
    )


def test_gamma_contrast_into_opposite_batch_directions_do_not_deadlock():
    _run_free_threaded(
        """
        import threading
        import numpy as np
        import cvcuda

        def make_batch():
            batch = cvcuda.ImageBatchVarShape(1)
            batch.pushback(cvcuda.Image((16, 16), cvcuda.Format.RGBA8))
            return batch

        first = make_batch()
        second = make_batch()
        gamma = cvcuda.Tensor((1,), np.float32, "N")
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]
        ready = threading.Barrier(2)

        def submit(index):
            try:
                source, destination = (
                    (first, second) if index == 0 else (second, first)
                )
                ready.wait(10 * _cvcuda_test_timeout_scale)
                for _ in range(50):
                    cvcuda.gamma_contrast_into(
                        dst=destination,
                        src=source,
                        gamma=gamma,
                        stream=streams[index],
                    )
            except BaseException as error:
                failures[index] = error

        workers = [threading.Thread(target=submit, args=(i,)) for i in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(_cvcuda_test_seconds(30))
            assert not worker.is_alive(), (
                "gamma_contrast_into batch structure acquisition deadlocked"
            )
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()
        """,
        gil=0,
        timeout=60,
    )


def test_tensor_batch_submission_locks_retained_children():
    _run_free_threaded(
        """
        import numpy as np

        retained = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        sibling = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        replacement_source = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        batch = cvcuda.TensorBatch(2)
        batch.pushback([retained, sibling])
        stack_stream = cvcuda.Stream()
        writer_stream = cvcuda.Stream()
        failures = [None, None]

        def stack():
            try:
                cvcuda.stack(batch, stream=stack_stream)
            except BaseException as error:
                failures[0] = error

        def overwrite_child():
            try:
                probe.cvcudaTestReachReaderEntry()
                cvcuda.flip_into(
                    dst=retained,
                    src=replacement_source,
                    flipCode=0,
                    stream=writer_stream,
                )
            except BaseException as error:
                failures[1] = error

        probe.cvcudaTestArmStackGate()
        stack_thread = threading.Thread(target=stack)
        writer_thread = threading.Thread(target=overwrite_child)
        stack_thread.start()
        try:
            assert probe.cvcudaTestWaitForStackSubmit(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmFlipGate()
            probe.cvcudaTestArmReaderEntryGate(1)
            writer_thread.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            child_writer_reached_during_batch_submit = (
                probe.cvcudaTestWaitForFirstFlip(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
                == 1
            )
            probe.cvcudaTestReleaseStackGate()
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseStackGate()
            probe.cvcudaTestReleaseFlip()

        stack_thread.join(_cvcuda_test_seconds(15))
        writer_thread.join(_cvcuda_test_seconds(15))
        assert not stack_thread.is_alive()
        assert not writer_thread.is_alive()
        assert failures == [None, None], failures
        assert not child_writer_reached_during_batch_submit
        stack_stream.sync()
        writer_stream.sync()
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


@pytest.mark.parametrize("warm_second", [True, False], ids=["cached", "registering"])
def test_overlapping_image_batches_serialize_shared_output_image(warm_second):
    _run_free_threaded(
        f"""
        import numpy as np

        shared_output = cvcuda.Image((16, 16), cvcuda.Format.RGBA8)
        sources = []
        destinations = []
        for _ in range(2):
            source = cvcuda.ImageBatchVarShape(2)
            source.pushback(cvcuda.Image((16, 16), cvcuda.Format.RGBA8))
            source.pushback(cvcuda.Image((16, 16), cvcuda.Format.RGBA8))
            destination = cvcuda.ImageBatchVarShape(2)
            destination.pushback(shared_output)
            destination.pushback(cvcuda.Image((16, 16), cvcuda.Format.RGBA8))
            sources.append(source)
            destinations.append(destination)

        flip_code = cvcuda.Tensor((2,), np.int32, "N")
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]

        # Exercise both cached overlap and registration while an older snapshot
        # is already inside native enqueue.
        for index in range(2 if {warm_second!r} else 1):
            cvcuda.flip_into(
                dst=destinations[index],
                src=sources[index],
                flipCode=flip_code,
                stream=streams[index],
            )
            streams[index].sync()

        def submit(index):
            try:
                if index == 1:
                    probe.cvcudaTestReachReaderEntry()
                cvcuda.flip_into(
                    dst=destinations[index],
                    src=sources[index],
                    flipCode=flip_code,
                    stream=streams[index],
                )
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmFlipGate()
        workers = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
        workers[0].start()
        try:
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            workers[1].start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            batches_overlapped = bool(
                probe.cvcudaTestWaitForSecondFlip(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseFlipAndObserveSecond()

        assert probe.cvcudaTestWaitForSecondFlip(_cvcuda_test_milliseconds(10_000)) == 1

        for worker in workers:
            if worker.ident is not None:
                worker.join(_cvcuda_test_seconds(15))
                assert not worker.is_alive()
        assert failures == [None, None], failures
        assert not batches_overlapped, "overlapping batch writers reached native enqueue"
        for stream in streams:
            stream.sync()
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_overlapping_tensor_batches_allow_shared_child_reads():
    _run_free_threaded(
        """
        import numpy as np

        shared = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        batches = [cvcuda.TensorBatch(2), cvcuda.TensorBatch(2)]
        for batch in batches:
            batch.pushback(
                [shared, cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")]
            )
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]

        for index in range(2):
            cvcuda.stack(batches[index], stream=streams[index])
            streams[index].sync()

        def submit(index):
            try:
                cvcuda.stack(batches[index], stream=streams[index])
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmConcurrentStackGate()
        workers = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
        workers[0].start()
        try:
            assert probe.cvcudaTestWaitForFirstConcurrentStack(_cvcuda_test_milliseconds(10_000)) == 1
            workers[1].start()
            readers_overlapped = bool(
                probe.cvcudaTestWaitForSecondConcurrentStack(_cvcuda_test_milliseconds(10_000))
            )
        finally:
            probe.cvcudaTestReleaseConcurrentStack()

        for worker in workers:
            if worker.ident is not None:
                worker.join(_cvcuda_test_seconds(15))
                assert not worker.is_alive()
        assert failures == [None, None], failures
        assert readers_overlapped, "compatible child readers were serialized"
        for stream in streams:
            stream.sync()
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_cold_batch_registration_waits_for_direct_child_submission():
    _run_free_threaded(
        """
        import numpy as np

        shared = cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")
        source = cvcuda.Tensor(shared.shape, shared.dtype, shared.layout)
        cold_batch = cvcuda.TensorBatch(2)
        cold_batch.pushback(
            [shared, cvcuda.Tensor((16, 16, 1), np.uint8, "HWC")]
        )
        writer_stream = cvcuda.Stream()
        batch_stream = cvcuda.Stream()
        failures = [None, None]

        def write_child():
            try:
                cvcuda.flip_into(
                    dst=shared,
                    src=source,
                    flipCode=0,
                    stream=writer_stream,
                )
            except BaseException as error:
                failures[0] = error

        def submit_cold_batch():
            try:
                probe.cvcudaTestReachReaderEntry()
                cvcuda.stack(cold_batch, stream=batch_stream)
            except BaseException as error:
                failures[1] = error

        probe.cvcudaTestArmFlipGate()
        writer = threading.Thread(target=write_child)
        submitter = threading.Thread(target=submit_cold_batch)
        writer.start()
        try:
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmStackGate()
            probe.cvcudaTestArmReaderEntryGate(1)
            submitter.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            batch_bypassed_direct_submit = bool(
                probe.cvcudaTestWaitForStackSubmit(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
            probe.cvcudaTestReleaseFlip()
            assert probe.cvcudaTestWaitForStackSubmit(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseFlip()
            probe.cvcudaTestReleaseStackGate()

        for worker in (writer, submitter):
            if worker.ident is not None:
                worker.join(_cvcuda_test_seconds(15))
                assert not worker.is_alive()
        assert failures == [None, None], failures
        assert not batch_bypassed_direct_submit, (
            "a cold batch registered while a child writer owned native enqueue"
        )
        writer_stream.sync()
        batch_stream.sync()
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


@pytest.mark.parametrize("alias_kind", ["reshape", "image_tensor"])
def test_library_created_aliases_share_submission_domain(alias_kind):
    _run_free_threaded(
        f"""
        import numpy as np

        if {alias_kind!r} == "reshape":
            first = cvcuda.Tensor((1, 16, 16, 1), np.uint8, "NHWC")
            second = first.reshape(first.shape, layout=first.layout)
        else:
            image = cvcuda.Image((16, 16), cvcuda.Format.RGBA8)
            first = cvcuda.as_tensor(image)
            second = cvcuda.as_tensor(image)

        sources = [
            cvcuda.Tensor(first.shape, first.dtype, first.layout),
            cvcuda.Tensor(first.shape, first.dtype, first.layout),
        ]
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]

        def write(index, destination):
            try:
                if index == 1:
                    probe.cvcudaTestReachReaderEntry()
                cvcuda.flip_into(
                    dst=destination,
                    src=sources[index],
                    flipCode=0,
                    stream=streams[index],
                )
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmFlipGate()
        writers = [
            threading.Thread(target=write, args=(0, first)),
            threading.Thread(target=write, args=(1, second)),
        ]
        writers[0].start()
        try:
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            writers[1].start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            alias_submissions_overlapped = bool(
                probe.cvcudaTestWaitForSecondFlip(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseFlipAndObserveSecond()

        assert probe.cvcudaTestWaitForSecondFlip(_cvcuda_test_milliseconds(10_000)) == 1

        for writer in writers:
            if writer.ident is not None:
                writer.join(_cvcuda_test_seconds(15))
                assert not writer.is_alive()
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()
        assert not alias_submissions_overlapped, (
            {alias_kind!r} + " aliases used independent submission locks"
        )
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


@pytest.mark.parametrize("alias_kind", ["reshape", "image_tensor"])
def test_library_created_aliases_share_stream_dependency_state(alias_kind):
    _run_free_threaded(
        f"""
        import numpy as np

        if {alias_kind!r} == "reshape":
            first = cvcuda.Tensor((1, 16, 16, 1), np.uint8, "NHWC")
            second = first.reshape(first.shape, layout=first.layout)
        else:
            image = cvcuda.Image((16, 16), cvcuda.Format.RGBA8)
            first = cvcuda.as_tensor(image)
            second = cvcuda.as_tensor(image)

        producer_input = cvcuda.Tensor(first.shape, first.dtype, first.layout)
        producer_stream = cvcuda.Stream()

        # Prime operator and allocation caches before parking the stream; CUDA
        # first-use work may synchronize the device.
        cvcuda.flip_into(
            dst=first, src=producer_input, flipCode=0, stream=producer_stream
        )
        producer_stream.sync()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(producer_stream.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        signal_reached_before_release = False
        probe.cvcudaTestArmGpuSignalGate()
        try:
            cvcuda.flip_into(
                dst=first, src=producer_input, flipCode=0, stream=producer_stream
            )
            exported = second.cuda()
            alias_stream = exported.__cuda_array_interface__["stream"]
            assert alias_stream not in (
                None,
                0,
                1,
                2,
                -1,
            )
            assert probe.cvcudaTestQueueGpuSignal(alias_stream) == 1
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        producer_stream.sync()
        assert not signal_reached_before_release, (
            {alias_kind!r} + " alias handoff ignored the producer's pending work"
        )
        """,
        gil=0,
        probe=True,
    )


def test_persistent_findhomography_operator_serializes_shared_submissions():
    _run_free_threaded(
        """

        batch_size = 1
        num_points = 8
        operator = cvcuda.get_findhomography_operator(batch_size, num_points)
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        src = [
            cvcuda.Tensor((batch_size, num_points), cvcuda.Type._2F32, "NW")
            for _ in range(2)
        ]
        dst = [
            cvcuda.Tensor((batch_size, num_points), cvcuda.Type._2F32, "NW")
            for _ in range(2)
        ]
        models = [
            cvcuda.Tensor((batch_size, 3, 3), cvcuda.Type.F32, "NHW")
            for _ in range(2)
        ]
        failures = [None, None]

        def submit(index):
            try:
                if index == 1:
                    probe.cvcudaTestReachReaderEntry()
                cvcuda.findhomography_into_with_op(
                    models[index],
                    src[index],
                    dst[index],
                    operator,
                    stream=streams[index],
                )
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmFindHomographyGate()
        threads = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
        threads[0].start()
        try:
            assert probe.cvcudaTestWaitForFirstFindHomography(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            threads[1].start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            same_operator_overlapped = bool(
                probe.cvcudaTestWaitForSecondFindHomography(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseFindHomographyAndObserveSecond()

        assert probe.cvcudaTestWaitForSecondFindHomography(_cvcuda_test_milliseconds(10_000)) == 1

        for thread in threads:
            if thread.ident is not None:
                thread.join(_cvcuda_test_seconds(15))
                assert not thread.is_alive()
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()
        assert not same_operator_overlapped, (
            "one persistent FindHomography operator was submitted concurrently"
        )
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


@pytest.mark.parametrize("argument", ["stream", "max_version"])
def test_dlpack_argument_conversion_can_reenter_export(argument):
    _run_free_threaded(
        f"""
        import cvcuda

        tensor = cvcuda.Tensor((4, 5), cvcuda.Type.U8, "HW")
        exported = tensor.cuda()
        conversions = []

        class ReentrantInteger:
            def __index__(self):
                conversions.append(True)
                nested = exported.__dlpack__(stream=1)
                assert type(nested).__name__ == "PyCapsule"
                return 1

        value = ReentrantInteger()
        argument = {argument!r}
        if argument == "max_version":
            value = (value, 0)
        capsule = exported.__dlpack__(**{{argument: value}})
        assert type(capsule).__name__ == "PyCapsule"
        assert conversions == [True]
        """,
        gil=0,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("batch_kind", "mutation", "action"),
    [
        ("image", "push", 1),
        ("tensor", "push", 0),
        ("image", "pop", 5),
        ("image", "clear", 6),
        ("tensor", "pop", 2),
        ("tensor", "clear", 3),
    ],
)
def test_batch_mutation_and_iteration_are_linearizable(batch_kind, mutation, action):
    _run_free_threaded(
        f"""
        import numpy as np

        batch_kind = {batch_kind!r}
        mutation = {mutation!r}
        action = {action}

        item_count = 1 if mutation == "push" else 6
        if batch_kind == "image":
            items = [
                cvcuda.Image((8, 8), cvcuda.Format.RGBA8)
                for _ in range(item_count)
            ]
            batch = cvcuda.ImageBatchVarShape(item_count)
        else:
            items = [
                cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC")
                for _ in range(item_count)
            ]
            batch = cvcuda.TensorBatch(item_count)

        if mutation != "push":
            batch.pushback(items)

        failure = []

        def mutate():
            try:
                if mutation == "push":
                    batch.pushback(items[0])
                elif mutation == "pop":
                    batch.popback(2)
                else:
                    batch.clear()
            except BaseException as error:
                failure.append(error)

        probe.cvcudaTestArmBatchMutation(action)
        mutator = threading.Thread(target=mutate)
        readers = []
        mutator.start()
        try:
            assert probe.cvcudaTestWaitForBatchPush(
                _cvcuda_test_milliseconds(10_000)
            ) == 1

            observations = [None, None]
            reader_failures = [None, None]
            reader_done = [threading.Event(), threading.Event()]

            def read_length():
                try:
                    probe.cvcudaTestReachReaderEntry()
                    observations[0] = len(batch)
                except BaseException as error:
                    reader_failures[0] = error
                finally:
                    reader_done[0].set()

            def read_snapshot():
                try:
                    probe.cvcudaTestReachReaderEntry()
                    observations[1] = list(batch)
                except BaseException as error:
                    reader_failures[1] = error
                finally:
                    reader_done[1].set()

            probe.cvcudaTestArmReaderEntryGate(2)
            readers = [
                threading.Thread(target=read_length),
                threading.Thread(target=read_snapshot),
            ]
            for reader in readers:
                reader.start()
            assert probe.cvcudaTestWaitForReaderEntry(
                _cvcuda_test_milliseconds(10_000)
            ) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            completed_during_mutation = [
                done.wait(_cvcuda_test_seconds(0.5)) for done in reader_done
            ]
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseBatchPush()
            mutator.join(_cvcuda_test_seconds(15))
            for reader in readers:
                if reader.ident is not None:
                    reader.join(_cvcuda_test_seconds(15))

        expected = (
            [items[0]]
            if mutation == "push"
            else items[:-2]
            if mutation == "pop"
            else []
        )
        assert not mutator.is_alive()
        assert all(not reader.is_alive() for reader in readers)
        assert failure == [], failure
        assert reader_failures == [None, None], reader_failures
        assert completed_during_mutation == [False, False]
        assert observations == [len(expected), expected]
        """,
        gil=0,
        probe=True,
    )


@pytest.mark.parametrize("batch_kind", ["tensor", "image"])
def test_batch_iterators_reject_clear_after_snapshot(batch_kind):
    _run_free_threaded(
        f"""
        import threading
        import numpy as np
        import cvcuda

        batch_kind = {batch_kind!r}
        if batch_kind == "tensor":
            items = [cvcuda.Tensor((1, 8, 8, 1), np.uint8, "NHWC") for _ in range(6)]
            batch = cvcuda.TensorBatch(len(items))
        else:
            items = [cvcuda.Image((8, 8), cvcuda.Format.RGBA8) for _ in range(6)]
            batch = cvcuda.ImageBatchVarShape(len(items))
        batch.pushback(items)

        expected_ids = [item.id for item in items]
        iterator = iter(batch)
        first = next(iterator)
        assert first.id == expected_ids[0]
        del first
        items.clear()
        failure = []

        def clear():
            try:
                batch.clear()
            except BaseException as error:
                failure.append(error)

        mutator = threading.Thread(target=clear)
        mutator.start()
        mutator.join(_cvcuda_test_seconds(15))
        assert not mutator.is_alive()
        assert failure == [], failure
        assert list(batch) == []

        try:
            list(iterator)
        except RuntimeError as error:
            assert "structurally modified during iteration" in str(error)
        else:
            raise AssertionError("clearing the batch did not invalidate its iterator")
        """,
        gil=0,
    )


def _require_cuda_device_count(minimum: int) -> None:
    import cupy

    count = cupy.cuda.runtime.getDeviceCount()
    if count < minimum:
        pytest.skip(f"requires at least {minimum} CUDA devices, found {count}")


def test_cache_operations_are_safe_during_concurrent_allocation_and_export():
    _run_free_threaded(
        """
        import gc
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        import cvcuda

        thread_count = 8
        iterations = 500
        barrier = threading.Barrier(thread_count)

        def allocate_and_export(thread_index):
            barrier.wait()
            for iteration in range(iterations):
                tensor = cvcuda.Tensor((8, 8, 4), np.uint8, "HWC")
                interface = tensor.cuda().__cuda_array_interface__
                assert interface["shape"] == (8, 8, 4)
                assert interface["data"][0] != 0
                if iteration % 50 == 0:
                    gc.collect()

        def clear_and_inspect(thread_index):
            barrier.wait()
            for _ in range(iterations):
                cvcuda.clear_cache()
                # Concurrent allocation makes relationships between two
                # snapshots inherently racy; these calls are fault injectors.
                cvcuda.cache_size(cvcuda.ThreadScope.GLOBAL)
                cvcuda.cache_size(cvcuda.ThreadScope.LOCAL)
                cvcuda.current_cache_size_inbytes()

        with ThreadPoolExecutor(max_workers=thread_count) as workers:
            futures = [
                workers.submit(
                    allocate_and_export if index < thread_count // 2 else clear_and_inspect,
                    index,
                )
                for index in range(thread_count)
            ]
            for future in futures:
                future.result()
        """,
        gil=0,
        timeout=75,
    )


@pytest.mark.parametrize(
    "cache_headroom_items", [0, 4], ids=["at-limit", "below-limit"]
)
def test_cache_pressure_reuses_pending_tensor_without_the_gil(cache_headroom_items):
    _run_free_threaded(
        f"""
        import gc
        import numpy as np

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        assert stream is not cvcuda.Stream.current
        source = cvcuda.Tensor((1, 256, 256, 3), np.uint8, "NHWC")

        warmed = cvcuda.flip(source, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        item_bytes = cvcuda.internal.nbytes_in_cache(warmed)
        baseline_bytes = cvcuda.current_cache_size_inbytes()
        del warmed
        gc.collect()
        cache_limit = baseline_bytes + {cache_headroom_items} * item_bytes
        cvcuda.set_cache_limit_inbytes(cache_limit)
        if {cache_headroom_items}:
            assert cvcuda.current_cache_size_inbytes() + item_bytes < cache_limit

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(stream.handle) == 1
        try:
            assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
            first = cvcuda.flip(source, 0, stream=stream)
            assert first.id == warmed_id
            first_id = first.id
            del first
            gc.collect()

            second = cvcuda.flip(source, 0, stream=stream)
            assert second.id == first_id, (
                "free-threaded cache path allocated instead of reusing its "
                "queue-only Tensor"
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        stream.sync()
        cvcuda.internal.drain_pending_releases()
        """,
        gil=0,
        timeout=40,
        probe=True,
    )


def test_below_limit_pending_reuse_rejects_stale_stream_history_without_the_gil():
    test_directory = os.path.dirname(os.path.abspath(__file__))
    _run_free_threaded(
        f"""
        import gc
        import sys

        import cupy
        import cvcuda

        sys.path.insert(0, {test_directory!r})
        from test_resource_lifetime import _park_gpu

        cvcuda.clear_cache()
        producer = cvcuda.Stream()
        previous_consumer = cvcuda.Stream()
        assert int(producer.handle) != int(previous_consumer.handle)

        shape = (1, 64, 64, 3)
        first_values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(shape)
        second_values = (255 - first_values).astype(cupy.uint8)
        cupy.cuda.get_current_stream().synchronize()
        first_source = cvcuda.as_tensor(first_values, "NHWC")
        second_source = cvcuda.as_tensor(second_values, "NHWC")

        # Compile the finite delay and prime both external source wrappers,
        # Flip paths, and the sole matching cached output before the ordering
        # window. A legitimate allocation miss below may synchronize CUDA.
        _park_gpu(producer, 0)
        warmed = cvcuda.flip(first_source, 0, stream=producer)
        cvcuda.flip_into(warmed, second_source, 0, stream=producer)
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        item_bytes = cvcuda.internal.nbytes_in_cache(warmed)
        baseline_bytes = cvcuda.current_cache_size_inbytes()
        del warmed
        gc.collect()

        cache_limit = baseline_bytes + 4 * item_bytes
        assert baseline_bytes + 2 * item_bytes < cache_limit
        cvcuda.set_cache_limit_inbytes(cache_limit)

        _park_gpu(producer, 2_000_000_000)
        first = cvcuda.flip(first_source, 0, stream=producer)
        assert first.id == warmed_id
        first_id = first.id

        # Move only the allocation dependency history to B. Unlike a B-side
        # ResourceGuard submission, submitStreamSync does not retain this Python
        # wrapper in B's release ledger, so lookup must consult that history.
        first.submitStreamSync(previous_consumer)
        del first
        gc.collect()

        _park_gpu(previous_consumer, 2_000_000_000)
        second = cvcuda.flip(second_source, 0, stream=producer)
        assert second.id != first_id, (
            "below-limit reuse ignored free-threaded stream history"
        )

        producer.sync()
        previous_consumer.sync()
        assert cupy.array_equal(
            cupy.asarray(second.cuda()), cupy.flip(second_values, axis=1)
        )
        """,
        gil=0,
        timeout=40,
    )


def test_pending_tensor_is_not_reused_before_first_wrapper_publication():
    _run_free_threaded(
        """
        import gc
        import threading
        import numpy as np

        cvcuda.clear_cache()
        pending_stream = cvcuda.Stream()
        contender_streams = [cvcuda.Stream(), cvcuda.Stream()]
        source = cvcuda.Tensor((1, 256, 256, 3), np.uint8, "NHWC")

        # Warm CUDA/operator setup before either probe can make an incidental
        # cold-path synchronization look like contender serialization.
        warmed = cvcuda.flip(source, 0, stream=pending_stream)
        pending_stream.sync()
        cvcuda.internal.drain_pending_releases()
        del warmed
        gc.collect()

        cvcuda.set_cache_limit_inbytes(0)
        outputs = [None, None]
        failures = [None, None]

        def allocate(index):
            try:
                outputs[index] = cvcuda.flip(
                    source,
                    0,
                    stream=contender_streams[index],
                )
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(pending_stream.handle) == 1
        workers = []
        flip_gate_armed = False
        try:
            assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
            original = cvcuda.flip(source, 0, stream=pending_stream)
            original_id = original.id
            del original
            gc.collect()
            assert cvcuda.current_cache_size_inbytes() > 0

            # The first contender claims the pending Tensor, then stops inside
            # C++ before nanobind can publish its Python wrapper. Its native
            # shared_ptr must still exclude a second concurrent reuse.
            probe.cvcudaTestArmFlipGate()
            flip_gate_armed = True
            workers = [
                threading.Thread(target=allocate, args=(index,)) for index in range(2)
            ]
            workers[0].start()
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            workers[1].start()
            second_reached_before_publication = bool(
                probe.cvcudaTestWaitForSecondFlip(_cvcuda_test_milliseconds(10_000))
            )
        finally:
            if flip_gate_armed:
                probe.cvcudaTestReleaseFlip()
            probe.cvcudaTestReleaseGpuGate()
            for worker in workers:
                if worker.ident is not None:
                    worker.join(_cvcuda_test_seconds(15))
            if not any(worker.is_alive() for worker in workers):
                pending_stream.sync()
                for stream in contender_streams:
                    stream.sync()
                cvcuda.internal.drain_pending_releases()

        assert not any(worker.is_alive() for worker in workers)
        assert failures == [None, None], failures
        assert second_reached_before_publication, (
            "second allocating Flip reused the first contender's unpublished Tensor"
        )
        assert outputs[0].id == original_id
        assert outputs[1].id != original_id
        assert outputs[0].id != outputs[1].id
        """,
        gil=0,
        timeout=50,
        probe=True,
    )


def test_concurrent_resource_ids_are_unique():
    _run_free_threaded(
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import cvcuda

        ready = threading.Barrier(8)
        def allocate(_):
            ready.wait()
            return [cvcuda.Tensor((1, 1), cvcuda.Type.U8) for _ in range(20)]

        with ThreadPoolExecutor(max_workers=8) as workers:
            groups = list(workers.map(allocate, range(8)))
        identifiers = [tensor.id for group in groups for tensor in group]
        assert len(set(identifiers)) == len(identifiers)
        """,
        gil=0,
    )


@pytest.mark.parametrize("plane_index", [1, 2])
@pytest.mark.parametrize("consumer_kind", ["resource", "cai", "dlpack"])
def test_contiguous_planar_image_tensor_alias_observes_future_plane_writes(
    plane_index,
    consumer_kind,
):
    _run_free_threaded(
        f"""
        import cupy as cp

        plane_index = {plane_index}
        consumer_kind = {consumer_kind!r}
        storage = cp.zeros((3, 6, 8), dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        source_planes = [
            cvcuda.as_tensor(storage[index], "HW") for index in range(3)
        ]
        image = cvcuda.as_image(
            [plane.cuda() for plane in source_planes],
            cvcuda.Format.RGB8p,
        )
        exported = image.cuda("CHW")
        assert not isinstance(exported, list)
        alias = cvcuda.as_tensor(exported, "CHW")

        writer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        consumer_external_stream = as_cupy_stream(consumer.handle)

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        source_planes[plane_index].submitStreamSync(writer)

        failures = []
        consumer_done = threading.Event()

        def consume():
            try:
                if consumer_kind == "resource":
                    alias.submitStreamSync(consumer)
                    assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
                elif consumer_kind == "cai":
                    advertised_stream = alias.cuda().__cuda_array_interface__["stream"]
                    assert advertised_stream not in (None, 0, 1, 2, -1)
                    assert probe.cvcudaTestQueueGpuSignal(advertised_stream) == 1
                else:
                    with consumer_external_stream:
                        cp.from_dlpack(alias.cuda())
                    assert probe.cvcudaTestQueueGpuSignal(consumer.handle) == 1
            except BaseException as error:
                failures.append(error)
            finally:
                consumer_done.set()

        signal_reached_before_release = False
        probe.cvcudaTestArmGpuSignalGate()
        worker = threading.Thread(target=consume)
        worker.start()
        try:
            assert consumer_done.wait(_cvcuda_test_seconds(10)), (
                "consumer did not reach the ordering probe before producer release"
            )
            signal_reached_before_release = bool(
                probe.cvcudaTestWaitForGpuSignal(
                    int(100 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()

        try:
            worker.join(_cvcuda_test_seconds(10))
            assert not worker.is_alive()
            assert failures == [], failures
            assert probe.cvcudaTestWaitForGpuSignal(_cvcuda_test_milliseconds(10_000)) == 1
        finally:
            probe.cvcudaTestReleaseGpuSignalGate()

        writer.sync()
        consumer.sync()
        assert not signal_reached_before_release, (
            f"Image -> Tensor {{consumer_kind}} alias ignored a future write "
            f"through contiguous plane {{plane_index}}"
        )
        """,
        gil=0,
        timeout=35,
        probe=True,
    )


def test_default_stream_readers_remain_device_specific_across_threads():
    _require_cuda_device_count(2)
    _run_free_threaded(
        """
        import threading
        import cupy as cp
        import cvcuda

        gate_kernel = cp.RawKernel(
            (
                'extern "C" __global__ void cvcuda_test_device_reader_gate('
                "unsigned int *release, unsigned long long timeout_cycles) {"
                "unsigned long long start = clock64();"
                "while (atomicAdd(release, 0U) == 0U"
                " && clock64() - start < timeout_cycles) {}"
                "}"
            ),
            "cvcuda_test_device_reader_gate",
        )
        set_gate_kernel = cp.RawKernel(
            (
                'extern "C" __global__ void cvcuda_test_set_device_reader_gate('
                "unsigned int *release, unsigned int value) {"
                "atomicExch(release, value);"
                "}"
            ),
            "cvcuda_test_set_device_reader_gate",
        )

        with cp.cuda.Device(0):
            source = cvcuda.Tensor((1, 8, 8, 1), cvcuda.Type.U8, "NHWC")

        # Compile both gate paths before the pending-reader state, then close
        # the gate through an independent device-1 stream.
        with cp.cuda.Device(1):
            release = cp.empty(1, dtype=cp.uint32)
            control_stream = cp.cuda.Stream(non_blocking=True)
            set_gate_kernel(
                (1,), (1,), (release, cp.uint32(1)), stream=control_stream
            )
            control_stream.synchronize()
            with cp.cuda.Stream.null:
                gate_kernel((1,), (1,), (release, cp.uint64(1)))
            cp.cuda.Stream.null.synchronize()
            set_gate_kernel(
                (1,), (1,), (release, cp.uint32(0)), stream=control_stream
            )
            control_stream.synchronize()

        default_stream = cvcuda.Stream.default
        stress_barrier = threading.Barrier(2)
        stress_failures = [None, None]

        def stress_default_reader(device):
            try:
                with cp.cuda.Device(device):
                    for _ in range(32):
                        stress_barrier.wait(_cvcuda_test_seconds(10))
                        cvcuda._test.resourceguard_legacy_commit(
                            default_stream, source
                        )
                        stress_barrier.wait(_cvcuda_test_seconds(10))
                    default_stream.sync()
            except BaseException as error:
                stress_failures[device] = error

        stress_threads = [
            threading.Thread(target=stress_default_reader, args=(device,))
            for device in range(2)
        ]
        for thread in stress_threads:
            thread.start()
        for thread in stress_threads:
            thread.join(_cvcuda_test_seconds(20))
        assert all(not thread.is_alive() for thread in stress_threads)
        assert stress_failures == [None, None], stress_failures

        # Reset the reader set, then publish device 0 as the most recent
        # reader for the shared null-handle Stream wrapper.
        with cp.cuda.Device(0):
            reset_stream = cvcuda.Stream()
            source.submitStreamSync(reset_stream)
            reset_stream.sync()
            cvcuda._test.resourceguard_legacy_commit(default_stream, source)
            default_stream.sync()

        # Hold device 1's default stream on the GPU. Its READ registration must
        # not hit device 0's published-reader identity merely because both use
        # the same process-wide Python Stream wrapper.
        reader_registered = threading.Event()
        reader_failure = [None]

        def register_blocked_device_one_reader():
            try:
                with cp.cuda.Device(1), cp.cuda.Stream.null:
                    gate_kernel(
                        (1,), (1,), (release, cp.uint64(10_000_000_000))
                    )
                    cvcuda._test.resourceguard_legacy_commit(default_stream, source)
            except BaseException as error:
                reader_failure[0] = error
            finally:
                reader_registered.set()

        reader = threading.Thread(target=register_blocked_device_one_reader)
        reader.start()
        writer = None
        writer_bypassed_device_one = False
        main_failure = [None]
        try:
            assert reader_registered.wait(_cvcuda_test_seconds(10))
            assert reader_failure == [None], reader_failure

            with cp.cuda.Device(0):
                writer = cvcuda.Stream()
                source.submitStreamSync(writer)
                writer_bypassed_device_one = (
                    cp.cuda.runtime.streamQuery(int(writer.handle)) == 0
                )
        except BaseException as error:
            main_failure[0] = error
        finally:
            with cp.cuda.Device(1):
                set_gate_kernel(
                    (1,), (1,), (release, cp.uint32(1)), stream=control_stream
                )
                control_stream.synchronize()

        reader.join(_cvcuda_test_seconds(15))
        assert not reader.is_alive()
        assert reader_failure == [None], reader_failure
        if writer is not None:
            with cp.cuda.Device(0):
                writer.sync()
        with cp.cuda.Device(1):
            default_stream.sync()
        assert main_failure == [None], main_failure
        assert not writer_bypassed_device_one, (
            "a writer bypassed a pending default-stream reader on another device"
        )
        """,
        gil=0,
        timeout=60,
    )


@pytest.mark.parametrize("batch_kind", ["tensor", "image_batch"])
def test_dirty_batch_reads_do_not_export_mutable_descriptors_concurrently(batch_kind):
    _run_free_threaded(
        f"""
        import numpy as np

        batch_kind = {batch_kind!r}
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]

        if batch_kind == "tensor":
            batch = cvcuda.TensorBatch(1)
            batch.pushback(cvcuda.Tensor((16, 16, 1), np.uint8, "HWC"))
            destinations = [
                cvcuda.Tensor((1, 16, 16, 1), np.uint8, "NHWC") for _ in range(2)
            ]
        else:
            batch = cvcuda.ImageBatchVarShape(1)
            batch.pushback(cvcuda.Image((16, 16), cvcuda.Format.RGB8))
            destinations = [
                cvcuda.Tensor((1, 16, 16, 3), np.uint8, "NHWC") for _ in range(2)
            ]

        def run(index):
            try:
                if index == 1:
                    probe.cvcudaTestReachReaderEntry()
                cvcuda.stack_into(destinations[index], batch, stream=streams[index])
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmConcurrentStackGate()
        first = threading.Thread(target=run, args=(0,))
        second = threading.Thread(target=run, args=(1,))
        first.start()
        try:
            assert probe.cvcudaTestWaitForFirstConcurrentStack(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            second.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            second_reached_submit = (
                probe.cvcudaTestWaitForSecondConcurrentStack(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
                == 1
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseConcurrentStackAndObserveSecond()

        assert probe.cvcudaTestWaitForSecondConcurrentStack(_cvcuda_test_milliseconds(10_000)) == 1

        first.join(_cvcuda_test_seconds(15))
        if second.ident is not None:
            second.join(_cvcuda_test_seconds(15))
        assert not first.is_alive()
        assert not second.is_alive()
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()
        assert not second_reached_submit, "dirty batch READ submissions ran mutable export concurrently"
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_dlpack_export_selects_the_buffer_device_across_gpus():
    _require_cuda_device_count(2)
    _run_free_threaded(
        """
        import cupy as cp
        import cvcuda

        with cp.cuda.Device(1):
            cvcuda.clear_cache()
            image = cvcuda.Image((8, 8), cvcuda.Format.RGB8)
            image_buffer = image.cuda()
            assert image_buffer.__dlpack_device__() == (2, 1), (
                "Image DLPack export reported the wrong owning CUDA device"
            )

            tensor = cvcuda.Tensor((1, 8, 8, 1), cvcuda.Type.U8, "NHWC")
            writer = cvcuda.Stream()
            consumer = cp.cuda.Stream(non_blocking=True)
            tensor.submitStreamSync(writer)
            tensor_buffer = tensor.cuda()
            assert tensor_buffer.__dlpack_device__() == (2, 1)

        capsule = None
        try:
            with cp.cuda.Device(0):
                capsule = tensor_buffer.__dlpack__(stream=int(consumer.ptr))
            with cp.cuda.Device(1):
                consumer.synchronize()
        finally:
            with cp.cuda.Device(1):
                writer.sync()
            del capsule
        """,
        gil=0,
    )


def test_dlpack_import_requests_legacy_stream_on_the_producer_device():
    _require_cuda_device_count(2)
    script = """
        import cupy as cp
        import cvcuda

        with cp.cuda.Device(1):
            array = cp.zeros((8, 8), dtype=cp.uint8)

        class DeviceAwareProducer:
            def __init__(self, buffer):
                self.buffer = buffer
                self.request_devices = []

            def __dlpack_device__(self):
                return self.buffer.__dlpack_device__()

            def __dlpack__(self, *args, **kwargs):
                self.request_devices.append(cp.cuda.runtime.getDevice())
                if "max_version" in kwargs:
                    raise TypeError(
                        "__dlpack__() got an unexpected keyword argument "
                        "'max_version'"
                    )
                return self.buffer.__dlpack__(*args, **kwargs)

        producer = DeviceAwareProducer(array)
        with cp.cuda.Device(0):
            tensor = cvcuda.as_tensor(producer, "HW")
            assert cp.cuda.runtime.getDevice() == 0

        assert producer.request_devices == [1, 1]
        with cp.cuda.Device(1):
            assert cp.array_equal(cp.asarray(tensor.cuda()), array)
    """
    _run_free_threaded(script, gil=0)


@pytest.mark.parametrize("versioned", [False, True])
def test_dlpack_manager_deleter_attaches_native_thread(versioned):
    _run_free_threaded(
        f"""
        import gc
        import weakref
        import cupy as cp

        versioned = {versioned!r}
        attached_probe = ctypes.PyDLL(None)
        attached_probe.cvcudaTestTakeDLPackManager.argtypes = [
            ctypes.py_object,
            ctypes.c_int,
        ]
        attached_probe.cvcudaTestTakeDLPackManager.restype = ctypes.c_void_p
        ctypes.pythonapi.PyGILState_Check.restype = ctypes.c_int

        destructor_attachment = []

        class Sentinel:
            def __del__(self):
                destructor_attachment.append(
                    bool(ctypes.pythonapi.PyGILState_Check())
                )

        sentinel = Sentinel()
        sentinel_ref = weakref.ref(sentinel)

        class ExternalOwner:
            def __init__(self, array, sentinel):
                self.array = array
                self.sentinel = sentinel

            @property
            def __cuda_array_interface__(self):
                return self.array.__cuda_array_interface__

        owner = ExternalOwner(cp.zeros((16, 16, 4), dtype=cp.uint8), sentinel)
        image = cvcuda.as_image(owner, cvcuda.Format.RGBA8)
        exported = image.cuda()
        if versioned:
            capsule = exported.__dlpack__(max_version=(1, 0))
        else:
            capsule = exported.__dlpack__()
        manager = attached_probe.cvcudaTestTakeDLPackManager(
            capsule, int(versioned)
        )
        assert manager

        del sentinel
        del owner
        del capsule
        del exported
        del image
        cvcuda.clear_cache()
        gc.collect()
        assert sentinel_ref() is not None

        assert (
            probe.cvcudaTestDeleteDLPackManagerOnNativeThread(
                manager, int(versioned)
            )
            == 1
        )
        assert probe.cvcudaTestNativeDLPackDeleteThreadWasAttached() == 0
        assert sentinel_ref() is None
        assert destructor_attachment == [True], destructor_attachment
        """,
        gil=0,
        timeout=20,
        probe=True,
    )


def test_extension_import_keeps_gil_disabled_without_warning():
    _run_free_threaded(
        """
        import sys
        import sysconfig
        import warnings
        from pathlib import Path

        assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
        assert not sys._is_gil_enabled()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import cvcuda
        assert not sys._is_gil_enabled()
        assert not [w for w in caught if "global interpreter lock" in str(w.message).lower()]
        assert Path(cvcuda._cvcuda.__file__).name.endswith(sysconfig.get_config_var("EXT_SUFFIX"))
        """,
        gil=None,
    )


def test_extension_imports_in_free_threaded_build_with_gil_enabled():
    _run_free_threaded(
        """
        import sys
        import sysconfig
        from pathlib import Path

        assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
        assert sys._is_gil_enabled()
        import cvcuda
        assert sys._is_gil_enabled()
        assert Path(cvcuda._cvcuda.__file__).name.endswith(sysconfig.get_config_var("EXT_SUFFIX"))
        """,
        gil=1,
    )


@pytest.mark.parametrize(
    "source_kind",
    ["internal", "independent_luma", "independent_chroma"],
)
def test_image_cpu_waits_for_nondefault_producer_stream(source_kind):
    _run_free_threaded(
        f"""

        source_kind = {source_kind!r}
        if source_kind == "internal":
            image = cvcuda.Image((8, 6), cvcuda.Format.RGBA8)
            producer_resource = image
        else:
            source_planes = [
                cvcuda.Tensor((6, 8, 1), cvcuda.Type.U8, "HWC"),
                cvcuda.Tensor((3, 4, 2), cvcuda.Type.U8, "HWC"),
            ]
            image = cvcuda.as_image(
                [plane.cuda() for plane in source_planes],
                cvcuda.Format.NV12_ER,
            )
            producer_resource = source_planes[
                0 if source_kind == "independent_luma" else 1
            ]

        writer = cvcuda.Stream()
        failures = []
        cpu_done = threading.Event()

        def copy_to_cpu():
            try:
                probe.cvcudaTestReachReaderEntry()
                image.cpu("HWC")
            except BaseException as error:
                failures.append(error)
            finally:
                cpu_done.set()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(writer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        producer_resource.submitStreamSync(writer)

        probe.cvcudaTestArmReaderEntryGate(1)
        worker = threading.Thread(target=copy_to_cpu)
        worker.start()
        assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestReleaseReaderEntryGate()
        cpu_copy_bypassed_producer = cpu_done.wait(_cvcuda_test_timeout_scale)

        probe.cvcudaTestReleaseGpuGate()
        worker.join(_cvcuda_test_seconds(15))
        assert not worker.is_alive()
        assert failures == [], failures
        writer.sync()
        assert not cpu_copy_bypassed_producer, (
            source_kind
            + " Image.cpu() copied before its nondefault producer stream completed"
        )
        """,
        gil=0,
        timeout=30,
        probe=True,
    )


@pytest.mark.parametrize("workspace_kind", ["tensor", "image_batch"])
def test_morphology_workspace_writes_are_exclusive(workspace_kind):
    _run_free_threaded(
        f"""
        import numpy as np
        import cupy as cp

        workspace_kind = {workspace_kind!r}
        streams = [cvcuda.Stream(), cvcuda.Stream()]
        failures = [None, None]

        if workspace_kind == "tensor":
            shape = (1, 16, 16, 1)
            sources = [cvcuda.Tensor(shape, np.uint8, "NHWC") for _ in range(2)]
            destinations = [cvcuda.Tensor(shape, np.uint8, "NHWC") for _ in range(2)]
            workspace = cvcuda.Tensor(shape, np.uint8, "NHWC")

            def submit(index):
                cvcuda.morphology_into(
                    dst=destinations[index],
                    src=sources[index],
                    morphologyType=cvcuda.MorphologyType.DILATE,
                    maskSize=(3, 3),
                    anchor=(-1, -1),
                    workspace=workspace,
                    iteration=2,
                    stream=streams[index],
                )
        else:
            def make_batch():
                batch = cvcuda.ImageBatchVarShape(1)
                batch.pushback(cvcuda.Image((16, 16), cvcuda.Format.RGB8))
                return batch

            sources = [make_batch(), make_batch()]
            destinations = [make_batch(), make_batch()]
            workspace = make_batch()
            masks = cvcuda.as_tensor(cp.asarray(np.array([[3, 3]], dtype=np.int32)), "NC")
            anchors = cvcuda.as_tensor(cp.asarray(np.array([[-1, -1]], dtype=np.int32)), "NC")

            def submit(index):
                cvcuda.morphology_into(
                    dst=destinations[index],
                    src=sources[index],
                    morphologyType=cvcuda.MorphologyType.DILATE,
                    masks=masks,
                    anchors=anchors,
                    workspace=workspace,
                    iteration=2,
                    stream=streams[index],
                )

        def run(index):
            try:
                if index == 1:
                    probe.cvcudaTestReachReaderEntry()
                submit(index)
            except BaseException as error:
                failures[index] = error

        probe.cvcudaTestArmMorphologyGate()
        first = threading.Thread(target=run, args=(0,))
        second = threading.Thread(target=run, args=(1,))
        first.start()
        try:
            assert probe.cvcudaTestWaitForFirstMorphology(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            second.start()
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            second_reached_submit = (
                probe.cvcudaTestWaitForSecondMorphology(
                    int(1_000 * _cvcuda_test_timeout_scale)
                )
                == 1
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseMorphologyAndObserveSecond()

        assert probe.cvcudaTestWaitForSecondMorphology(_cvcuda_test_milliseconds(10_000)) == 1

        first.join(_cvcuda_test_seconds(15))
        if second.ident is not None:
            second.join(_cvcuda_test_seconds(15))
        assert not first.is_alive()
        assert not second.is_alive()
        assert failures == [None, None], failures
        for stream in streams:
            stream.sync()
        assert not second_reached_submit, "shared writable workspace was admitted concurrently"
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_pending_release_drain_detaches_from_free_threaded_gc():
    _run_free_threaded(
        r"""
        import gc
        import numpy as np

        stream = cvcuda.Stream()
        src = cvcuda.Tensor((1, 32, 32, 4), np.uint8, "NHWC")
        dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)

        cvcuda.flip_into(dst=dst, src=src, flipCode=0, stream=stream)
        stream.sync()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(stream.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        cvcuda.flip_into(dst=dst, src=src, flipCode=0, stream=stream)

        failures = [None]
        drain_started = threading.Event()
        drain_finished = threading.Event()

        def drain():
            try:
                drain_started.set()
                cvcuda.internal.drain_pending_releases()
            except BaseException as error:
                failures[0] = error
            finally:
                drain_finished.set()

        worker = threading.Thread(target=drain)
        worker.start()
        assert drain_started.wait(_cvcuda_test_seconds(10))
        assert not drain_finished.wait(0.1 * _cvcuda_test_timeout_scale)

        gc_returned = threading.Event()
        watchdog_released_gate = threading.Event()

        def release_if_gc_stalls():
            if not gc_returned.wait(_cvcuda_test_seconds(15)):
                watchdog_released_gate.set()
            probe.cvcudaTestReleaseGpuGate()

        watchdog = threading.Thread(target=release_if_gc_stalls)
        watchdog.start()

        gc.collect()
        gc_returned.set()

        worker.join(_cvcuda_test_seconds(10))
        watchdog.join(_cvcuda_test_seconds(10))
        assert not worker.is_alive()
        assert not watchdog.is_alive()
        assert failures == [None], failures
        assert not watchdog_released_gate.is_set(), (
            "gc.collect waited for a thread draining pending GPU releases"
        )
        """,
        gil=0,
        timeout=40,
        probe=True,
    )


def test_python_exceptions_are_isolated_between_concurrent_threads():
    _run_free_threaded(
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        import cupy as cp
        import cvcuda

        thread_count = 8
        iterations = 25
        barrier = threading.Barrier(thread_count)
        streams = [cvcuda.Stream() for _ in range(thread_count)]
        valid_sources = [
            cvcuda.as_tensor(
                cp.asarray(np.full((1, 8, 8, 1), index + 1, dtype=np.uint8)),
                "NHWC",
            )
            for index in range(thread_count)
        ]
        valid_destinations = [
            cvcuda.Tensor(source.shape, source.dtype, source.layout)
            for source in valid_sources
        ]
        invalid_sources = [
            cvcuda.Tensor((1, 8, 8, 1), np.int16, "NHWC") for _ in range(thread_count)
        ]
        invalid_destinations = [
            cvcuda.Tensor(source.shape, source.dtype, source.layout)
            for source in invalid_sources
        ]
        exception_counts = [0] * thread_count

        def run(index):
            barrier.wait()
            if index % 2 == 0:
                for _ in range(iterations):
                    cvcuda.flip_into(
                        src=valid_sources[index],
                        dst=valid_destinations[index],
                        flipCode=0,
                        stream=streams[index],
                    )
            else:
                for _ in range(iterations):
                    try:
                        cvcuda.flip_into(
                            src=invalid_sources[index],
                            dst=invalid_destinations[index],
                            flipCode=0,
                            stream=streams[index],
                        )
                    except RuntimeError as error:
                        assert "NVCV_ERROR_INVALID_ARGUMENT" in str(error)
                        assert "Invalid data type" in str(error)
                        exception_counts[index] += 1
                    else:
                        raise AssertionError("invalid operation did not raise")
                cvcuda.flip_into(
                    src=valid_sources[index],
                    dst=valid_destinations[index],
                    flipCode=0,
                    stream=streams[index],
                )
            streams[index].sync()

        with ThreadPoolExecutor(max_workers=thread_count) as workers:
            list(workers.map(run, range(thread_count)))
        assert exception_counts == [0, iterations, 0, iterations, 0, iterations, 0, iterations]

        for index, destination in enumerate(valid_destinations):
            expected = np.full((1, 8, 8, 1), index + 1, dtype=np.uint8)
            assert np.array_equal(cp.asarray(destination.cuda()).get(), expected)
        """,
        gil=0,
        timeout=75,
    )


@pytest.mark.parametrize("entrypoint", ["public", "capi"])
def test_submit_stream_sync_waits_for_inflight_writer_submission(entrypoint):
    _run_free_threaded(
        f"""
        import numpy as np

        entrypoint = {entrypoint!r}
        src = cvcuda.Tensor((1, 16, 16, 1), np.uint8, "NHWC")
        dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
        writer_stream = cvcuda.Stream()
        sync_stream = cvcuda.Stream()
        failures = [None, None]

        if entrypoint == "capi":
            capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
            capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            capsule_get_pointer.restype = ctypes.c_void_p
            capi = capsule_get_pointer(cvcuda._C_API, b"cvcuda._C_API")
            assert capi
            resource_submit_sync_offset = (
                cvcuda._test.capi_legacy_resource_submit_sync_offset()
            )
            expected_offset = 4 * ctypes.sizeof(ctypes.c_void_p)
            assert resource_submit_sync_offset == expected_offset, (
                "legacy C API Resource_SubmitSync moved from slot 4: "
                f"byte offset {{resource_submit_sync_offset}}, expected {{expected_offset}}"
            )
            function_pointer = ctypes.cast(
                capi + resource_submit_sync_offset,
                ctypes.POINTER(ctypes.c_void_p),
            )[0]
            assert function_pointer
            legacy_submit_sync = ctypes.PYFUNCTYPE(
                None, ctypes.py_object, ctypes.py_object
            )(function_pointer)

        def submit_writer():
            try:
                cvcuda.flip_into(dst=dst, src=src, flipCode=0, stream=writer_stream)
            except BaseException as error:
                failures[0] = error

        def release_writer():
            try:
                probe.cvcudaTestReleaseFlipAfterDelay(_cvcuda_test_milliseconds(2_000))
            except BaseException as error:
                failures[1] = error

        probe.cvcudaTestArmFlipGate()
        writer = threading.Thread(target=submit_writer)
        writer.start()
        try:
            assert probe.cvcudaTestWaitForFirstFlip(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmDelayedFlipRelease()
            releaser = threading.Thread(target=release_writer)
            releaser.start()
            assert probe.cvcudaTestWaitForDelayedFlipRelease(_cvcuda_test_milliseconds(10_000)) == 1
            if entrypoint == "public":
                dst.submitStreamSync(sync_stream)
            else:
                legacy_submit_sync(dst, sync_stream)
            writer_was_released_when_sync_returned = bool(
                probe.cvcudaTestDelayedFlipReleaseReached()
            )
        finally:
            probe.cvcudaTestReleaseFlip()

        writer.join(_cvcuda_test_seconds(15))
        releaser.join(_cvcuda_test_seconds(15))
        assert not writer.is_alive()
        assert not releaser.is_alive()
        assert failures == [None, None], failures
        writer_stream.sync()
        sync_stream.sync()
        assert writer_was_released_when_sync_returned, (
            f"{{entrypoint}} submitStreamSync bypassed the writer's submission lock"
        )
        """,
        gil=0,
        timeout=45,
        probe=True,
    )


def test_wait_stream_keeps_cached_event_record_and_wait_atomic():
    script = """

        producers = [cvcuda.Stream(), cvcuda.Stream()]
        consumer = cvcuda.Stream()
        failures = []

        def wait_on_producer(producer, position=False):
            try:
                if position:
                    probe.cvcudaTestReachReaderEntry()
                consumer.wait_stream(producer)
            except BaseException as error:
                failures.append(error)

        first = threading.Thread(target=wait_on_producer, args=(producers[0],))
        second = threading.Thread(
            target=wait_on_producer, args=(producers[1], True)
        )
        second_started = False
        probe.cvcudaTestArmEventRecordGate()
        try:
            first.start()
            assert probe.cvcudaTestWaitForEventRecordCount(1, _cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestArmReaderEntryGate(1)
            second.start()
            second_started = True
            assert probe.cvcudaTestWaitForReaderEntry(_cvcuda_test_milliseconds(10_000)) == 1
            probe.cvcudaTestReleaseReaderEntryGate()
            second_recorded_while_first_was_paused = bool(
                probe.cvcudaTestWaitForEventRecordCount(
                    2, int(1_000 * _cvcuda_test_timeout_scale)
                )
            )
        finally:
            probe.cvcudaTestReleaseReaderEntryGate()
            probe.cvcudaTestReleaseEventRecordGate()

        first.join(_cvcuda_test_seconds(10))
        if second_started:
            second.join(_cvcuda_test_seconds(10))
        assert not first.is_alive()
        assert not second.is_alive()
        assert failures == [], failures
        assert not second_recorded_while_first_was_paused, (
            "a concurrent wait_stream re-recorded the consumer's shared event "
            "before the first wait_stream installed its dependency"
        )
        consumer.sync()
    """
    _run_free_threaded(script, gil=0, timeout=35, probe=True)


def test_workspace_cache_first_use_and_reuse_are_thread_safe():
    _run_free_threaded(
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        import cupy as cp
        import cvcuda

        thread_count = 8
        iterations = 20
        barrier = threading.Barrier(thread_count)
        inputs = [
            cp.full((64, 64, 3), index + 1, dtype=cp.uint8)
            for index in range(thread_count)
        ]
        sources = [cvcuda.as_tensor(array, "HWC") for array in inputs]
        destinations = [
            cvcuda.Tensor((32, 32, 3), np.uint8, "HWC")
            for _ in range(thread_count)
        ]
        streams = [cvcuda.Stream() for _ in range(thread_count)]

        def resize(index):
            barrier.wait()
            for _ in range(iterations):
                result = cvcuda.hq_resize_into(
                    destinations[index],
                    sources[index],
                    min_interpolation=cvcuda.Interp.NEAREST,
                    mag_interpolation=cvcuda.Interp.NEAREST,
                    antialias=False,
                    stream=streams[index],
                )
                assert result is destinations[index]
            streams[index].sync()

        with ThreadPoolExecutor(max_workers=thread_count) as workers:
            list(workers.map(resize, range(thread_count)))
        for index, destination in enumerate(destinations):
            assert bool(cp.all(cp.asarray(destination.cuda()) == index + 1))
        """,
        gil=0,
        timeout=75,
    )


# Submission-lock probes use try_lock_submission to verify that every resource
# synchronization domain can be write-locked at the observation point.


def _image_cpu_submission_deadlock_script() -> str:
    return """
        import time
        image = cvcuda.Image((8, 8), cvcuda.Format.RGBA8)
        image_tensor = cvcuda.as_tensor(
            image.cuda(cvcuda.TensorLayout.HWC), "HWC"
        )

        source = cvcuda.Tensor(
            image_tensor.shape, image_tensor.dtype, image_tensor.layout
        )
        producer = cvcuda.Stream()
        writer_stream = cvcuda.Stream()
        failures = []
        start_writer = threading.Event()
        cpu_started = threading.Event()
        writer_done = threading.Event()

        def copy_to_cpu():
            cpu_started.set()
            try:
                image.cpu("HWC")
            except BaseException as error:
                failures.append(("cpu", error))

        def submit_writer():
            start_writer.wait()
            try:
                cvcuda.flip_into(
                    dst=image_tensor,
                    src=source,
                    flipCode=0,
                    stream=writer_stream,
                )
                probe.cvcudaTestReleaseGpuGate()
            except BaseException as error:
                failures.append(("writer", error))
            finally:
                writer_done.set()

        probe.cvcudaTestArmImageCpuCopyFailure()
        cvcuda.flip_into(
            dst=image_tensor, src=source, flipCode=0, stream=producer
        )
        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(producer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1

        cpu_worker = threading.Thread(target=copy_to_cpu)
        cpu_worker.start()
        assert cpu_started.wait(_cvcuda_test_seconds(10))

        read_lock_held = False
        first_copy_reached = False
        deadline = time.monotonic() + _cvcuda_test_seconds(10)
        while time.monotonic() < deadline:
            read_lock_held = not cvcuda._test.try_lock_submission(
                image
            )
            if read_lock_held:
                break
            first_copy_reached = bool(
                probe.cvcudaTestWaitForImageCpuFirstCopy(10)
            )
            if first_copy_reached:
                break
        assert read_lock_held or first_copy_reached, (
            "Image.cpu neither held its captured submission read lock nor "
            "reached its copy after the worker started"
        )

        writer_worker = threading.Thread(target=submit_writer)
        writer_worker.start()
        start_writer.set()

        try:
            made_progress_without_external_gpu_release = writer_done.wait(
                _cvcuda_test_seconds(15)
            )
        finally:
            probe.cvcudaTestReleaseImageCpuFirstCopy()
            probe.cvcudaTestReleaseGpuGate()
            cpu_worker.join(_cvcuda_test_seconds(15))
            writer_worker.join(_cvcuda_test_seconds(15))
        assert not cpu_worker.is_alive()
        assert not writer_worker.is_alive()
        producer.sync()
        writer_stream.sync()
        assert failures == [], failures
        assert made_progress_without_external_gpu_release, (
            "Image.cpu() held the submission read lock while waiting for its "
            "producer, preventing a writer from releasing that producer"
        )
        """


def _readonly_multidomain_export_deadlock_script() -> str:
    return """
        import cupy as cp
        class ReadOnlyPlane:
            def __init__(self, buffer):
                self.buffer = buffer

            @property
            def __cuda_array_interface__(self):
                interface = dict(self.buffer.__cuda_array_interface__)
                interface["data"] = (interface["data"][0], True)
                return interface

        storage = cp.zeros((3, 6, 8), dtype=cp.uint8)
        source_planes = [
            cvcuda.as_tensor(storage[index], "HW") for index in range(3)
        ]
        plane_exports = [plane.cuda() for plane in source_planes]

        image = cvcuda.as_image(
            [ReadOnlyPlane(plane_exports[0]), *plane_exports[1:]],
            cvcuda.Format.RGB8p,
        )
        exported = image.cuda("CHW")
        assert not isinstance(exported, list)
        assert exported.__cuda_array_interface__["data"][1]

        producer = cvcuda.Stream()
        writer = cvcuda.Stream()
        failures = []
        exported_state = []
        export_done = threading.Event()
        writer_done = threading.Event()

        def read_producer_state():
            try:
                exported_state.append(exported.__cuda_array_interface__)
            except BaseException as error:
                failures.append(("export", error))
            finally:
                export_done.set()

        def submit_writer():
            try:
                source_planes[1].submitStreamSync(writer)
                probe.cvcudaTestReleaseGpuGate()
            except BaseException as error:
                failures.append(("writer", error))
            finally:
                writer_done.set()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(producer.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        source_planes[1].submitStreamSync(producer)

        exporter = threading.Thread(target=read_producer_state)
        writer_worker = None
        exporter.start()
        try:
            assert export_done.wait(_cvcuda_test_seconds(10)), (
                "read-only multi-domain export did not complete while its "
                "producer remained gated"
            )
            exporter.join(_cvcuda_test_seconds(15))
            assert not exporter.is_alive()
            assert failures == [], failures
            assert len(exported_state) == 1
            handoff_stream = exported_state[0]["stream"]
            assert handoff_stream not in (None, 0, 1, 2, -1)
            assert cp.cuda.runtime.streamQuery(handoff_stream) != 0, (
                "read-only multi-domain export did not preserve its pending "
                "producer ordering on the advertised handoff stream"
            )

            writer_worker = threading.Thread(target=submit_writer)
            writer_worker.start()
            made_progress_without_external_gpu_release = writer_done.wait(
                _cvcuda_test_seconds(15)
            )
        finally:
            probe.cvcudaTestReleaseGpuGate()
            exporter.join(_cvcuda_test_seconds(15))
            if writer_worker is not None:
                writer_worker.join(_cvcuda_test_seconds(15))
        assert not exporter.is_alive()
        assert writer_worker is not None
        assert not writer_worker.is_alive()
        producer.sync()
        writer.sync()
        cp.cuda.runtime.streamSynchronize(handoff_stream)
        assert failures == [], failures
        assert made_progress_without_external_gpu_release, (
            "read-only multi-domain producer export held submission read "
            "locks while waiting for a producer"
        )
        """


def test_hold_failure_drain_releases_locks_before_waiting_with_the_gil_enabled():
    _run_free_threaded(
        """
        import numpy as np

        warmup_stream = cvcuda.Stream()
        warmup_src = cvcuda.Tensor((1, 1, 1, 1), np.uint8, "NHWC")
        warmup_dst = cvcuda.Tensor(warmup_src.shape, warmup_src.dtype, warmup_src.layout)
        cvcuda.flip_into(dst=warmup_dst, src=warmup_src, flipCode=0, stream=warmup_stream)
        warmup_stream.sync()

        stream = cvcuda.Stream()
        src = cvcuda.Tensor((1, 16, 16, 1), np.uint8, "NHWC")
        dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
        failures = [None, None]

        def submit_with_failed_hold():
            try:
                cvcuda.flip_into(dst=dst, src=src, flipCode=0, stream=stream)
            except BaseException as error:
                failures[0] = error

        def export_after_kernel_submission():
            try:
                assert probe.cvcudaTestWaitForFlipReturn(_cvcuda_test_milliseconds(10_000)) == 1
                dst.cuda()
            except BaseException as error:
                failures[1] = error
            finally:
                probe.cvcudaTestReleaseGpuGate()

        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(stream.handle) == 1
        assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
        probe.cvcudaTestArmFlipReturnProbe()
        cvcuda._test.fail_hold_resources(True)
        submitter = threading.Thread(target=submit_with_failed_hold)
        exporter = threading.Thread(target=export_after_kernel_submission)
        exporter.start()
        submitter.start()
        submitter_finished_before_recovery = False
        exporter_finished_before_recovery = False
        try:
            submitter.join(_cvcuda_test_seconds(15))
            submitter_finished_before_recovery = not submitter.is_alive()
            exporter.join(_cvcuda_test_seconds(15))
            exporter_finished_before_recovery = not exporter.is_alive()
        finally:
            cvcuda._test.fail_hold_resources(False)
            probe.cvcudaTestReleaseGpuGate()

        assert submitter_finished_before_recovery, (
            "hold-failure drain deadlocked with a waiting exporter"
        )
        assert exporter_finished_before_recovery, (
            "exporter deadlocked behind the hold-failure drain"
        )
        assert failures[1] is None, failures
        assert isinstance(failures[0], ValueError), failures
        assert "injected Stream_HoldResources failure" in str(failures[0])
        """,
        gil=1,
        timeout=40,
        probe=True,
    )


@pytest.mark.parametrize("gil", [0, 1])
def test_image_cpu_copy_failure_drains_pending_copy_before_unwind(gil):
    _run_free_threaded(
        """
        import gc
        import time
        import weakref
        image = cvcuda.Image((64, 48), cvcuda.Format.NV12_ER)
        image.cuda()
        producer = cvcuda.Stream()

        copy_done = threading.Event()
        failures = []

        def copy_to_cpu():
            try:
                image.cpu("HWC")
            except BaseException as error:
                failures.append(error)
            finally:
                copy_done.set()

        worker = threading.Thread(target=copy_to_cpu)
        probe.cvcudaTestArmImageCpuCopyFailure()
        probe.cvcudaTestArmGpuGate()
        assert probe.cvcudaTestQueueGpuGate(producer.handle) == 1
        try:
            assert probe.cvcudaTestWaitForGpuGate(_cvcuda_test_milliseconds(10_000)) == 1
            image.submitStreamSync(producer)
            worker.start()
            assert probe.cvcudaTestWaitForImageCpuFirstCopy(_cvcuda_test_milliseconds(10_000)) == 1
            assert probe.cvcudaTestImageCpuCopyApi() == 2
            assert probe.cvcudaTestWaitForImageCpuCopyFailure(_cvcuda_test_milliseconds(10_000)) == 1
            assert probe.cvcudaTestWaitForImageCpuCleanupSync(_cvcuda_test_milliseconds(10_000)) == 1
            assert probe.cvcudaTestImageCpuCopyPending() == 1
            assert not copy_done.is_set(), (
                "Image.cpu unwound after the cleanup sync was entered while "
                "its first plane copy was still pending"
            )

            submission_lock_released = False
            deadline = time.monotonic() + _cvcuda_test_seconds(10)
            while time.monotonic() < deadline:
                submission_lock_released = (
                    cvcuda._test.try_lock_submission(image)
                )
                if submission_lock_released:
                    break
                time.sleep(0.001)
            assert submission_lock_released, (
                "Image.cpu retained its resource read lock while draining "
                "the copy stream"
            )

            class Cycle:
                pass

            cycle = Cycle()
            cycle.link = cycle
            gc_completed = threading.Event()
            weakref.finalize(cycle, gc_completed.set)
            del cycle
            gc.collect()
            assert gc_completed.wait(_cvcuda_test_seconds(15)), (
                "Image.cpu held the Python runtime while its copy was pending"
            )

        finally:
            probe.cvcudaTestReleaseImageCpuFirstCopy()
            probe.cvcudaTestReleaseGpuGate()

        worker.join(_cvcuda_test_seconds(15))
        assert not worker.is_alive()
        producer.sync()
        assert len(failures) == 1, failures
        assert isinstance(failures[0], RuntimeError), failures
        assert "invalid argument" in str(failures[0]).lower(), failures
        """,
        gil=gil,
        timeout=40,
        probe=True,
    )


@pytest.mark.parametrize("gil", [0, 1])
def test_image_cpu_does_not_hold_submission_lock_while_waiting_for_copy(gil):
    _run_free_threaded(
        _image_cpu_submission_deadlock_script(),
        gil=gil,
        timeout=45,
        probe=True,
    )


def test_readonly_multidomain_export_does_not_wait_under_submission_locks():
    _run_free_threaded(
        _readonly_multidomain_export_deadlock_script(),
        gil=0,
        timeout=45,
        probe=True,
    )
