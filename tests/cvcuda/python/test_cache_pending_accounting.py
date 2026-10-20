# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pending allocation reuse must survive cache limits and worker exit."""

import sysconfig

import pytest

from test_resource_lifetime import _run_repro


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Free-threaded argument wrappers still retain consumed intermediates",
)
@pytest.mark.parametrize("batch", [1, 32])
def test_pending_preprocessing_pipeline_has_bounded_cache(batch):
    """Repeated native preprocessing reuses its warm pool before GPU completion."""
    _run_repro(
        f"""
        import gc
        import numpy as np

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        batch = {batch}
        iterations = 8
        input_shape = (batch, 64, 96, 3)
        resized_shape = (batch, 32, 48, 3)
        output_shape = (batch, 3, 32, 48)

        # Spatial constants survive Flip and PillowResize exactly. Vary each
        # submission, image, and channel to detect premature buffer overwrite.
        pixels = [
            (np.arange(batch * 3).reshape(batch, 1, 1, 3) + 7 * index).astype(np.uint8)
            for index in range(iterations)
        ]
        arrays = [cupy.asarray(np.broadcast_to(values, input_shape).copy()) for values in pixels]
        cupy.cuda.get_current_stream().synchronize()
        sources = [cvcuda.as_tensor(values, "NHWC") for values in arrays]
        snapshots = [cvcuda.Tensor(output_shape, cvcuda.Type.F32, "NCHW") for _ in pixels]

        def pipeline(source, snapshot):
            value = cvcuda.flip(source, 0, stream=stream)
            ids = [value.id]
            value = cvcuda.pillowresize(
                value, resized_shape, cvcuda.Format.RGB8, cvcuda.Interp.LINEAR, stream=stream
            )
            ids.append(value.id)
            value = cvcuda.convertto(value, cvcuda.Type.F32, scale=0.5, offset=1.0, stream=stream)
            ids.append(value.id)
            value = cvcuda.normalize(value, [1., 2., 3.], [.5, .25, .125], stream=stream)
            ids.append(value.id)
            value = cvcuda.reformat(value, "NCHW", stream=stream)
            ids.append(value.id)
            cvcuda.reformat_into(snapshot, value, stream=stream)
            return ids

        # Prime external imports separately: their first CAI use can sync the
        # device. Warm only one allocating pipeline, so growth cannot hide in
        # an oversized warm pool. Compile the park kernel before measuring.
        prime = cvcuda.Tensor(input_shape, cvcuda.Type.U8, "NHWC")
        for source in sources:
            cvcuda.flip_into(prime, source, 0, stream=stream)
        warmed_ids = set(pipeline(sources[0], snapshots[0]))
        _park_gpu(stream, 0)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        gc.collect()
        assert len(warmed_ids) == 5
        baseline_bytes = cvcuda.current_cache_size_inbytes()
        assert baseline_bytes > 0
        # Even allocating a fresh pipeline per iteration fits: pressure-driven
        # eviction must not rescue a regression in below-budget early reuse.
        cvcuda.set_cache_limit_inbytes(baseline_bytes * (iterations + 2))

        _park_gpu(stream, 2_000_000_000)
        for index, (source, snapshot) in enumerate(zip(sources, snapshots)):
            ids = pipeline(source, snapshot)
            current_bytes = cvcuda.current_cache_size_inbytes()
            assert set(ids) == warmed_ids, (index, ids, warmed_ids, current_bytes, baseline_bytes)
            assert current_bytes == baseline_bytes, (index, current_bytes, baseline_bytes)
        assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0

        stream.sync()
        for snapshot, values in zip(snapshots, pixels):
            expected = (values.astype(np.float32) * 0.5 + 1.0 - np.array([1., 2., 3.], np.float32))
            expected *= np.array([.5, .25, .125], np.float32)
            expected = np.broadcast_to(expected.transpose(0, 3, 1, 2), output_shape)
            np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(snapshot.cuda())), expected)
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Free-threaded argument wrappers still retain consumed intermediates",
)
@pytest.mark.parametrize("consumer_kind", ["direct", "wrapped"])
def test_same_stream_pending_tensor_reuses_below_cache_limit(consumer_kind):
    _run_repro(
        f"""
        import ctypes
        import gc

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        assert stream is not cvcuda.Stream.current
        consumer = (
            stream
            if {consumer_kind!r} == "direct"
            else cvcuda.as_stream(ctypes.c_void_p(int(stream.handle)))
        )
        assert int(consumer.handle) == int(stream.handle)
        shape = (1, 64, 64, 3)
        iterations = 6

        base = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(shape)
        expected = [(base + 17 * index).astype(cupy.uint8) for index in range(iterations)]
        cupy.cuda.get_current_stream().synchronize()
        sources = [cvcuda.as_tensor(values, "NHWC") for values in expected]
        snapshots = [cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC") for _ in expected]

        # Prime every external source before the pending phase. Its first CAI
        # import may conservatively synchronize the device because CuPy does
        # not identify the actual writer stream.
        for source, snapshot in zip(sources, snapshots):
            cvcuda.flip_into(snapshot, source, 0, stream=stream)
        warmed = cvcuda.flip(sources[0], 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        item_bytes = cvcuda.internal.nbytes_in_cache(warmed)
        del warmed
        gc.collect()

        baseline_bytes = cvcuda.current_cache_size_inbytes()
        limit = baseline_bytes + (iterations + 1) * item_bytes
        assert baseline_bytes + iterations * item_bytes < limit
        cvcuda.set_cache_limit_inbytes(limit)

        _park_gpu(stream, 2_000_000_000)
        identifiers = []
        for source, snapshot in zip(sources, snapshots):
            output = cvcuda.flip(source, 0, stream=consumer)
            identifiers.append(output.id)
            cvcuda.flip_into(snapshot, output, 0, stream=consumer)
            del output
            gc.collect()

        # Check the allocation identity first: a miss may synchronize the
        # finite GPU park, but it must not obscure the reuse failure.
        assert set(identifiers) == {{warmed_id}}, identifiers
        assert cvcuda.current_cache_size_inbytes() == baseline_bytes
        assert cupy.cuda.runtime.streamQuery(int(consumer.handle)) != 0

        consumer.sync()
        for snapshot, values in zip(snapshots, expected):
            assert cupy.array_equal(cupy.asarray(snapshot.cuda()), values)
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


@pytest.mark.parametrize("alias_kind", ["python_wrapper", "reshape"])
def test_below_limit_pending_reuse_excludes_live_alias(alias_kind):
    _run_repro(
        f"""
        import gc

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        assert stream is not cvcuda.Stream.current
        shape = (1, 64, 64, 3)
        first_values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(shape)
        second_values = 255 - first_values
        cupy.cuda.get_current_stream().synchronize()
        first_source = cvcuda.as_tensor(first_values, "NHWC")
        second_source = cvcuda.as_tensor(second_values, "NHWC")

        warmed = cvcuda.flip(first_source, 0, stream=stream)
        cvcuda.flip_into(warmed, second_source, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        item_bytes = cvcuda.internal.nbytes_in_cache(warmed)
        del warmed
        gc.collect()

        baseline_bytes = cvcuda.current_cache_size_inbytes()
        limit = baseline_bytes + 4 * item_bytes
        assert baseline_bytes + 2 * item_bytes < limit
        cvcuda.set_cache_limit_inbytes(limit)

        _park_gpu(stream, 2_000_000_000)
        first = cvcuda.flip(first_source, 0, stream=stream)
        assert first.id == warmed_id
        first_id = first.id
        if {alias_kind!r} == "python_wrapper":
            alias = first
        else:
            alias = first.reshape((1, 64 * 64, 3), "NWC")
        del first
        gc.collect()

        second = cvcuda.flip(second_source, 0, stream=stream)
        assert second.id != first_id, "pending reuse ignored a live allocation alias"

        stream.sync()
        expected_alias = cupy.flip(first_values, axis=1)
        if {alias_kind!r} == "reshape":
            expected_alias = expected_alias.reshape(1, 64 * 64, 3)
        assert cupy.array_equal(cupy.asarray(alias.cuda()), expected_alias)
        assert cupy.array_equal(cupy.asarray(second.cuda()), cupy.flip(second_values, axis=1))
        del alias, second
        gc.collect()
        cvcuda.internal.drain_pending_releases()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_below_limit_pending_reuse_rejects_mixed_stream_owners():
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        producer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        assert producer is not cvcuda.Stream.current
        assert consumer is not cvcuda.Stream.current
        assert int(producer.handle) != int(consumer.handle)
        shape = (1, 64, 64, 3)
        first_values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(shape)
        second_values = 255 - first_values
        cupy.cuda.get_current_stream().synchronize()
        first_source = cvcuda.as_tensor(first_values, "NHWC")
        second_source = cvcuda.as_tensor(second_values, "NHWC")
        first_snapshot = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")

        warmed = cvcuda.flip(first_source, 0, stream=producer)
        cvcuda.flip_into(first_snapshot, second_source, 0, stream=producer)
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        item_bytes = cvcuda.internal.nbytes_in_cache(warmed)
        del warmed
        gc.collect()

        baseline_bytes = cvcuda.current_cache_size_inbytes()
        limit = baseline_bytes + 4 * item_bytes
        assert baseline_bytes + 2 * item_bytes < limit
        cvcuda.set_cache_limit_inbytes(limit)

        _park_gpu(producer, 300_000_000)
        first = cvcuda.flip(first_source, 0, stream=producer)
        assert first.id == warmed_id
        first_id = first.id
        # The allocation now has a pending writer on producer and a pending
        # reader on consumer. Matching only one of those queues is insufficient
        # for below-limit early reuse.
        cvcuda.flip_into(first_snapshot, first, 0, stream=consumer)
        del first
        gc.collect()

        # No ready matching allocation remains. A miss may synchronize the
        # finite producer park while allocating, but matching the producer
        # owner must not hide the allocation's pending consumer owner.
        second = cvcuda.flip(second_source, 0, stream=producer)
        assert second.id != first_id, "below-limit reuse ignored a pending reader"

        consumer.sync()
        producer.sync()
        assert cupy.array_equal(cupy.asarray(first_snapshot.cuda()), first_values)
        assert cupy.array_equal(cupy.asarray(second.cuda()), cupy.flip(second_values, axis=1))
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_cross_stream_pending_pressure_uses_requested_size():
    _run_repro(
        """
        import gc
        import threading

        cvcuda.clear_cache()
        small_shape = (1, 32, 32, 1)
        large_shape = (1, 256, 256, 1)
        producer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        small_values = cupy.arange(32 * 32, dtype=cupy.uint8).reshape(small_shape)
        large_values = cupy.arange(256 * 256, dtype=cupy.uint8).reshape(large_shape)
        cupy.cuda.get_current_stream().synchronize()
        small_source = cvcuda.as_tensor(small_values, "NHWC")
        large_source = cvcuda.as_tensor(large_values, "NHWC")

        small_probe = cvcuda.Tensor(small_shape, cvcuda.Type.U8, "NHWC")
        small_bytes = cvcuda.internal.nbytes_in_cache(small_probe)
        del small_probe
        cvcuda.clear_cache()

        pending_id = []
        capacity_bytes = []
        failures = []

        def publish_pending():
            try:
                large = cvcuda.flip(large_source, 0, stream=producer)
                producer.sync()
                cvcuda.internal.drain_pending_releases()
                identifier = large.id
                capacity = cvcuda.internal.nbytes_in_cache(large)
                del large
                gc.collect()

                # Rebind the large allocation to the small descriptor before
                # publishing it to the global pending index on thread exit.
                small = cvcuda.flip(small_source, 0, stream=producer)
                assert small.id == identifier
                producer.sync()
                cvcuda.internal.drain_pending_releases()
                del small
                gc.collect()

                _park_gpu(producer, 2_000_000_000)
                pending = cvcuda.flip(small_source, 0, stream=producer)
                assert pending.id == identifier
                pending_id.append(identifier)
                capacity_bytes.append(capacity)
                del pending
                gc.collect()
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=publish_pending)
        worker.start()
        worker.join(timeout=20)
        assert not worker.is_alive()
        assert not failures, failures
        assert capacity_bytes[0] > small_bytes

        current = cvcuda.current_cache_size_inbytes()
        cvcuda.set_cache_limit_inbytes(current + small_bytes)

        # The requested small allocation fits below the limit. Its oversized
        # pending backing capacity must not manufacture pressure and serialize
        # an otherwise independent consumer stream.
        output = cvcuda.flip(small_source, 0, stream=consumer)
        assert output.id != pending_id[0], (
            output.id,
            pending_id,
            small_bytes,
            capacity_bytes,
        )

        consumer.sync()
        producer.sync()
        expected = cupy.flip(small_values, axis=1)
        assert cupy.array_equal(cupy.asarray(output.cuda()), expected)
        print("PASS", flush=True)
        """,
        timeout=40,
        cuda_prelude=True,
    )


def test_pending_tensor_remains_reusable_after_worker_exit():
    _run_repro(
        """
        import gc
        import threading

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source = cvcuda.as_tensor(cupy.zeros((1, 64, 64, 3), dtype=cupy.uint8), "NHWC")
        warmed = cvcuda.flip(source, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        limit = cvcuda.internal.nbytes_in_cache(warmed)
        del warmed
        cvcuda.clear_cache()
        cvcuda.set_cache_limit_inbytes(limit)

        identifiers = []
        failures = []
        def submit():
            try:
                output = cvcuda.flip(source, 0, stream=stream)
                identifiers.append(output.id)
            except BaseException as error:
                failures.append(error)

        _park_gpu(stream, 2_000_000_000)
        for _ in range(3):
            worker = threading.Thread(target=submit)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive()
            gc.collect()
        assert not failures, failures
        stream.sync()
        assert len(set(identifiers)) == 1, identifiers
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() <= limit
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Free-threaded argument wrappers still retain consumed intermediates",
)
@pytest.mark.parametrize("limit", [0, 1])
def test_oversized_pending_tensor_is_reused_without_retaining_ready_cache(limit):
    _run_repro(
        f"""
        import gc

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source = cvcuda.as_tensor(cupy.zeros((1, 64, 64, 3), dtype=cupy.uint8), "NHWC")
        warmed = cvcuda.flip(source, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        cvcuda.set_cache_limit_inbytes({limit})
        _park_gpu(stream, 2_000_000_000)
        cvcuda.flip_into(warmed, source, 0, stream=stream)
        expected_id = warmed.id
        del warmed
        gc.collect()

        output = cvcuda.flip(source, 0, stream=stream)
        assert output.id == expected_id, (output.id, expected_id)
        del output
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )
