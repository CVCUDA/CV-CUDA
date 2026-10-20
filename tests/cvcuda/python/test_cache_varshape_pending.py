# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variable-shape outputs reuse storage without rewriting in-flight descriptors."""

import sysconfig

import pytest

from test_resource_lifetime import _run_repro


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Ordinary binding child-owner snapshots",
)
@pytest.mark.parametrize("scope", ["LOCAL", "GLOBAL"])
def test_live_varshape_reindexes_after_cache_clear(scope):
    _run_repro(
        f"""
        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source = cvcuda.ImageBatchVarShape(2)
        output = cvcuda.ImageBatchVarShape(2)
        for batch, fmt in ((source, cvcuda.Format.RGB8), (output, cvcuda.Format.RGBA8)):
            for _ in range(2):
                batch.pushback(cvcuda.Image((32, 32), fmt))
        expected = sum(
            cvcuda.internal.nbytes_in_cache(batch)
            + sum(cvcuda.internal.nbytes_in_cache(image) for image in batch)
            for batch in (source, output)
        )
        cvcuda.cvtcolor_into(output, source, cvcuda.ColorConversion.RGB2RGBA, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        for _ in range(3):
            cvcuda.clear_cache(cvcuda.ThreadScope.{scope})
            assert cvcuda.current_cache_size_inbytes() == 0
            with _hold_gpu(stream):
                cvcuda.cvtcolor_into(output, source, cvcuda.ColorConversion.RGB2RGBA, stream=stream)
                assert cvcuda.current_cache_size_inbytes() == expected
            stream.sync()
            cvcuda.internal.drain_pending_releases()
            assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Ordinary binding child-owner snapshots",
)
@pytest.mark.parametrize("mutation", ["clear", "popback"])
def test_pending_varshape_snapshot_mutation_and_shared_child_accounting(mutation):
    _run_repro(
        f"""
        import gc
        import numpy as np

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source = cvcuda.ImageBatchVarShape(1)
        shared = cvcuda.ImageBatchVarShape(1)
        old = cvcuda.Image((48, 32), cvcuda.Format.RGB8)
        replacement = cvcuda.Image((48, 32), cvcuda.Format.RGB8)
        cupy.asarray(old.cuda()).fill(17)
        cupy.asarray(replacement.cuda()).fill(93)
        source.pushback(old)
        shared.pushback(old)
        codes = cvcuda.as_tensor(cupy.zeros((1, 1), cupy.int32), "NC")
        cupy.cuda.get_current_stream().synchronize()
        outputs = [cvcuda.flip(source, codes, stream=stream) for _ in range(3)]
        cvcuda.flip_into(outputs[1], shared, codes, stream=stream)
        _park_gpu(stream, 0)
        stream.sync()
        cvcuda.internal.drain_pending_releases()

        def batch_bytes(batch):
            return cvcuda.internal.nbytes_in_cache(batch) + sum(
                cvcuda.internal.nbytes_in_cache(image) for image in batch
            )

        shared_descriptor_bytes = cvcuda.internal.nbytes_in_cache(shared)
        replacement_bytes = cvcuda.internal.nbytes_in_cache(replacement)
        output_bytes = [batch_bytes(output) for output in outputs]
        del old
        gc.collect()
        cvcuda.set_cache_limit_inbytes(0)
        assert cvcuda.current_cache_size_inbytes() == 0

        _park_gpu(stream, 2_000_000_000)
        cvcuda.flip_into(outputs[0], source, codes, stream=stream)
        first_bytes = cvcuda.current_cache_size_inbytes()
        cvcuda.flip_into(outputs[1], shared, codes, stream=stream)
        shared_bytes = cvcuda.current_cache_size_inbytes()
        # Two active snapshots of one Image must charge that allocation once.
        assert shared_bytes - first_bytes == shared_descriptor_bytes + output_bytes[1]

        if {mutation!r} == "clear":
            source.clear()
        else:
            source.popback(1)
        source.pushback(replacement)
        cvcuda.flip_into(outputs[2], source, codes, stream=stream)
        assert cvcuda.current_cache_size_inbytes() - shared_bytes == replacement_bytes + output_bytes[2]
        del shared, replacement
        gc.collect()
        assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0

        stream.sync()
        for output, value in zip(outputs, (17, 17, 93)):
            image = next(iter(output))
            np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(image.cuda())), value)
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Ordinary binding child-owner snapshots",
)
def test_shared_snapshot_survives_deferred_owner_retirement():
    _run_repro(
        """
        producer = cupy.cuda.Stream(non_blocking=True)
        first = cvcuda.as_stream(producer)
        second = cvcuda.Stream()
        retirement = cvcuda.Stream()
        source = cvcuda.ImageBatchVarShape(1)
        image = cvcuda.Image((48, 32), cvcuda.Format.RGB8)
        cupy.asarray(image.cuda()).fill(31)
        source.pushback(image)
        codes = cvcuda.as_tensor(cupy.zeros((1, 1), cupy.int32), "NC")
        cupy.cuda.get_current_stream().synchronize()
        outputs = [cvcuda.flip(source, codes, stream=first) for _ in range(2)]
        cvcuda.flip_into(outputs[1], source, codes, stream=second)
        # Leave shared resources ordered on first before parking second;
        # otherwise the next first submission would depend on that park.
        cvcuda.flip_into(outputs[0], source, codes, stream=first)
        _park_gpu(first, 0)
        first.sync()
        second.sync()
        cvcuda.internal.drain_pending_releases()
        cvcuda.set_cache_limit_inbytes(0)
        first_completed = cupy.cuda.Event()

        _park_gpu(first, 2_000_000_000)
        cvcuda.flip_into(outputs[0], source, codes, stream=first)
        first_completed.record(producer)
        _park_gpu(second, 10_000_000_000)
        cvcuda.flip_into(outputs[1], source, codes, stream=second)
        assert cvcuda.internal._pending_release_batch_count() == 2
        # Wait for this generation, not later cross-stream bookkeeping.
        # Its Python owner must defer while the second still holds the snapshot.
        first_completed.synchronize()
        assert cupy.cuda.runtime.streamQuery(int(second.handle)) != 0
        cvcuda._test.retire_pending_releases_off_gil(retirement)

        # Releasing the deferred hold must restore regular child ownership,
        # so the remaining record can retire without stranding cache charges.
        assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


@pytest.mark.parametrize(
    "submission",
    ["fresh", "append_new", "append_existing", "reuse_completed"],
)
def test_varshape_release_event_failure_rolls_back_owners(submission):
    _run_repro(
        f"""
        import gc
        import sysconfig
        import numpy as np

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source = cvcuda.ImageBatchVarShape(2)
        arrays = [
            np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
            for width, height in ((48, 32), (56, 36))
        ]
        source_values = [cupy.zeros(array.shape, cupy.uint8) for array in arrays]
        for values in source_values:
            source.pushback(cvcuda.as_image(values))
        codes = cvcuda.as_tensor(cupy.zeros((2, 1), cupy.int32), "NC")
        cupy.cuda.get_current_stream().synchronize()
        target = cvcuda.flip(source, codes, stream=stream)
        prime = cvcuda.flip(source, codes, stream=stream)
        _park_gpu(stream, 0)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        target_id = target.id
        image_ids = {{image.id for image in target}}
        for values, array in zip(source_values, arrays):
            values.set(array)
        cupy.cuda.get_current_stream().synchronize()

        # Prime descriptor uploads before the finite delay. Two successful
        # submissions force the third into the coalesced-record append path.
        if {submission!r} == "reuse_completed":
            cvcuda.flip_into(target, source, codes, stream=stream)
            # Finish the GPU work without retiring its ownership record.
            cupy.cuda.runtime.streamSynchronize(int(stream.handle))
        else:
            _park_gpu(stream, 2_000_000_000)
        if {submission!r}.startswith("append"):
            for _ in range(2):
                cvcuda.flip_into(
                    target if {submission!r} == "append_existing" else prime,
                    source, codes, stream=stream,
                )
        expected_records = {{"fresh": 0, "reuse_completed": 1}}.get({submission!r}, 2)
        assert cvcuda.internal._pending_release_batch_count() == expected_records
        ready = cupy.cuda.runtime.streamQuery(int(stream.handle)) == 0
        assert ready == ({submission!r} == "reuse_completed")

        cvcuda.internal._fail_next_release_event_record(True)
        try:
            try:
                cvcuda.flip_into(target, source, codes, stream=stream)
            except ValueError as error:
                assert "injected pending-release event record failure" in str(error)
            else:
                raise AssertionError("release-event failure was not propagated")
        finally:
            cvcuda.internal._fail_next_release_event_record(False)

        # The guard must drain already-submitted work before propagating the
        # record error; no explicit synchronization may precede this assertion.
        assert cupy.cuda.runtime.streamQuery(int(stream.handle)) == 0
        for image, array in zip(target, arrays):
            np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(image.cuda())), array[::-1])
        del image
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.internal._pending_release_batch_count() == 0
        del target
        gc.collect()

        recovered = cvcuda.flip(source, codes, stream=stream)
        if not sysconfig.get_config_var("Py_GIL_DISABLED"):
            assert recovered.id == target_id
            assert {{image.id for image in recovered}} == image_ids
        stream.sync()
        for image, array in zip(recovered, arrays):
            np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(image.cuda())), array[::-1])
        del image, recovered, prime, source, codes
        gc.collect()
        cvcuda.set_cache_limit_inbytes(0)
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Free-threaded argument wrappers still retain consumed intermediates",
)
@pytest.mark.parametrize("batch", [1, 32])
def test_pending_varshape_pipeline_has_bounded_cache(batch):
    _run_repro(
        f"""
        import gc
        import numpy as np

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        batch = {batch}
        iterations = 8
        sizes = [(96 + 2 * (index % 5), 64 + 2 * (index % 3)) for index in range(batch)]
        resized_sizes = [(width // 2, height // 2) for width, height in sizes]
        pixels = [
            (np.arange(batch * 3).reshape(batch, 3) + 4 + 7 * index).astype(np.uint8)
            for index in range(iterations)
        ]

        def make_batch(image_sizes, values=None):
            result = cvcuda.ImageBatchVarShape(batch)
            for index, (width, height) in enumerate(image_sizes):
                if values is None:
                    image = cvcuda.Image((width, height), cvcuda.Format.RGB8)
                else:
                    array = cupy.asarray(np.broadcast_to(values[index], (height, width, 3)).copy())
                    image = cvcuda.as_image(array, cvcuda.Format.RGB8)
                result.pushback(image)
            return result

        sources = [make_batch(sizes, values) for values in pixels]
        snapshots = [make_batch(resized_sizes) for _ in pixels]
        prime = make_batch(sizes)
        flip_codes = cvcuda.as_tensor(cupy.zeros((batch, 1), dtype=cupy.int32), "NC")
        base = cvcuda.as_tensor(cupy.asarray([1., 3., 2.], dtype=cupy.float32).reshape(1, 1, 1, 3), "NHWC")
        scale = cvcuda.as_tensor(cupy.ones((1, 1, 1, 3), dtype=cupy.float32), "NHWC")
        cupy.cuda.get_current_stream().synchronize()

        def identities(value):
            return {{value.id, *(image.id for image in value)}}

        def pipeline(source):
            value = cvcuda.flip(source, flip_codes, stream=stream)
            ids = identities(value)
            value = cvcuda.pillowresize(value, resized_sizes, cvcuda.Interp.LINEAR, stream=stream)
            ids.update(identities(value))
            value = cvcuda.normalize(value, base, scale, stream=stream)
            ids.update(identities(value))
            return value, ids

        # Prime every external import and snapshot descriptor before parking.
        # Only one allocating pipeline establishes the warm pool.
        for source in sources:
            cvcuda.flip_into(prime, source, flip_codes, stream=stream)
        warmed, warmed_ids = pipeline(sources[0])
        for snapshot in snapshots:
            cvcuda.flip_into(snapshot, warmed, flip_codes, stream=stream)
        del warmed
        _park_gpu(stream, 0)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        gc.collect()
        assert len(warmed_ids) == 3 * (batch + 1)
        baseline_bytes = cvcuda.current_cache_size_inbytes()
        assert baseline_bytes > 0
        cvcuda.set_cache_limit_inbytes(baseline_bytes * (iterations + 2))

        _park_gpu(stream, 2_000_000_000)
        for index, (source, snapshot) in enumerate(zip(sources, snapshots)):
            output, ids = pipeline(source)
            cvcuda.flip_into(snapshot, output, flip_codes, stream=stream)
            del output
            current_bytes = cvcuda.current_cache_size_inbytes()
            assert ids == warmed_ids, (index, ids, warmed_ids, current_bytes, baseline_bytes)
            assert current_bytes == baseline_bytes, (index, current_bytes, baseline_bytes)
        assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0

        stream.sync()
        for snapshot, values in zip(snapshots, pixels):
            for index, image in enumerate(snapshot):
                width, height = resized_sizes[index]
                expected = np.broadcast_to(values[index] - np.array([1, 3, 2], np.uint8), (height, width, 3))
                np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(image.cuda())), expected)
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Free-threaded argument wrappers still retain consumed intermediates",
)
@pytest.mark.parametrize("limit", [0, 1])
def test_pending_varshape_reuses_and_accounts_for_children_above_limit(limit):
    _run_repro(
        f"""
        import gc
        import numpy as np

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source = cvcuda.ImageBatchVarShape(2)
        for width, height in ((48, 32), (56, 36)):
            source.pushback(cvcuda.as_image(cupy.full((height, width, 3), 73, cupy.uint8)))
        codes = cvcuda.as_tensor(cupy.zeros((2, 1), cupy.int32), "NC")
        cupy.cuda.get_current_stream().synchronize()
        warmed = cvcuda.flip(source, codes, stream=stream)
        _park_gpu(stream, 0)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        image_ids = {{image.id for image in warmed}}
        output_bytes = cvcuda.internal.nbytes_in_cache(warmed) + sum(
            cvcuda.internal.nbytes_in_cache(image) for image in warmed
        )
        cvcuda.set_cache_limit_inbytes({limit})
        _park_gpu(stream, 2_000_000_000)
        cvcuda.flip_into(warmed, source, codes, stream=stream)
        del warmed
        gc.collect()
        pending_bytes = cvcuda.current_cache_size_inbytes()
        assert pending_bytes >= output_bytes

        for _ in range(4):
            output = cvcuda.flip(source, codes, stream=stream)
            assert output.id == warmed_id
            assert {{image.id for image in output}} == image_ids
            assert cvcuda.current_cache_size_inbytes() == pending_bytes
            del output
        assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )
