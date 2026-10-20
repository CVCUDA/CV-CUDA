# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for cache reuse versus a resource's stream history."""

from test_resource_lifetime import _run_repro


def test_pending_reuse_rejects_completed_reader_with_new_unrelated_work():
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        producer = cvcuda.Stream()
        reader = cvcuda.Stream()
        assert producer is not cvcuda.Stream.current
        assert reader is not cvcuda.Stream.current
        assert int(producer.handle) != int(reader.handle)

        shape = (1, 64, 64, 3)
        values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(shape)
        next_values = (255 - values).astype(cupy.uint8)
        unrelated_values = (values + 37).astype(cupy.uint8)
        cupy.cuda.get_current_stream().synchronize()

        source = cvcuda.as_tensor(values, "NHWC")
        next_source = cvcuda.as_tensor(next_values, "NHWC")
        unrelated_source = cvcuda.as_tensor(unrelated_values, "NHWC")
        unrelated_output = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        reader_snapshot = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")

        # Prime every external input and retain a ready spare so rejection below
        # cannot call cudaMalloc while the producer is held.
        cvcuda.flip_into(unrelated_output, next_source, 0, stream=producer)
        cvcuda.flip_into(reader_snapshot, unrelated_source, 0, stream=producer)
        warmed = [cvcuda.flip(source, 0, stream=producer) for _ in range(2)]
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        item_bytes = cvcuda.internal.nbytes_in_cache(warmed[0])
        del warmed
        gc.collect()

        baseline_bytes = cvcuda.current_cache_size_inbytes()
        limit = baseline_bytes + 4 * item_bytes
        assert baseline_bytes + 2 * item_bytes < limit
        cvcuda.set_cache_limit_inbytes(limit)

        candidate = cvcuda.flip(source, 0, stream=producer)
        candidate_id = candidate.id
        cvcuda.flip_into(reader_snapshot, candidate, 0, stream=reader)
        reader.sync()
        cvcuda.internal.drain_pending_releases()

        with _hold_gpu(producer):
            # Model the remaining producer-generation hold after the reader has
            # retired, without changing the resource's reader-side history.
            cvcuda._test.hold_resources_on_device(
                producer, [candidate], cupy.cuda.runtime.getDevice()
            )
            del candidate
            gc.collect()
            remaining_batches = cvcuda.internal._pending_release_batch_count()
            assert remaining_batches == 1, remaining_batches
            assert cupy.cuda.runtime.streamQuery(int(producer.handle)) != 0

            # These resources last used producer, so new unrelated reader work
            # waits on its held tail without requiring a second host callback.
            cvcuda.flip_into(unrelated_output, unrelated_source, 0, stream=reader)
            assert cupy.cuda.runtime.streamQuery(int(reader.handle)) != 0
            replacement = cvcuda.flip(next_source, 0, stream=producer)
            assert replacement.id != candidate_id, (
                "cache reused an allocation whose last-stream history was stale",
                replacement.id,
                candidate_id,
            )

        producer.sync()
        reader.sync()
        assert cupy.array_equal(cupy.asarray(reader_snapshot.cuda()), values)
        assert cupy.array_equal(
            cupy.asarray(replacement.cuda()), cupy.flip(next_values, axis=1)
        )
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )
