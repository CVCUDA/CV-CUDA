/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../CppBenchUtils.hpp"
#include "ops/generated/BenchNormalizeConfig.hpp"

#include <cvcuda/OpNormalize.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename BT>
nvcv::Tensor MakeNormalizeTensor(const long3 &shape, int channels, bool isPlanar)
{
    if (isPlanar)
    {
        return nvcv::Tensor(
            {
                {shape.x, channels, shape.y, shape.z},
                "NCHW"
        },
            benchutils::GetDataType<BT>());
    }
    return nvcv::Tensor(
        {
            {shape.x, shape.y, shape.z, channels},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
}

template<typename BT>
void RunScalarNormalizeBench(nvbench::state &state, cvcuda::Normalize &op, const long3 &srcShape, const long3 &dstShape,
                             int channels, bool isPlanar, bool isFakePlanar, float globalScale, float globalShift,
                             float epsilon, uint32_t flags)
{
    if (isFakePlanar)
    {
        state.skip("Scalar-parameter normalize benchmark supports only native NHWC and NCHW tensors");
        return;
    }

    auto src = MakeNormalizeTensor<BT>(srcShape, channels, isPlanar);
    auto dst = MakeNormalizeTensor<BT>(dstShape, channels, isPlanar);
    benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

    const float4 baseValue  = {0.25f, 0.5f, 0.75f, 1.f};
    const float4 scaleValue = {1.25f, 1.5f, 1.75f, 2.f};

    benchutils::warmup_and_exec(
        state, BENCH_NORMALIZE_WARMUP_ITERATIONS,
        [&op, &src, &baseValue, &scaleValue, channels, &dst, globalScale, globalShift, epsilon,
         flags](cudaStream_t stream)
        { op(stream, src, baseValue, scaleValue, channels, channels, dst, globalScale, globalShift, epsilon, flags); });
}

template<typename T>
void FillStridedNormalizeImages(nvcv::ImageBatchVarShape &batch, const nvcv::TensorDataStridedCuda::Buffer &buffer,
                                const long3 &shape, bool fillSource)
{
    using BT                   = typename nvcv::cuda::BaseType<T>;
    constexpr int channels     = nvcv::cuda::NumElements<T>;
    const int64_t rowStride    = buffer.strides[2];
    const int64_t planeStride  = buffer.strides[1];
    const int64_t sampleStride = buffer.strides[0];
    for (int sample = 0; sample < shape.x; ++sample)
    {
        nvcv::ImageDataStridedCuda::Buffer image{};
        image.numPlanes = channels;
        for (int channel = 0; channel < channels; ++channel)
        {
            auto &plane     = image.planes[channel];
            plane.width     = shape.z;
            plane.height    = shape.y;
            plane.rowStride = rowStride;
            plane.basePtr   = buffer.basePtr + sample * sampleStride + channel * planeStride;
            if (fillSource)
                benchutils::launchCheckerboardImageTyped<BT>(
                    reinterpret_cast<BT *>(plane.basePtr), planeStride / sizeof(BT), rowStride / sizeof(BT), 1,
                    benchutils::DefaultRangeMax<BT>(), static_cast<BT>(0), nullptr);
        }
        batch.pushBack(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{benchutils::GetPlanarFormat<T>(), image}));
    }
}

template<typename T>
void RunStridedNormalizeBench(nvbench::state &state, cvcuda::Normalize &op, const long3 &shape,
                              benchutils::InputKind inputKind, const nvcv::Tensor &base, const nvcv::Tensor &scale,
                              float globalScale, float globalShift, float epsilon, uint32_t flags)
{
    using BT                              = typename nvcv::cuda::BaseType<T>;
    constexpr int            channels     = nvcv::cuda::NumElements<T>;
    const int64_t            rowStride    = shape.z * sizeof(BT) + 2;
    const int64_t            planeStride  = rowStride * shape.y;
    const int64_t            sampleStride = planeStride * channels;
    // Owners outlive every wrapped view and all asynchronous benchmark invocations.
    nvcv::Tensor             owners[2], tensors[2];
    nvcv::ImageBatchVarShape batches[2]{nvcv::ImageBatchVarShape(shape.x), nvcv::ImageBatchVarShape(shape.x)};
    for (int side = 0; side < 2; ++side)
    {
        owners[side] = nvcv::Tensor({{2 + sampleStride * shape.x}, "W"}, nvcv::TYPE_U8);
        auto data    = owners[side].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);
        CUDA_CHECK_ERROR(cudaMemset(data->basePtr(), 0, 2 + sampleStride * shape.x));
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = reinterpret_cast<NVCVByte *>(data->basePtr()) + 2;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = planeStride;
        buffer.strides[2] = rowStride;
        buffer.strides[3] = sizeof(BT);
        tensors[side]     = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            {{shape.x, channels, shape.y, shape.z}, "NCHW"},
            benchutils::GetDataType<BT>(),
            buffer
        });
        if (inputKind == benchutils::InputKind::Tensor)
        {
            if (side == 0)
                benchutils::FillTensor<BT>(tensors[side], benchutils::CheckerboardValues<BT>());
        }
        else
        {
            FillStridedNormalizeImages<T>(batches[side], buffer, shape, side == 0);
        }
    }
    CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
    benchutils::warmup_and_exec(
        state, BENCH_NORMALIZE_WARMUP_ITERATIONS,
        [&op, &tensors, &batches, &base, &scale, &inputKind, &globalScale, &globalShift, &epsilon,
         &flags](cudaStream_t stream)
        {
            if (inputKind == benchutils::InputKind::Tensor)
                op(stream, tensors[0], base, scale, tensors[1], globalScale, globalShift, epsilon, flags);
            else
                op(stream, batches[0], base, scale, batches[1], globalScale, globalShift, epsilon, flags);
        });
}

template<typename T>
bool ValidateNormalizeStorage(nvbench::state &state, const std::string &storage, bool isPlanar, bool scalarParams,
                              benchutils::InputKind inputKind, bool isSpatial)
{
    using BT                = typename nvcv::cuda::BaseType<T>;
    constexpr int channels  = nvcv::cuda::NumElements<T>;
    const bool    isStrided = storage == "Offset2Pitch2";
    if (storage != "Native" && storage != "NativePitch" && !isStrided)
    {
        state.skip("Normalize benchmark supports only Native, NativePitch, and Offset2Pitch2 storage");
        return false;
    }
    if (storage == "NativePitch" && (!isPlanar || scalarParams || inputKind != benchutils::InputKind::Tensor))
    {
        state.skip("NativePitch storage requires native planar Tensor");
        return false;
    }
    if ((isStrided && (!isPlanar || scalarParams || channels != 3 || benchutils::GetDataType<BT>() != nvcv::TYPE_F16))
        || (isSpatial && (!isPlanar || scalarParams || inputKind != benchutils::InputKind::Tensor)))
    {
        state.skip("Offset2Pitch2 requires planar F16 C3; spatial parameters require native planar Tensor");
        return false;
    }
    return true;
}

template<typename T>
void AddNormalizeMemoryCounts(nvbench::state &state, const long3 &srcShape, const long3 &dstShape, long paramSamples,
                              bool scalarParams, bool isSpatial, bool isFakePlanar)
{
    constexpr int channels   = nvcv::cuda::NumElements<T>;
    const long    paramBytes = scalarParams ? 0
                             : isSpatial    ? (srcShape.x * channels * srcShape.y + srcShape.z) * sizeof(float)
                                            : paramSamples * channels * sizeof(float) * 2;
    const long    bytes      = srcShape.x * srcShape.y * srcShape.z * sizeof(T);
    if (isFakePlanar)
    {
        // reformat(NCHW→NHWC) + normalize + reformat(NHWC→NCHW): reads src + interleaved src + dst
        // (plus base/scale once), writes interleaved src + interleaved dst + dst. Normalize preserves
        // size, so every transfer moves `bytes`.
        state.add_global_memory_reads(3 * bytes + paramBytes);
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes + paramBytes);
        state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));
    }
}

inline void InitNormalizeParams(nvcv::Tensor &base, nvcv::Tensor &scale, const long3 &srcShape, long paramSamples,
                                int channels, bool isSpatial, bool isStrided, bool isPlanar)
{
    // Fake-planar runs the interleaved kernel, so its base/scale are interleaved (NHWC) like the
    // interleaved path; only native planar (NCHW) uses planar params.
    // Strided/spatial parameters are tight to match Python's logical LCG sequence;
    // other modes retain their padded allocation.
    const bool tightParams = isSpatial || isStrided;
    base                   = tightParams ? nvcv::Tensor(
                                 {
                                     {paramSamples, channels, isSpatial ? srcShape.y : 1, 1},
                                     "NCHW"
    },
                                 nvcv::TYPE_F32, nvcv::MemAlignment{}.rowAddr(1))
                                         : MakeNormalizeTensor<float>({paramSamples, 1, 1}, channels, isPlanar);
    scale                  = tightParams ? nvcv::Tensor(
                                 {
                                     {isSpatial ? 1 : paramSamples, isSpatial ? 1 : channels, 1, isSpatial ? srcShape.z : 1},
                                     "NCHW"
    },
                                 nvcv::TYPE_F32, nvcv::MemAlignment{}.rowAddr(1))
                                         : MakeNormalizeTensor<float>({paramSamples, 1, 1}, channels, isPlanar);
    benchutils::FillTensor<float>(base, benchutils::LcgValues<float>());
    benchutils::FillTensor<float>(scale, benchutils::LcgValues<float>());
}

template<typename T>
inline void normalize(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                       srcShape      = benchutils::GetShape<3>(state.get_string("shape"));
    auto                        layout        = benchutils::GetStringParam(state, "layout", "NHWC");
    const std::string           inputKindName = state.get_string("inputKind");
    const bool                  scalarParams  = inputKindName == "TensorScalar";
    const benchutils::InputKind inputKind
        = scalarParams ? benchutils::InputKind::Tensor : benchutils::GetInputKind(inputKindName);
    long3 dstShape = srcShape;

    // A per-channel (base, scale) set that is the same for every sample -- one mean/stddev triple for
    // the whole batch, which is how a model's input normalization is expressed -- is a different
    // dispatch from one parameter set per sample: it lets an implementation resolve the parameters
    // once per thread instead of per sample. "PerSample" is the historical shape of this benchmark.
    const auto paramKind   = benchutils::GetStringParam(state, "paramKind", "PerSample");
    const bool isBroadcast = paramKind == "Broadcast";
    const bool isSpatial   = paramKind == "RowBaseColumnScale";
    if (paramKind != "PerSample" && paramKind != "Broadcast" && !isSpatial)
    {
        state.skip("Normalize benchmark supports only PerSample, Broadcast, and RowBaseColumnScale paramKind");
        return;
    }

    // Both halves of the flag are separately dispatched, so both need measuring.
    const auto scaleMode = benchutils::GetStringParam(state, "scaleMode", "stddev");
    if (scaleMode != "stddev" && scaleMode != "scale")
    {
        state.skip("Normalize benchmark supports only stddev and scale scaleMode");
        return;
    }

    float    globalScale = 1.234f;
    float    globalShift = 2.345f;
    float    epsilon     = 12.34f;
    uint32_t flags       = scaleMode == "stddev" ? CVCUDA_NORMALIZE_SCALE_IS_STDDEV : 0u;

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("Normalize benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // NCHW_FAKE ("fake planar") is a tensor-only comparison path: planar data is reformatted to
    // interleaved, normalized with the interleaved kernel, and reformatted back to planar — all timed
    // together — so the native planar path (NCHW) can be shown to be faster than this naive
    // convert→normalize→convert pipeline.
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) normalize benchmark is tensor-only");
        return;
    }
    // TensorScalar passes base/scale by value, so it has no parameter tensor to broadcast and the
    // Broadcast row would just duplicate the PerSample one.
    if (isBroadcast && scalarParams)
    {
        state.skip("Broadcast paramKind does not apply to by-value (TensorScalar) parameters");
        return;
    }
    const auto storage   = benchutils::GetStringParam(state, "storage", "Native");
    const bool isStrided = storage == "Offset2Pitch2";
    if (!ValidateNormalizeStorage<T>(state, storage, isPlanar, scalarParams, inputKind, isSpatial))
    {
        return;
    }

    // The native-planar var-shape path has always broadcast its parameters, so it is unaffected.
    const long paramSamples
        = (isBroadcast || (isPlanar && inputKind == benchutils::InputKind::VarShape)) ? 1 : srcShape.x;
    AddNormalizeMemoryCounts<T>(state, srcShape, dstShape, paramSamples, scalarParams, isSpatial, isFakePlanar);

    cvcuda::Normalize op;

    // clang-format off

    if (scalarParams)
    {
        RunScalarNormalizeBench<BT>(state, op, srcShape, dstShape, ch, isPlanar, isFakePlanar, globalScale,
                                    globalShift, epsilon, flags);
        return;
    }

    nvcv::Tensor base, scale;
    InitNormalizeParams(base, scale, srcShape, paramSamples, ch, isSpatial, isStrided, isPlanar);

    if (isStrided)
    {
        RunStridedNormalizeBench<T>(state, op, srcShape, inputKind, base, scale, globalScale, globalShift, epsilon, flags);
        return;
    }

    if (isFakePlanar) // tensor-only: planar→interleaved→normalize→interleaved→planar
    {
        nvcv::Tensor src     ({{srcShape.x, ch, srcShape.y, srcShape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{srcShape.x, srcShape.y, srcShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{dstShape.x, dstShape.y, dstShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{dstShape.x, ch, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_NORMALIZE_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &base, &scale, &globalScale, &globalShift, &epsilon, &flags](cudaStream_t s) {
                reformatOp(s, src, interSrc);                                              // NCHW → NHWC
                op(s, interSrc, base, scale, interDst, globalScale, globalShift, epsilon, flags); // interleaved
                reformatOp(s, interDst, dst);                                              // NHWC → NCHW
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        auto src = MakeNormalizeTensor<BT>(srcShape, ch, isPlanar);
        auto dst = MakeNormalizeTensor<BT>(dstShape, ch, isPlanar);

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_NORMALIZE_WARMUP_ITERATIONS,
            [&op, &src, &base, &scale, &dst, &globalScale, &globalShift, &epsilon, &flags](cudaStream_t s) {
                op(s, src, base, scale, dst, globalScale, globalShift, epsilon, flags);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(static_cast<int32_t>(srcShape.x));
        nvcv::ImageBatchVarShape dst(static_cast<int32_t>(dstShape.x));

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0});
            benchutils::FillPlanarImageBatchLike<T>(dst, src);
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        benchutils::warmup_and_exec(state, BENCH_NORMALIZE_WARMUP_ITERATIONS,
            [&op, &src, &base, &scale, &dst, &globalScale, &globalShift, &epsilon, &flags](cudaStream_t s) {
                op(s, src, base, scale, dst, globalScale, globalShift, epsilon, flags);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(normalize, NVBENCH_TYPE_AXES(BENCH_NORMALIZE_TYPES))
BENCH_NORMALIZE_AXES;
