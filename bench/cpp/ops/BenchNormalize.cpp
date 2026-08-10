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

    float    globalScale = 1.234f;
    float    globalShift = 2.345f;
    float    epsilon     = 12.34f;
    uint32_t flags       = CVCUDA_NORMALIZE_SCALE_IS_STDDEV;

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

    const long paramSamples = (isPlanar && inputKind == benchutils::InputKind::VarShape) ? 1 : srcShape.x;
    const long paramBytes   = scalarParams ? 0 : paramSamples * ch * sizeof(float) * 2;
    const long bytes        = srcShape.x * srcShape.y * srcShape.z * sizeof(T);
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

    cvcuda::Normalize op;

    // clang-format off

    if (scalarParams)
    {
        RunScalarNormalizeBench<BT>(state, op, srcShape, dstShape, ch, isPlanar, isFakePlanar, globalScale,
                                    globalShift, epsilon, flags);
        return;
    }

    // Fake-planar runs the interleaved kernel, so its base/scale are interleaved (NHWC) like the
    // interleaved path; only native planar (NCHW) uses planar params.
    nvcv::Tensor base = isPlanar ? nvcv::Tensor({{paramSamples, ch, 1, 1}, "NCHW"}, nvcv::TYPE_F32)
                                 : nvcv::Tensor({{paramSamples, 1, 1, ch}, "NHWC"}, nvcv::TYPE_F32);
    nvcv::Tensor scale = isPlanar ? nvcv::Tensor({{paramSamples, ch, 1, 1}, "NCHW"}, nvcv::TYPE_F32)
                                  : nvcv::Tensor({{paramSamples, 1, 1, ch}, "NHWC"}, nvcv::TYPE_F32);

    // base, scale: default-range LcgValues<float>() so the GPU LCG fast
    // path produces float [-1, +1] bytes identical to the Python bench's
    // create_tensor(..., fill_mode="lcg").
    benchutils::FillTensor<float>(base, benchutils::LcgValues<float>());
    benchutils::FillTensor<float>(scale, benchutils::LcgValues<float>());

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
