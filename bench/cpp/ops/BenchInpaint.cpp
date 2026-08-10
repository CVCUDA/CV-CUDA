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
#include "ops/generated/BenchInpaintConfig.hpp"

#include <cvcuda/OpInpaint.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <type_traits>

struct InpaintBenchLayout
{
    benchutils::InputKind inputKind;
    bool                  isPlanar;
    bool                  isFakePlanar;
    const char           *skipMessage;
};

template<typename T>
inline void AddInpaintMemoryCounters(nvbench::state &state, const int3 &shape, bool isFakePlanar)
{
    const long imageBytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const long maskBytes  = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(uint8_t);
    if (isFakePlanar)
    {
        // reformat(NCHW->NHWC) + inpaint + reformat(NHWC->NCHW), with the same NHWC mask stripe.
        state.add_global_memory_reads(3 * imageBytes + maskBytes);
        state.add_global_memory_writes(3 * imageBytes);
        return;
    }

    state.add_global_memory_reads(imageBytes + maskBytes);
    state.add_global_memory_writes(imageBytes);
}

template<typename BT>
inline nvcv::Tensor MakeInpaintTensor(const int3 &shape, int channels, bool isPlanar)
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

inline void FillInpaintMask(nvcv::Tensor &mask)
{
    // Mask semantics: 0 = known (don't inpaint), non-zero = needs inpainting.
    // Use deterministic stripe pattern (every other row) for exactly 50% coverage.
    benchutils::FillTensor<uint8_t>(mask, [](const long4_16a &c) -> uint8_t { return (c.y % 2 == 1) ? 1 : 0; });
}

template<typename BT>
inline void RunFakePlanarInpaintBench(nvbench::state &state, cvcuda::Inpaint &op, const int3 &shape, int channels,
                                      double inpaintRadius)
{
    nvcv::Tensor src      = MakeInpaintTensor<BT>(shape, channels, true);
    nvcv::Tensor interSrc = MakeInpaintTensor<BT>(shape, channels, false);
    nvcv::Tensor interDst = MakeInpaintTensor<BT>(shape, channels, false);
    nvcv::Tensor dst      = MakeInpaintTensor<BT>(shape, channels, true);
    nvcv::Tensor mask(
        {
            {shape.x, shape.y, shape.z, 1},
            "NHWC"
    },
        nvcv::TYPE_U8);

    benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());
    FillInpaintMask(mask);

    cvcuda::Reformat reformatOp;
    benchutils::warmup_and_exec(
        state, BENCH_INPAINT_WARMUP_ITERATIONS,
        [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &mask, inpaintRadius](cudaStream_t s)
        {
            reformatOp(s, src, interSrc);
            op(s, interSrc, mask, interDst, inpaintRadius);
            reformatOp(s, interDst, dst);
        });
}

template<typename BT>
inline void RunTensorInpaintBench(nvbench::state &state, cvcuda::Inpaint &op, const int3 &shape, int channels,
                                  bool isPlanar, double inpaintRadius)
{
    nvcv::Tensor src = MakeInpaintTensor<BT>(shape, channels, isPlanar);
    nvcv::Tensor dst = MakeInpaintTensor<BT>(shape, channels, isPlanar);
    nvcv::Tensor mask(
        {
            {shape.x, shape.y, shape.z, 1},
            "NHWC"
    },
        nvcv::TYPE_U8);

    benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());
    FillInpaintMask(mask);

    benchutils::warmup_and_exec(state, BENCH_INPAINT_WARMUP_ITERATIONS,
                                [&op, &src, &mask, &dst, inpaintRadius](cudaStream_t s)
                                { op(s, src, mask, dst, inpaintRadius); });
}

template<typename T>
inline void FillInpaintImageBatch(nvcv::ImageBatchVarShape &src, nvcv::ImageBatchVarShape &dst, const int3 &shape,
                                  bool isPlanar)
{
    if (constexpr int kChannels = nvcv::cuda::NumElements<T>; isPlanar && kChannels > 1)
    {
        benchutils::FillPlanarImageBatchLcg<T>(src, long2{shape.z, shape.y}, long2{0, 0});
        benchutils::FillPlanarImageBatchLike<T>(dst, src);
        return;
    }

    benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0}, benchutils::LcgValues<T>());
    // Use FillImageBatchLike to ensure dst has same per-sample shapes as src.
    benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
}

template<typename T, typename BT>
inline void RunVarShapeInpaintBench(nvbench::state &state, cvcuda::Inpaint &op, const int3 &shape, bool isPlanar,
                                    double inpaintRadius)
{
    if (isPlanar && std::is_same_v<BT, uint8_t> && nvcv::cuda::NumElements<T> == 4)
    {
        state.skip("uchar4 planar var-shape Inpaint benchmark is unsupported by the Python image API");
        return;
    }

    nvcv::ImageBatchVarShape src(shape.x);
    nvcv::ImageBatchVarShape dst(shape.x);
    nvcv::ImageBatchVarShape mask(shape.x);

    FillInpaintImageBatch<T>(src, dst, shape, isPlanar);

    // Mask semantics: 0 = known (don't inpaint), non-zero = needs inpainting.
    // Use deterministic stripe pattern (every other row) for exactly 50% coverage.
    benchutils::FillImageBatch<uint8_t>(mask, long2{shape.z, shape.y}, long2{0, 0},
                                        [](const long4_16a &c) -> uint8_t { return (c.y % 2 == 1) ? 1 : 0; });

    benchutils::warmup_and_exec(state, BENCH_INPAINT_WARMUP_ITERATIONS,
                                [&op, &src, &mask, &dst, inpaintRadius](cudaStream_t s)
                                { op(s, src, mask, dst, inpaintRadius); });
}

inline InpaintBenchLayout ValidateInpaintLayout(nvbench::state &state)
{
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        return {inputKind, isPlanar, isFakePlanar, "Inpaint benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"};
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        return {inputKind, isPlanar, isFakePlanar, "Fake-planar (NCHW_FAKE) Inpaint benchmark is tensor-only"};
    }
    return {inputKind, isPlanar, isFakePlanar, nullptr};
}

template<typename T, typename BT>
inline void ExecuteInpaintPath(nvbench::state &state, const int3 &shape, int channels, const InpaintBenchLayout &layout,
                               double inpaintRadius)
{
    AddInpaintMemoryCounters<T>(state, shape, layout.isFakePlanar);

    cvcuda::Inpaint op(shape.x, nvcv::Size2D{shape.z, shape.y});

    if (layout.isFakePlanar) // tensor-only: planar->interleaved->inpaint->interleaved->planar
    {
        RunFakePlanarInpaintBench<BT>(state, op, shape, channels, inpaintRadius);
        return;
    }
    if (layout.inputKind == benchutils::InputKind::Tensor)
    {
        RunTensorInpaintBench<BT>(state, op, shape, channels, layout.isPlanar, inpaintRadius);
        return;
    }

    RunVarShapeInpaintBench<T, BT>(state, op, shape, layout.isPlanar, inpaintRadius);
}

template<typename T>
inline void inpaint(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3 shape  = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto layout = ValidateInpaintLayout(state);
    if (layout.skipMessage != nullptr)
    {
        state.skip(layout.skipMessage);
        return;
    }

    // Use base type and channel count for consistent tensor creation (matches BenchHistogramEq pattern)
    using BT = typename nvcv::cuda::BaseType<T>;
    int ch   = nvcv::cuda::NumElements<T>;

    double inpaintRadius = state.get_float64("inpaintRadius");

    ExecuteInpaintPath<T, BT>(state, shape, ch, layout, inpaintRadius);
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(inpaint, NVBENCH_TYPE_AXES(BENCH_INPAINT_TYPES))
BENCH_INPAINT_AXES;
