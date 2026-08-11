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
#include "ops/generated/BenchCompositeConfig.hpp"

#include <cvcuda/OpComposite.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <stdexcept>
#include <string>
#include <string_view>

enum class CompositeLayout
{
    NHWC,
    NCHW,
    NCHW_FAKE,
    INVALID
};

inline bool IsPlanar(CompositeLayout layout)
{
    return layout == CompositeLayout::NCHW;
}

inline bool IsFakePlanar(CompositeLayout layout)
{
    return layout == CompositeLayout::NCHW_FAKE;
}

inline void ValidateOutChannels(int outChannels)
{
    if (outChannels == 3 || outChannels == 4)
    {
        return;
    }

    throw std::invalid_argument("Invalid outChannels = " + std::to_string(outChannels));
}

inline CompositeLayout ParseCompositeLayout(nvbench::state &state, std::string_view layout)
{
    if (layout == "NHWC")
    {
        return CompositeLayout::NHWC;
    }
    if (layout == "NCHW")
    {
        return CompositeLayout::NCHW;
    }
    if (layout == "NCHW_FAKE")
    {
        return CompositeLayout::NCHW_FAKE;
    }

    state.skip("Composite benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
    return CompositeLayout::INVALID;
}

inline CompositeLayout ValidateCompositeBenchParams(nvbench::state &state, std::string_view layout, int outChannels)
{
    ValidateOutChannels(outChannels);
    return ParseCompositeLayout(state, layout);
}

inline bool SkipUnsupportedCompositeCase(nvbench::state &state, benchutils::InputKind inputKind, CompositeLayout layout)
{
    if (IsFakePlanar(layout) && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) Composite benchmark is tensor-only");
        return true;
    }

    return false;
}

template<typename T, typename M>
void RunFakePlanarComposite(nvbench::state &state, cvcuda::Composite &op, int3 shape, int outChannels)
{
    using BT = typename nvcv::cuda::BaseType<T>;

    const int ch = nvcv::cuda::NumElements<T>;

    nvcv::Tensor fg(
        {
            {shape.x, ch, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor bg(
        {
            {shape.x, ch, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor mask(
        {
            {shape.x, 1, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<M>());
    nvcv::Tensor interFg(
        {
            {shape.x, shape.y, shape.z, ch},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor interBg(
        {
            {shape.x, shape.y, shape.z, ch},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor interMask(
        {
            {shape.x, shape.y, shape.z, 1},
            "NHWC"
    },
        benchutils::GetDataType<M>());
    nvcv::Tensor interDst(
        {
            {shape.x, shape.y, shape.z, outChannels},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor dst(
        {
            {shape.x, outChannels, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<BT>());

    benchutils::FillTensor<BT>(fg, benchutils::CheckerboardValues<BT>());
    benchutils::FillTensor<BT>(bg, benchutils::CheckerboardValues<BT>());
    benchutils::FillTensor<M>(mask, [](const long4_16a &) { return 1; });

    cvcuda::Reformat reformatOp;

    benchutils::warmup_and_exec(
        state, BENCH_COMPOSITE_WARMUP_ITERATIONS,
        [&op, &reformatOp, &fg, &bg, &mask, &interFg, &interBg, &interMask, &interDst, &dst](cudaStream_t s)
        {
            reformatOp(s, fg, interFg);
            reformatOp(s, bg, interBg);
            reformatOp(s, mask, interMask);
            op(s, interFg, interBg, interMask, interDst);
            reformatOp(s, interDst, dst);
        });
}

template<typename BT>
nvcv::Tensor MakeCompositeTensor(int3 shape, int channels, bool isPlanar)
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

template<typename T, typename M>
void RunTensorComposite(nvbench::state &state, cvcuda::Composite &op, int3 shape, bool isPlanar, int outChannels)
{
    using BT = typename nvcv::cuda::BaseType<T>;

    const int ch = nvcv::cuda::NumElements<T>;

    nvcv::Tensor fg   = MakeCompositeTensor<BT>(shape, ch, isPlanar);
    nvcv::Tensor bg   = MakeCompositeTensor<BT>(shape, ch, isPlanar);
    nvcv::Tensor mask = MakeCompositeTensor<M>(shape, 1, isPlanar);
    nvcv::Tensor dst  = MakeCompositeTensor<BT>(shape, outChannels, isPlanar);

    benchutils::FillTensor<BT>(fg, benchutils::CheckerboardValues<BT>());
    benchutils::FillTensor<BT>(bg, benchutils::CheckerboardValues<BT>());
    benchutils::FillTensor<M>(mask, [](const long4_16a &) { return 1; });

    benchutils::warmup_and_exec(state, BENCH_COMPOSITE_WARMUP_ITERATIONS,
                                [&op, &fg, &bg, &mask, &dst](cudaStream_t s) { op(s, fg, bg, mask, dst); });
}

template<typename T, typename M>
void FillCompositeVarShapeInputs(nvcv::ImageBatchVarShape &fg, nvcv::ImageBatchVarShape &bg,
                                 nvcv::ImageBatchVarShape &mask, int3 shape, bool isPlanar)
{
    if (isPlanar)
    {
        benchutils::FillPlanarImageBatch<T>(fg, long2{shape.z, shape.y}, long2{0, 0});
        benchutils::FillPlanarImageBatch<T>(bg, long2{shape.z, shape.y}, long2{0, 0});
    }
    else
    {
        benchutils::FillImageBatch<T>(fg, long2{shape.z, shape.y}, long2{0, 0}, benchutils::CheckerboardValues<T>());
        benchutils::FillImageBatchLike<T>(bg, fg, benchutils::CheckerboardValues<T>());
    }

    benchutils::FillImageBatchLike<M>(mask, fg, [](const long4_16a &) { return 1; });
}

template<typename OutT>
void FillCompositeVarShapeOutput(nvcv::ImageBatchVarShape &dst, const nvcv::ImageBatchVarShape &fg, bool isPlanar)
{
    if (isPlanar)
    {
        benchutils::FillPlanarImageBatchLike<OutT>(dst, fg);
    }
    else
    {
        nvcv::ImageFormat format;
        if constexpr (nvcv::cuda::NumElements<OutT> == 3)
        {
            format = nvcv::FMT_RGB8;
        }
        else
        {
            static_assert(nvcv::cuda::NumElements<OutT> == 4);
            format = nvcv::FMT_RGBA8;
        }
        benchutils::FillImageBatchLike<OutT>(
            dst, fg, [](const long4_16a &) { return OutT{0}; }, format);
    }
}

template<typename T, typename M>
void RunVarShapeComposite(nvbench::state &state, cvcuda::Composite &op, int3 shape, bool isPlanar, int outChannels)
{
    nvcv::ImageBatchVarShape fg(shape.x);
    nvcv::ImageBatchVarShape bg(shape.x);
    nvcv::ImageBatchVarShape mask(shape.x);
    nvcv::ImageBatchVarShape dst(shape.x);

    FillCompositeVarShapeInputs<T, M>(fg, bg, mask, shape, isPlanar);
    if (outChannels == 3)
    {
        FillCompositeVarShapeOutput<uchar3>(dst, fg, isPlanar);
    }
    else
    {
        FillCompositeVarShapeOutput<uchar4>(dst, fg, isPlanar);
    }

    benchutils::warmup_and_exec(state, BENCH_COMPOSITE_WARMUP_ITERATIONS,
                                [&op, &fg, &bg, &mask, &dst](cudaStream_t s) { op(s, fg, bg, mask, dst); });
}

template<typename T, typename M>
bool DispatchCompositeRun(nvbench::state &state, cvcuda::Composite &op, int3 shape, benchutils::InputKind inputKind,
                          CompositeLayout layout, int outChannels)
{
    if (SkipUnsupportedCompositeCase(state, inputKind, layout))
    {
        return false;
    }

    if (IsFakePlanar(layout))
    {
        RunFakePlanarComposite<T, M>(state, op, shape, outChannels);
        return true;
    }

    if (inputKind == benchutils::InputKind::Tensor)
    {
        RunTensorComposite<T, M>(state, op, shape, IsPlanar(layout), outChannels);
        return true;
    }

    RunVarShapeComposite<T, M>(state, op, shape, IsPlanar(layout), outChannels);
    return true;
}

template<typename T, typename M = uint8_t>
inline void composite(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape       = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind   = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layout      = benchutils::GetStringParam(state, "layout", "NHWC");
    int                         outChannels = benchutils::GetIntParam<int>(state, "outChannels");

    CompositeLayout compositeLayout = ValidateCompositeBenchParams(state, layout, outChannels);
    if (compositeLayout == CompositeLayout::INVALID)
    {
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * (sizeof(T) * 2 + sizeof(M)));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * outChannels * sizeof(BT));

    cvcuda::Composite op;
    DispatchCompositeRun<T, M>(state, op, shape, inputKind, compositeLayout, outChannels);
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(composite, NVBENCH_TYPE_AXES(BENCH_COMPOSITE_TYPES))
BENCH_COMPOSITE_AXES;
