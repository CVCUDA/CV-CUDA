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
#include "ops/generated/BenchChannelReorderConfig.hpp"

#include <cvcuda/OpChannelReorder.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <string_view>

namespace {

enum class ChannelReorderLayout
{
    NHWC,
    NCHW,
    NCHWFake,
    Invalid
};

enum class ChannelOrderPattern
{
    Rotate,
    ZeroFill
};

ChannelReorderLayout GetChannelReorderLayout(nvbench::state &state, std::string_view layout,
                                             benchutils::InputKind inputKind)
{
    if (layout == "NHWC")
    {
        return ChannelReorderLayout::NHWC;
    }
    if (layout == "NCHW")
    {
        return ChannelReorderLayout::NCHW;
    }
    if (layout == "NCHW_FAKE")
    {
        if (inputKind == benchutils::InputKind::VarShape)
        {
            state.skip("Fake-planar (NCHW_FAKE) ChannelReorder benchmark is tensor-only");
            return ChannelReorderLayout::Invalid;
        }
        return ChannelReorderLayout::NCHWFake;
    }

    state.skip("ChannelReorder benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
    return ChannelReorderLayout::Invalid;
}

bool IsPlanar(ChannelReorderLayout layout)
{
    return layout == ChannelReorderLayout::NCHW;
}

bool IsFakePlanar(ChannelReorderLayout layout)
{
    return layout == ChannelReorderLayout::NCHWFake;
}

ChannelOrderPattern GetChannelOrderPattern(std::string_view orderPattern)
{
    if (orderPattern == "rotate")
    {
        return ChannelOrderPattern::Rotate;
    }
    if (orderPattern == "zero_fill")
    {
        return ChannelOrderPattern::ZeroFill;
    }

    throw std::invalid_argument("Invalid orderPattern = " + std::string(orderPattern));
}

template<int NumChannels>
std::array<int32_t, NumChannels> MakeChannelOrder(ChannelOrderPattern pattern)
{
    std::array<int32_t, NumChannels> order;
    std::iota(order.begin(), order.end(), 0);
    if (pattern == ChannelOrderPattern::Rotate)
    {
        std::rotate(order.begin(), order.begin() + 1, order.end());
    }
    else if constexpr (NumChannels > 1)
    {
        order[1] = -1;
    }
    return order;
}

void AddChannelReorderMemoryTraffic(nvbench::state &state, long bytes, ChannelReorderLayout layout)
{
    const long traffic = IsFakePlanar(layout) ? 3 * bytes : bytes;
    state.add_global_memory_reads(traffic);
    state.add_global_memory_writes(traffic);
}

template<typename BT>
nvcv::Tensor MakeChannelReorderTensor(const int3 &shape, int channels, bool planar)
{
    if (planar)
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

template<typename BT, int NumChannels>
void RunFakePlanarChannelReorder(nvbench::state &state, cvcuda::ChannelReorder &op, const int3 &shape,
                                 const std::array<int32_t, NumChannels> &order)
{
    nvcv::Tensor src      = MakeChannelReorderTensor<BT>(shape, NumChannels, true);
    nvcv::Tensor interSrc = MakeChannelReorderTensor<BT>(shape, NumChannels, false);
    nvcv::Tensor interDst = MakeChannelReorderTensor<BT>(shape, NumChannels, false);
    nvcv::Tensor dst      = MakeChannelReorderTensor<BT>(shape, NumChannels, true);
    benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

    cvcuda::Reformat reformat;
    benchutils::warmup_and_exec(state, BENCH_CHANNELREORDER_WARMUP_ITERATIONS,
                                [&op, &reformat, &src, &interSrc, &interDst, &dst, &order](cudaStream_t stream)
                                {
                                    reformat(stream, src, interSrc);
                                    op(stream, interSrc, interDst, order.data(), NumChannels);
                                    reformat(stream, interDst, dst);
                                });
}

template<typename BT, int NumChannels>
void RunNativeTensorChannelReorder(nvbench::state &state, cvcuda::ChannelReorder &op, const int3 &shape, bool planar,
                                   const std::array<int32_t, NumChannels> &order)
{
    nvcv::Tensor src = MakeChannelReorderTensor<BT>(shape, NumChannels, planar);
    nvcv::Tensor dst = MakeChannelReorderTensor<BT>(shape, NumChannels, planar);
    benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());
    benchutils::warmup_and_exec(state, BENCH_CHANNELREORDER_WARMUP_ITERATIONS,
                                [&op, &src, &dst, &order](cudaStream_t stream)
                                { op(stream, src, dst, order.data(), NumChannels); });
}

template<typename T>
void RunTensorChannelReorder(nvbench::state &state, cvcuda::ChannelReorder &op, const int3 &shape,
                             ChannelReorderLayout layout, ChannelOrderPattern pattern)
{
    using BT                  = typename nvcv::cuda::BaseType<T>;
    constexpr int NumChannels = nvcv::cuda::NumElements<T>;
    const auto    order       = MakeChannelOrder<NumChannels>(pattern);

    if (IsFakePlanar(layout))
    {
        RunFakePlanarChannelReorder<BT, NumChannels>(state, op, shape, order);
        return;
    }
    RunNativeTensorChannelReorder<BT, NumChannels>(state, op, shape, IsPlanar(layout), order);
}

template<typename T>
void RunVarShapeChannelReorder(nvbench::state &state, cvcuda::ChannelReorder &op, const int3 &shape, bool planar,
                               ChannelOrderPattern pattern)
{
    constexpr int NumChannels = nvcv::cuda::NumElements<T>;
    nvcv::Tensor  orders(
         {
             {shape.x, 4},
             "NC"
    },
         nvcv::TYPE_S32);
    if (pattern == ChannelOrderPattern::Rotate)
    {
        benchutils::FillTensor<int>(orders,
                                    [](const long4_16a &coord) { return (int)((coord.x + coord.y) % NumChannels); });
    }
    else
    {
        benchutils::FillTensor<int>(
            orders, [](const long4_16a &coord) { return coord.y == 1 ? -1 : (int)(coord.y % NumChannels); });
    }

    nvcv::ImageBatchVarShape src(shape.x);
    nvcv::ImageBatchVarShape dst(shape.x);
    if (planar)
    {
        benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
        benchutils::FillPlanarImageBatchLike<T>(dst, src);
    }
    else
    {
        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0}, benchutils::CheckerboardValues<T>());
        benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
    }

    benchutils::warmup_and_exec(state, BENCH_CHANNELREORDER_WARMUP_ITERATIONS,
                                [&op, &src, &dst, &orders](cudaStream_t stream) { op(stream, src, dst, orders); });
}

} // namespace

template<typename T>
inline void channelreorder(nvbench::state &state, nvbench::type_list<T>)
try
{
    const int3                  shape        = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind    = benchutils::GetInputKind(state.get_string("inputKind"));
    const auto                  layoutString = benchutils::GetStringParam(state, "layout", "NHWC");
    const auto                  layout       = GetChannelReorderLayout(state, layoutString, inputKind);
    if (layout == ChannelReorderLayout::Invalid)
    {
        return;
    }
    const auto pattern = GetChannelOrderPattern(state.get_string("orderPattern"));

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    AddChannelReorderMemoryTraffic(state, bytes, layout);

    cvcuda::ChannelReorder op;
    if (inputKind == benchutils::InputKind::Tensor)
    {
        RunTensorChannelReorder<T>(state, op, shape, layout, pattern);
        return;
    }
    RunVarShapeChannelReorder<T>(state, op, shape, IsPlanar(layout), pattern);
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(channelreorder, NVBENCH_TYPE_AXES(BENCH_CHANNELREORDER_TYPES))
BENCH_CHANNELREORDER_AXES;
