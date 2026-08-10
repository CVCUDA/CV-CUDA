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
#include "ops/generated/BenchGaussianConfig.hpp"

#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <string_view>

namespace {

enum class GaussianLayout
{
    NHWC,
    NCHW,
    NCHWFake
};

GaussianLayout GetGaussianLayout(std::string_view layout, benchutils::InputKind inputKind)
{
    if (layout == "NHWC")
    {
        return GaussianLayout::NHWC;
    }
    if (layout == "NCHW")
    {
        return GaussianLayout::NCHW;
    }
    if (layout == "NCHW_FAKE")
    {
        if (inputKind == benchutils::InputKind::VarShape)
        {
            throw std::invalid_argument("Fake-planar (NCHW_FAKE) Gaussian benchmark is tensor-only");
        }
        return GaussianLayout::NCHWFake;
    }

    throw std::invalid_argument("Gaussian benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
}

bool IsPlanar(GaussianLayout layout)
{
    return layout == GaussianLayout::NCHW;
}

bool IsFakePlanar(GaussianLayout layout)
{
    return layout == GaussianLayout::NCHWFake;
}

void AddGaussianMemoryTraffic(nvbench::state &state, long bytes, GaussianLayout layout)
{
    if (IsFakePlanar(layout))
    {
        state.add_global_memory_reads(3 * bytes);
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes);
        state.add_global_memory_writes(bytes);
    }
}

template<typename T>
void ValidatePlanarChannels(GaussianLayout layout)
{
    if (!IsPlanar(layout))
    {
        return;
    }

    constexpr int ch = nvcv::cuda::NumElements<T>;
    if constexpr (ch == 2)
    {
        throw std::invalid_argument("Gaussian planar benchmark does not support 2-channel formats");
    }
    if constexpr (ch == 1)
    {
        throw std::invalid_argument("Gaussian planar benchmark requires multi-channel planar formats");
    }
}

template<typename T>
nvcv::Tensor MakeGaussianTensor(int3 shape, GaussianLayout layout)
{
    using BT         = typename nvcv::cuda::BaseType<T>;
    constexpr int ch = nvcv::cuda::NumElements<T>;

    if (IsPlanar(layout))
    {
        return nvcv::Tensor(
            nvcv::TensorShape{
                {shape.x, ch, shape.y, shape.z},
                "NCHW"
        },
            benchutils::GetDataType<BT>());
    }

    return nvcv::Tensor(
        nvcv::TensorShape{
            {shape.x, shape.y, shape.z, ch},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
}

template<typename T>
void FillGaussianImageBatches(nvcv::ImageBatchVarShape &src, nvcv::ImageBatchVarShape &dst, int3 shape,
                              GaussianLayout layout)
{
    if (IsPlanar(layout))
    {
        benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
        benchutils::FillPlanarImageBatchLike<T>(dst, src);
    }
    else
    {
        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0}, benchutils::CheckerboardValues<T>());
        benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
    }
}

} // namespace

template<typename T>
inline void gaussian(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layoutStr = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    double                      sigma     = state.get_float64("sigma");

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    GaussianLayout layout = GetGaussianLayout(layoutStr, inputKind);

    int  kernelSize = (int)std::round(sigma * (std::is_same_v<nvcv::cuda::BaseType<T>, uint8_t> ? 3 : 4) * 2 + 1) | 1;
    int2 ksize2{kernelSize, kernelSize};

    nvcv::Size2D kernelSize2{kernelSize, kernelSize};
    double2      sigma2{sigma, sigma};

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    AddGaussianMemoryTraffic(state, bytes, layout);

    cvcuda::Gaussian op(kernelSize2, shape.x);

    // clang-format off

    if (IsFakePlanar(layout)) // tensor-only: planar->interleaved->Gaussian->interleaved->planar
    {
        using BT = typename nvcv::cuda::BaseType<T>;
        int  ch  = nvcv::cuda::NumElements<T>;

        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_GAUSSIAN_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &kernelSize2, &sigma2, &borderType](cudaStream_t s) {
                reformatOp(s, src, interSrc);                                      // NCHW -> NHWC
                op(s, interSrc, interDst, kernelSize2, sigma2, borderType);         // interleaved Gaussian
                reformatOp(s, interDst, dst);                                      // NHWC -> NCHW
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        using BT = typename nvcv::cuda::BaseType<T>;

        ValidatePlanarChannels<T>(layout);
        nvcv::Tensor src = MakeGaussianTensor<T>(shape, layout);
        nvcv::Tensor dst = MakeGaussianTensor<T>(shape, layout);

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_GAUSSIAN_WARMUP_ITERATIONS,
            [&op, &src, &dst, &kernelSize2, &sigma2, &borderType](cudaStream_t s) {
                op(s, src, dst, kernelSize2, sigma2, borderType);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        ValidatePlanarChannels<T>(layout);

        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);
        FillGaussianImageBatches<T>(src, dst, shape, layout);

        nvcv::Tensor kernelSizeTensor({{shape.x}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor sigmaTensor({{shape.x}, "N"}, nvcv::TYPE_2F64);

        benchutils::FillTensor<int2>(kernelSizeTensor, [&ksize2](const long4_16a &){ return ksize2; });
        benchutils::FillTensor<double2>(sigmaTensor, [&sigma2](const long4_16a &){ return sigma2; });

        benchutils::warmup_and_exec(state, BENCH_GAUSSIAN_WARMUP_ITERATIONS,
            [&op, &src, &dst, &kernelSizeTensor, &sigmaTensor, &borderType](cudaStream_t s) {
                op(s, src, dst, kernelSizeTensor, sigmaTensor, borderType);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(gaussian, NVBENCH_TYPE_AXES(BENCH_GAUSSIAN_TYPES))
BENCH_GAUSSIAN_AXES;
