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
#include "ops/generated/BenchHistogramConfig.hpp"

#include <cvcuda/OpHistogram.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <string>
#include <string_view>

namespace {

bool IsHistogramLayoutSupported(std::string_view layout)
{
    return layout == "NHWC" || layout == "NCHW" || layout == "NCHW_FAKE";
}

bool IsPlanarBenchmarkLayout(std::string_view layout)
{
    return layout == "NCHW" || layout == "NCHW_FAKE";
}

bool UseHistogramMask(std::string_view maskMode)
{
    if (maskMode == "checkerboard")
    {
        return true;
    }
    if (maskMode == "none")
    {
        return false;
    }
    throw std::invalid_argument("Unsupported maskMode: " + std::string(maskMode));
}

struct HistogramBenchParams
{
    int3                  shape;
    benchutils::InputKind inputKind;
    bool                  useMask;
    std::string           layout;
    bool                  isPlanar;
    bool                  isFakePlanar;
};

inline HistogramBenchParams ParseHistogramParams(nvbench::state &state)
{
    std::string layout = benchutils::GetStringParam(state, "layout", "NHWC");
    return {
        benchutils::GetShape<3, int3>(state.get_string("shape")),
        benchutils::GetInputKind(state.get_string("inputKind")),
        UseHistogramMask(state.get_string("maskMode")),
        layout,
        layout == "NCHW",
        layout == "NCHW_FAKE",
    };
}

template<typename T>
void AddMemoryCounters(nvbench::state &state, int3 shape, bool useMask, bool hasPlanarAdapter)
{
    constexpr int numBins = 256;

    const long imageBytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const long maskBytes  = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(uint8_t);
    const long histBytes  = static_cast<long>(shape.x) * numBins * sizeof(int);

    if (hasPlanarAdapter)
    {
        state.add_global_memory_reads(2 * imageBytes + (useMask ? 2 * maskBytes : 0));
        state.add_global_memory_writes(imageBytes + (useMask ? maskBytes : 0) + histBytes);
    }
    else
    {
        state.add_global_memory_reads(imageBytes + (useMask ? maskBytes : 0));
        state.add_global_memory_writes(histBytes);
    }
}

template<typename T>
bool ValidateAndAccount(nvbench::state &state, const HistogramBenchParams &params)
{
    if (!IsHistogramLayoutSupported(params.layout))
    {
        state.skip("Histogram benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return false;
    }
    if (IsPlanarBenchmarkLayout(params.layout) && params.inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Planar Histogram benchmark is tensor-only");
        return false;
    }

    AddMemoryCounters<T>(state, params.shape, params.useMask, params.isPlanar || params.isFakePlanar);
    return true;
}

template<typename T>
void RunFakePlanarHistogram(nvbench::state &state, int3 shape, bool useMask, cvcuda::Histogram &op)
{
    constexpr int numBins = 256;

    nvcv::Tensor src(
        {
            {shape.x, 1, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<T>());
    nvcv::Tensor interSrc(
        {
            {shape.x, shape.y, shape.z, 1},
            "NHWC"
    },
        benchutils::GetDataType<T>());
    nvcv::Tensor hist(
        {
            {shape.x, numBins, 1},
            "HWC"
    },
        nvcv::TYPE_S32);

    benchutils::FillTensor<T>(src, benchutils::CheckerboardValues<T>());
    benchutils::FillTensor<int>(hist, [](auto &) { return 0; });

    cvcuda::Reformat reformatOp;

    if (useMask)
    {
        nvcv::Tensor mask(
            {
                {shape.x, 1, shape.y, shape.z},
                "NCHW"
        },
            nvcv::TYPE_U8);
        nvcv::Tensor interMask(
            {
                {shape.x, shape.y, shape.z, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);

        benchutils::FillTensor<uint8_t>(mask, benchutils::CheckerboardValues<uint8_t>());

        benchutils::warmup_and_exec(state, BENCH_HISTOGRAM_WARMUP_ITERATIONS,
                                    [&op, &reformatOp, &src, &interSrc, &mask, &interMask, &hist](cudaStream_t s)
                                    {
                                        reformatOp(s, src, interSrc);
                                        reformatOp(s, mask, interMask);
                                        op(s, interSrc, nvcv::OptionalTensorConstRef{interMask}, hist);
                                    });
        return;
    }

    benchutils::warmup_and_exec(state, BENCH_HISTOGRAM_WARMUP_ITERATIONS,
                                [&op, &reformatOp, &src, &interSrc, &hist](cudaStream_t s)
                                {
                                    reformatOp(s, src, interSrc);
                                    op(s, interSrc, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, hist);
                                });
}

template<typename T>
void RunTensorHistogram(nvbench::state &state, int3 shape, bool useMask, bool isPlanar, cvcuda::Histogram &op)
{
    constexpr int numBins = 256;

    nvcv::Tensor src = isPlanar ? nvcv::Tensor(
                           {
                               {shape.x, 1, shape.y, shape.z},
                               "NCHW"
    },
                           benchutils::GetDataType<T>())
                                : nvcv::Tensor({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
    nvcv::Tensor hist(
        {
            {shape.x, numBins, 1},
            "HWC"
    },
        nvcv::TYPE_S32);

    benchutils::FillTensor<T>(src, benchutils::CheckerboardValues<T>());
    benchutils::FillTensor<int>(hist, [](auto &) { return 0; });

    if (useMask)
    {
        nvcv::Tensor mask = isPlanar ? nvcv::Tensor(
                                {
                                    {shape.x, 1, shape.y, shape.z},
                                    "NCHW"
        },
                                nvcv::TYPE_U8)
                                     : nvcv::Tensor({{shape.x, shape.y, shape.z, 1}, "NHWC"}, nvcv::TYPE_U8);

        benchutils::FillTensor<uint8_t>(mask, benchutils::CheckerboardValues<uint8_t>());

        benchutils::warmup_and_exec(state, BENCH_HISTOGRAM_WARMUP_ITERATIONS,
                                    [&op, &src, &mask, &hist](cudaStream_t s)
                                    { op(s, src, nvcv::OptionalTensorConstRef{mask}, hist); });
        return;
    }

    benchutils::warmup_and_exec(state, BENCH_HISTOGRAM_WARMUP_ITERATIONS,
                                [&op, &src, &hist](cudaStream_t s)
                                { op(s, src, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, hist); });
}

template<typename T>
void DispatchHistogram(nvbench::state &state, const HistogramBenchParams &params, cvcuda::Histogram &op)
{
    if (params.isFakePlanar)
    {
        RunFakePlanarHistogram<T>(state, params.shape, params.useMask, op);
    }
    else if (params.inputKind == benchutils::InputKind::Tensor)
    {
        RunTensorHistogram<T>(state, params.shape, params.useMask, params.isPlanar, op);
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}

} // namespace

template<typename T>
inline void histogram(nvbench::state &state, nvbench::type_list<T>)
try
{
    const HistogramBenchParams params = ParseHistogramParams(state);
    if (!ValidateAndAccount<T>(state, params))
        return;

    cvcuda::Histogram op;
    DispatchHistogram<T>(state, params, op);
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(histogram, NVBENCH_TYPE_AXES(BENCH_HISTOGRAM_TYPES))
BENCH_HISTOGRAM_AXES;
