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
#include "ops/generated/BenchLabelConfig.hpp"

#include <cvcuda/OpLabel.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <string>
#include <string_view>
#include <utility>

inline bool HasRunChoice(std::string_view runChoice, std::string_view choice)
{
    return runChoice.find(choice) != std::string::npos;
}

struct LabelLayout
{
    std::string runChoice;
    bool        isPlanar        = false;
    bool        isFakePlanar    = false;
    bool        usePlanarLayout = false;
    bool        valid           = true;
};

template<typename ST, typename DT>
struct LabelTensors
{
    nvcv::Tensor src;
    nvcv::Tensor dst;
};

inline LabelLayout ParseLabelLayout(nvbench::state &state, std::string runChoice)
{
    if (runChoice == "DEFAULT")
    {
        runChoice = "";
    }

    const std::string layout = benchutils::GetStringParam(state, "layout", "NHWC");
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("Label benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return {"", false, false, false, false};
    }

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";

    return {std::move(runChoice), isPlanar, isFakePlanar, isPlanar || isFakePlanar, true};
}

template<typename ST, typename DT>
void SetupOptionalLabelInputs(const std::string &runChoice, long3 srcShape, long3 staShape, bool usePlanarLayout,
                              nvcv::Tensor &bgT, nvcv::Tensor &minT, nvcv::Tensor &maxT, nvcv::Tensor &mszT,
                              nvcv::Tensor &countT, nvcv::Tensor &statsT, nvcv::Tensor &maskT)
{
    if (HasRunChoice(runChoice, "BG"))
    {
        bgT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<ST>());

        benchutils::FillTensor<ST>(bgT, benchutils::LcgValues<ST>());
    }
    if (HasRunChoice(runChoice, "MIN"))
    {
        minT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<ST>());

        const ST minThreshold = std::is_signed_v<ST> ? static_cast<ST>(-64) : static_cast<ST>(64);
        benchutils::FillTensor<ST>(minT, benchutils::LcgValues<ST>(minThreshold, minThreshold));
    }
    if (HasRunChoice(runChoice, "MAX"))
    {
        maxT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<ST>());

        const ST maxThreshold = std::is_signed_v<ST> ? static_cast<ST>(64) : static_cast<ST>(192);
        benchutils::FillTensor<ST>(maxT, benchutils::LcgValues<ST>(maxThreshold, maxThreshold));
    }
    if (HasRunChoice(runChoice, "ISLAND"))
    {
        mszT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<DT>());

        benchutils::FillTensor<DT>(mszT, benchutils::LcgValues<DT>(16, 16));
    }
    if (HasRunChoice(runChoice, "COUNT"))
    {
        countT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<DT>());
    }
    if (HasRunChoice(runChoice, "STAT"))
    {
        statsT = nvcv::Tensor(
            {
                {staShape.x, staShape.y, staShape.z},
                "NMA"
        },
            benchutils::GetDataType<DT>());
    }
    if (HasRunChoice(runChoice, "MASK"))
    {
        maskT = usePlanarLayout ? nvcv::Tensor(
                    {
                        {srcShape.x, 1, srcShape.y, srcShape.z},
                        "NCHW"
        },
                    nvcv::TYPE_U8)
                                : nvcv::Tensor({{srcShape.x, srcShape.y, srcShape.z, 1}, "NHWC"}, nvcv::TYPE_U8);

        benchutils::FillTensor<uint8_t>(maskT, benchutils::CheckerboardValues<uint8_t>());
    }
}

template<typename ST, typename DT>
LabelTensors<ST, DT> BuildLabelTensors(long3 srcShape, long3 dstShape, bool usePlanarLayout)
{
    if (usePlanarLayout)
    {
        return {
            nvcv::Tensor({{srcShape.x, 1, srcShape.y, srcShape.z}, "NCHW"},
            benchutils::GetDataType<ST>()),
            nvcv::Tensor({{dstShape.x, 1, dstShape.y, dstShape.z}, "NCHW"},
            benchutils::GetDataType<DT>())
        };
    }

    return {
        nvcv::Tensor({{srcShape.x, srcShape.y, srcShape.z, 1}, "NHWC"},
        benchutils::GetDataType<ST>()),
        nvcv::Tensor({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"},
        benchutils::GetDataType<DT>())
    };
}

template<typename ST, typename DT>
void RunLabelFakePlanar(nvbench::state &state, cvcuda::Label &op, cvcuda::Reformat &reformatOp, nvcv::Tensor &src,
                        nvcv::Tensor &dst, long3 srcShape, long3 dstShape, nvcv::Tensor &bgT, nvcv::Tensor &minT,
                        nvcv::Tensor &maxT, nvcv::Tensor &mszT, nvcv::Tensor &countT, nvcv::Tensor &statsT,
                        nvcv::Tensor &maskT, NVCVConnectivityType conn, NVCVLabelType alab, NVCVLabelMaskType mType)
{
    nvcv::Tensor interSrc(
        {
            {srcShape.x, srcShape.y, srcShape.z, 1},
            "NHWC"
    },
        benchutils::GetDataType<ST>());
    nvcv::Tensor interDst(
        {
            {dstShape.x, dstShape.y, dstShape.z, 1},
            "NHWC"
    },
        benchutils::GetDataType<DT>());
    nvcv::Tensor interMaskT;
    if (maskT)
    {
        interMaskT = nvcv::Tensor(
            {
                {srcShape.x, srcShape.y, srcShape.z, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);
    }

    benchutils::warmup_and_exec(state, BENCH_LABEL_WARMUP_ITERATIONS,
                                [&op, &reformatOp, &src, &dst, &interSrc, &interDst, &bgT, &minT, &maxT, &mszT, &countT,
                                 &statsT, &maskT, &interMaskT, &conn, &alab, &mType](cudaStream_t s)
                                {
                                    reformatOp(s, src, interSrc);
                                    if (maskT)
                                    {
                                        reformatOp(s, maskT, interMaskT);
                                    }
                                    op(s, interSrc, interDst, bgT, minT, maxT, mszT, countT, statsT, interMaskT, conn,
                                       alab, mType);
                                    reformatOp(s, interDst, dst);
                                });
}

template<typename ST>
inline void label(nvbench::state &state, nvbench::type_list<ST>)
try
{
    // Use int (S32) for label output dtype to match Python benchmark
    using DT = int;

    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long3 dstShape = srcShape;

    LabelLayout layout = ParseLabelLayout(state, state.get_string("runChoice"));
    if (!layout.valid)
    {
        return;
    }

    // Use [BG][MIN][MAX][ISLAND][COUNT][STAT][MASK] in runChoice to run Label with:
    // background; minThreshold; maxThreshold; island removal; count; statistics; mask

    long3 staShape{srcShape.x, 10000, 7}; // using fixed 10K max. cap. and 2D problem

    NVCVConnectivityType conn  = NVCV_CONNECTIVITY_4_2D;
    NVCVLabelType        alab  = NVCV_LABEL_FAST;
    NVCVLabelMaskType    mType = NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY;

    nvcv::Tensor bgT;
    nvcv::Tensor minT;
    nvcv::Tensor maxT;
    nvcv::Tensor countT;
    nvcv::Tensor statsT;
    nvcv::Tensor mszT;
    nvcv::Tensor maskT;

    cvcuda::Label op;

    const long imageBytes = srcShape.x * srcShape.y * srcShape.z * sizeof(ST);
    const long outBytes   = dstShape.x * dstShape.y * dstShape.z * sizeof(DT);
    if (layout.isFakePlanar)
    {
        state.add_global_memory_reads(2 * imageBytes + outBytes);
        state.add_global_memory_writes(imageBytes + 2 * outBytes);
    }
    else
    {
        state.add_global_memory_reads(imageBytes);
        state.add_global_memory_writes(outBytes);
    }

    SetupOptionalLabelInputs<ST, DT>(layout.runChoice, srcShape, staShape, layout.usePlanarLayout, bgT, minT, maxT,
                                     mszT, countT, statsT, maskT);

    LabelTensors<ST, DT> tensors = BuildLabelTensors<ST, DT>(srcShape, dstShape, layout.usePlanarLayout);

    benchutils::FillTensor<ST>(tensors.src, benchutils::LcgValues<ST>());

    if (layout.isFakePlanar)
    {
        cvcuda::Reformat reformatOp;
        RunLabelFakePlanar<ST, DT>(state, op, reformatOp, tensors.src, tensors.dst, srcShape, dstShape, bgT, minT, maxT,
                                   mszT, countT, statsT, maskT, conn, alab, mType);
    }
    else
    {
        benchutils::warmup_and_exec(
            state, BENCH_LABEL_WARMUP_ITERATIONS,
            [&op, &src = tensors.src, &dst = tensors.dst, &bgT, &minT, &maxT, &mszT, &countT, &statsT, &maskT, &conn,
             &alab, &mType](cudaStream_t s)
            { op(s, src, dst, bgT, minT, maxT, mszT, countT, statsT, maskT, conn, alab, mType); });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(label, NVBENCH_TYPE_AXES(BENCH_LABEL_TYPES))
BENCH_LABEL_AXES;
