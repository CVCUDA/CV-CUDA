/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "BenchUtils.hpp"

#include <cvcuda/OpLabel.hpp>

#include <nvbench/nvbench.cuh>

template<typename ST>
inline void Label(nvbench::state &state, nvbench::type_list<ST>)
try
{
    using DT = uint32_t;

    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long3 dstShape = srcShape;

    std::string runChoice = state.get_string("runChoice");

    // Use [BG][MIN][MAX][ISLAND][COUNT][STAT] in runChoice to run Label with:
    // background; minThreshold; maxThreshold; island removal; count; statistics

    long3 staShape{srcShape.x, 10000, 6}; // using fixed 10K max. cap. and 2D problem

    NVCVConnectivityType conn = NVCV_CONNECTIVITY_4_2D;
    NVCVLabelType        alab = NVCV_LABEL_FAST;

    nvcv::Tensor bgT, minT, maxT, countT, statsT, mszT;

    cvcuda::Label op;

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(ST));
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(DT));

    // clang-format off

    if (runChoice.find("BG") != std::string::npos)
    {
        bgT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<ST>());

        benchutils::FillTensor<ST>(bgT, benchutils::RandomValues<ST>());
    }
    if (runChoice.find("MIN") != std::string::npos)
    {
        minT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<ST>());

        benchutils::FillTensor<ST>(minT, benchutils::RandomValues<ST>());
    }
    if (runChoice.find("MAX") != std::string::npos)
    {
        maxT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<ST>());

        benchutils::FillTensor<ST>(maxT, benchutils::RandomValues<ST>());
    }
    if (runChoice.find("ISLAND") != std::string::npos)
    {
        mszT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<DT>());

        benchutils::FillTensor<DT>(mszT, benchutils::RandomValues<DT>());
    }
    if (runChoice.find("COUNT") != std::string::npos)
    {
        countT = nvcv::Tensor({{srcShape.x}, "N"}, benchutils::GetDataType<DT>());
    }
    if (runChoice.find("STAT") != std::string::npos)
    {
        statsT = nvcv::Tensor({{staShape.x, staShape.y, staShape.z}, "NMA"}, benchutils::GetDataType<DT>());
    }

    nvcv::Tensor src({{srcShape.x, srcShape.y, srcShape.z, 1}, "NHWC"}, benchutils::GetDataType<ST>());
    nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"}, benchutils::GetDataType<DT>());

    benchutils::FillTensor<ST>(src, benchutils::RandomValues<ST>());

    state.exec(nvbench::exec_tag::sync,
               [&op, &src, &dst, &bgT, &minT, &maxT, &mszT, &countT, &statsT, &conn, &alab](nvbench::launch &launch)
               {
                   op(launch.get_stream(), src, dst, bgT, minT, maxT, mszT, countT, statsT, conn, alab);
               });
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using LabelTypes = nvbench::type_list<uint8_t, uint32_t>;

NVBENCH_BENCH_TYPES(Label, NVBENCH_TYPE_AXES(LabelTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_string_axis("runChoice", {""});
