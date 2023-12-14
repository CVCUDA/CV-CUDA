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

#include <../priv/Types.hpp>
#include <cvcuda/OpOSD.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void OSD(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    int   numElem  = static_cast<int>(state.get_int64("numElem"));

    int ch = nvcv::cuda::NumElements<T>;

    using BT = nvcv::cuda::BaseType<T>;

    std::vector<std::vector<std::shared_ptr<cvcuda::priv::NVCVElement>>> elementVec;

    for (int n = 0; n < (int)shape.x; n++)
    {
        std::vector<std::shared_ptr<cvcuda::priv::NVCVElement>> curVec;
        for (int i = 0; i < numElem; i++)
        {
            NVCVPoint point;
            point.centerPos.x = shape.z / 2;
            point.centerPos.y = shape.y / 2;
            point.radius      = std::min(shape.z, shape.y) / 2;
            point.color       = {0, 0, 0, 255};
            auto element      = std::make_shared<cvcuda::priv::NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point);
            curVec.push_back(element);
        }
        elementVec.push_back(curVec);
    }

    std::shared_ptr<cvcuda::priv::NVCVElementsImpl> ctx = std::make_shared<cvcuda::priv::NVCVElementsImpl>(elementVec);

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) + numElem * sizeof(int) * 16);
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::OSD op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::RandomValues<BT>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &ctx](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, (NVCVElements)ctx.get());
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using OSDTypes = nvbench::type_list<uchar3, uchar4>;

NVBENCH_BENCH_TYPES(OSD, NVBENCH_TYPE_AXES(OSDTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_int64_axis("numElem", {100});
