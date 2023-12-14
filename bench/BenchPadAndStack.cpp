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

#include <cvcuda/OpPadAndStack.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void PadAndStack(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    long3 dstShape = srcShape;

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    float borderValue{0.f};

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T) + srcShape.x * sizeof(int) * 2);
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::PadAndStack op;

    // clang-format off

    nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

    nvcv::Tensor top({{srcShape.x, 1, 1, 1}, "NHWC"}, nvcv::TYPE_S32);
    nvcv::Tensor left({{srcShape.x, 1, 1, 1}, "NHWC"}, nvcv::TYPE_S32);

    benchutils::FillTensor<int>(top, [&srcShape](const long4 &){ return srcShape.y / 2; });
    benchutils::FillTensor<int>(left, [&srcShape](const long4 &){ return srcShape.z / 2; });

    if (varShape < 0) // negative var shape means use Tensor
    {
        throw std::invalid_argument("Tensor not implemented for this operator");
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(srcShape.x);

        benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &top, &left, &borderType, &borderValue](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, top, left, borderType, borderValue);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using PadAndStackTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(PadAndStack, NVBENCH_TYPE_AXES(PadAndStackTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {0})
    .add_string_axis("border", {"REFLECT101"});
