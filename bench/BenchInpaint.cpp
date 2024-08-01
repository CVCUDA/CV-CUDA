/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpInpaint.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Inpaint(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    double inpaintRadius = 5.0;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * (sizeof(T) + sizeof(uint8_t)));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Inpaint op(shape.x, nvcv::Size2D{(int)shape.z, (int)shape.y});

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor mask({{shape.x, shape.y, shape.z, 1}, "NHWC"}, nvcv::TYPE_U8);

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());
        benchutils::FillTensor<uint8_t>(mask, benchutils::RandomValues<uint8_t>(0, 1));

        state.exec(nvbench::exec_tag::sync, [&op, &src, &mask, &dst, &inpaintRadius](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, mask, dst, inpaintRadius);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);
        nvcv::ImageBatchVarShape mask(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        benchutils::FillImageBatch<uint8_t>(mask, long2{shape.z, shape.y}, long2{varShape, varShape},
                                            benchutils::RandomValues<uint8_t>(0, 1));

        state.exec(nvbench::exec_tag::sync, [&op, &src, &mask, &dst, &inpaintRadius](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, mask, dst, inpaintRadius);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using InpaintTypes = nvbench::type_list<uint8_t, uchar4>;

NVBENCH_BENCH_TYPES(Inpaint, NVBENCH_TYPE_AXES(InpaintTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0});
