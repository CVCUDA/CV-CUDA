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

#include <cvcuda/OpRemap.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Remap(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    long3 dstShape = srcShape;
    long3 mapShape;

    NVCVInterpolationType srcInterp, mapInterp;
    NVCVBorderType        borderType;
    NVCVRemapMapValueType mapValueType;

    bool   alignCorners{true};
    float4 borderValue{0, 0, 0, 0};

    if (state.get_string("mapType") == "DENSE")
    {
        srcInterp    = NVCV_INTERP_NEAREST;
        mapInterp    = NVCV_INTERP_NEAREST;
        borderType   = NVCV_BORDER_CONSTANT;
        mapValueType = NVCV_REMAP_ABSOLUTE_NORMALIZED;
        mapShape     = srcShape;
    }
    else if (state.get_string("mapType") == "RELATIVE")
    {
        srcInterp    = NVCV_INTERP_CUBIC;
        mapInterp    = NVCV_INTERP_CUBIC;
        borderType   = NVCV_BORDER_REFLECT101;
        mapValueType = NVCV_REMAP_RELATIVE_NORMALIZED;
        mapShape     = long3{srcShape.x, 4, 4};
    }
    else
    {
        throw std::invalid_argument("Invalid mapType = " + state.get_string("mapType"));
    }

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T)
                                  + mapShape.x * mapShape.y * mapShape.z * sizeof(float2));
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::Remap op;

    // clang-format off

    nvcv::Tensor map({{mapShape.x, mapShape.y, mapShape.z, 1}, "NHWC"}, nvcv::TYPE_2F32);

    benchutils::FillTensor<float2>(map, benchutils::RandomValues<float2>());

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{srcShape.x, srcShape.y, srcShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &map, &srcInterp, &mapInterp, &mapValueType, &alignCorners, &borderType,
                    &borderValue](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderType,
               borderValue);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(srcShape.x);
        nvcv::ImageBatchVarShape dst(dstShape.x);

        benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        benchutils::FillImageBatch<T>(dst, long2{dstShape.z, dstShape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &map, &srcInterp, &mapInterp, &mapValueType, &alignCorners, &borderType,
                    &borderValue](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderType,
               borderValue);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using RemapTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Remap, NVBENCH_TYPE_AXES(RemapTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0})
    .add_string_axis("mapType", {"DENSE"});
