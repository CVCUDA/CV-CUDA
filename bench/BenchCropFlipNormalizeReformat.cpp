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

#include <cvcuda/OpCropFlipNormalizeReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void CropFlipNormalizeReformat(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    long3 dstShape = srcShape;

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    float borderValue{0.f};

    float    globalScale = 1.234f;
    float    globalShift = 2.345f;
    float    epsilon     = 12.34f;
    uint32_t flags       = 0;

    long3 baseShape{srcShape.x, 1, 1};
    long3 scaleShape{srcShape.x, 1, 1};
    long3 cropShape{srcShape.x, 1, 1};

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T)
                                  + baseShape.x * baseShape.y * baseShape.z * sizeof(float)
                                  + scaleShape.x * scaleShape.y * scaleShape.z * sizeof(float)
                                  + cropShape.x * cropShape.y * cropShape.z * sizeof(int) * 4);
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::CropFlipNormalizeReformat op;

    // clang-format off

    nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

    nvcv::Tensor flipCode({{srcShape.x}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor base({{baseShape.x, baseShape.y, baseShape.z, 1}, "NHWC"}, nvcv::TYPE_F32);
    nvcv::Tensor scale({{scaleShape.x, scaleShape.y, scaleShape.z, 1}, "NHWC"}, nvcv::TYPE_F32);

    nvcv::Tensor crop({{cropShape.x, cropShape.y, cropShape.z, 4}, "NHWC"}, nvcv::TYPE_S32);

    benchutils::FillTensor<int>(flipCode, [](const long4 &){ return -1; });

    benchutils::FillTensor<float>(base, benchutils::RandomValues<T>());
    benchutils::FillTensor<float>(scale, benchutils::RandomValues<float>(0.f, 1.f));

    // Always crop entire source image for easy bandwidth calculations
    benchutils::FillTensor<int>(crop, [&srcShape](const long4 &c)
    {
        if (c.w == 2)
        {
            return (int)srcShape.z;
        }
        else if (c.w == 3)
        {
            return (int)srcShape.y;
        }
        return 0;
    });

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
                   [&op, &src, &dst, &crop, &borderType, &borderValue, &flipCode, &base, &scale, &globalScale,
                    &globalShift, &epsilon, &flags](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, crop, borderType, borderValue, flipCode, base, scale, globalScale,
               globalShift, epsilon, flags);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using CropFlipNormalizeReformatTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(CropFlipNormalizeReformat, NVBENCH_TYPE_AXES(CropFlipNormalizeReformatTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {0})
    .add_string_axis("border", {"REFLECT101"});
