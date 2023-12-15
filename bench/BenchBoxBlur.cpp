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
#include <cvcuda/OpBoxBlur.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void BoxBlur(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape      = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape   = state.get_int64("varShape");
    int   numBoxes   = static_cast<int>(state.get_int64("numBoxes"));
    int   kernelSize = static_cast<int>(state.get_int64("kernelSize"));

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    NVCVBlurBoxI blurBox{
        {43, 21, 12, 34}, // box x, y position w, h size
        kernelSize  // median filter kernel size
    };

    std::vector<std::vector<NVCVBlurBoxI>> blurBoxesVec;

    for (int i = 0; i < shape.x; i++)
    {
        std::vector<NVCVBlurBoxI> curVec;
        for (int j = 0; j < numBoxes; j++)
        {
            curVec.push_back(blurBox);
        }
        blurBoxesVec.push_back(curVec);
    }

    std::shared_ptr<cvcuda::priv::NVCVBlurBoxesImpl> blurBoxesImpl
        = std::make_shared<cvcuda::priv::NVCVBlurBoxesImpl>(blurBoxesVec);
    NVCVBlurBoxesI blurBoxes = (NVCVBlurBoxesI)blurBoxesImpl.get();

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) + shape.x * numBoxes * sizeof(NVCVBlurBoxI));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::BoxBlur op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::RandomValues<BT>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &blurBoxes](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, blurBoxes);
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

using BoxBlurTypes = nvbench::type_list<uchar3, uchar4>;

NVBENCH_BENCH_TYPES(BoxBlur, NVBENCH_TYPE_AXES(BoxBlurTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_int64_axis("numBoxes", {4})
    .add_int64_axis("kernelSize", {5});
