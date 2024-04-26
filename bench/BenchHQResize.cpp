/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpHQResize.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void HQResize(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                 srcShape      = benchutils::GetShape<3>(state.get_string("shape"));
    bool                  antialias     = state.get_int64("antialias");
    NVCVInterpolationType interpolation = benchutils::GetInterpolationType(state.get_string("interpolation"));
    bool                  batch         = state.get_int64("batch");

    long3 dstShape;
    if (state.get_string("resizeType") == "EXPAND")
    {
        if (antialias)
        {
            state.skip("Antialias is no-op for expanding");
            return;
        }
        dstShape = long3{srcShape.x, srcShape.y * 2, srcShape.z * 2};
    }
    else if (state.get_string("resizeType") == "CONTRACT")
    {
        // resize from shape to shape/2
        dstShape = long3{srcShape.x, srcShape.y / 2, srcShape.z / 2};
    }
    else
    {
        throw std::invalid_argument("Invalid resizeType = " + state.get_string("resizeType"));
    }

    nvcv::Size2D srcSize{(int)srcShape.z, (int)srcShape.y};
    nvcv::Size2D dstSize{(int)dstShape.z, (int)dstShape.y};

    nvcv::DataType    dtype{benchutils::GetDataType<T>()};
    nvcv::ImageFormat fmt(nvcv::MemLayout::PITCH_LINEAR, dtype.dataKind(), nvcv::Swizzle::S_X000, dtype.packing());

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T));
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::HQResize op;

    if (!batch)
    {
        HQResizeTensorShapeI inShapeDesc{
            {srcSize.h, srcSize.w},
            2,
            1
        };
        HQResizeTensorShapeI outShapeDesc{
            {dstSize.h, dstSize.w},
            2,
            1
        };
        cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
            op.getWorkspaceRequirements(1, inShapeDesc, outShapeDesc, interpolation, interpolation, antialias));

        // clang-format off
        nvcv::Tensor src({{srcShape.x, srcShape.y, srcShape.z, 1}, "NHWC"}, dtype);
        nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"}, dtype);
        // clang-format on

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &ws, &src, &dst, interpolation, antialias](nvbench::launch &launch)
                   { op(launch.get_stream(), ws.get(), src, dst, interpolation, interpolation, antialias); });
    }
    else
    {
        HQResizeTensorShapeI maxShape{
            {std::max(srcSize.h, dstSize.h), std::max(srcSize.w, dstSize.w)},
            2,
            1
        };
        cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, maxShape));

        // clang-format off
        nvcv::Tensor src({{srcShape.y, srcShape.z, 1}, "HWC"}, dtype);
        nvcv::Tensor dst({{dstShape.y, dstShape.z, 1}, "HWC"}, dtype);
        // clang-format on

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());
        nvcv::TensorBatch srcTensors(1);
        nvcv::TensorBatch dstTensors(1);
        srcTensors.pushBack(src);
        dstTensors.pushBack(dst);

        state.exec(
            nvbench::exec_tag::sync,
            [&op, &ws, &srcTensors, &dstTensors, interpolation, antialias](nvbench::launch &launch)
            { op(launch.get_stream(), ws.get(), srcTensors, dstTensors, interpolation, interpolation, antialias); });
    }
}

catch (const std::exception &err)
{
    state.skip(err.what());
}

using HQResizeTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(HQResize, NVBENCH_TYPE_AXES(HQResizeTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_int64_axis("batch", {false, true})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_string_axis("interpolation", {"CUBIC"})
    .add_int64_axis("antialias", {false, true})
    .add_string_axis("resizeType", {"CONTRACT"});
