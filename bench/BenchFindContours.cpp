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

#include <cvcuda/OpFindContours.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <stdlib.h>

#include <nvbench/nvbench.cuh>

using CPUImage = std::vector<uint8_t>;

static void generateRectangle(CPUImage &image, nvcv::Size2D boundary, nvcv::Size2D anchor = {0, 0},
                              nvcv::Size2D size = {5, 5}, double angle = 0.0, bool fill = true, uint8_t setValue = 1);

static void generateRectangle(CPUImage &image, nvcv::Size2D boundary, nvcv::Size2D anchor, nvcv::Size2D size,
                              double angle, bool fill, uint8_t setValue)
{
    auto rad      = angle * (M_PI / 180.0);
    auto cosAngle = std::cos(rad);
    auto sinAngle = std::sin(rad);

    auto transformed = anchor;
    for (auto y = 0; y < size.h; ++y)
    {
        for (auto x = 0; x < size.w; ++x)
        {
            transformed.w = anchor.w + (x * cosAngle - y * sinAngle);
            transformed.h = anchor.h + (x * sinAngle + y * cosAngle);

            if (fill || y == 0 || y == size.h - 1 || x == 0 || x == size.w - 1)
            {
                if (transformed.w >= 0 && transformed.w < boundary.w && transformed.h >= 0
                    && transformed.h < boundary.h)
                {
                    image[transformed.h * boundary.w + transformed.w] = setValue;
                }
            }
        }
    }
}

template<typename T>
inline void FindContours(nvbench::state &state, nvbench::type_list<T>)
try
{
    srand(0U); // Use a fixed random seed
    long3 shape     = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape  = state.get_int64("varShape");
    int   numPoints = static_cast<int>(state.get_int64("numPoints"));

    // R/W bandwidth rationale:
    // Read image + connected components (S32)
    // Write points + contours (U32)
    state.add_global_memory_reads(shape.x * shape.y * shape.z * (sizeof(T) + sizeof(int)));
    state.add_global_memory_writes(shape.x * numPoints * sizeof(int) * 2 + shape.x * 4 * sizeof(int));

    cvcuda::FindContours op(nvcv::Size2D{(int)shape.z, (int)shape.y}, shape.x);

    // clang-format off

    nvcv::Tensor points({{shape.x, numPoints, 2}, "NCW"}, nvcv::TYPE_S32);
    nvcv::Tensor counts({{shape.x, 4}, "NW"}, nvcv::TYPE_S32);

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        auto inData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);

        //Generate input
        CPUImage srcVec(shape.y * shape.z, 0);
        for (auto i = 0; i < 10; ++i)
        {
            auto anchorX = rand() % shape.z;
            auto anchorY = rand() % shape.y;
            auto sizeX = rand() % (shape.z - anchorX);
            auto sizeY = rand() % (shape.y - anchorY);
            generateRectangle(srcVec, {anchorX, anchorY}, {sizeX, sizeY});
        }

        for (auto i = 0; i < shape.x; ++i)
        {
            CUDA_CHECK_ERROR(cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec.data(), shape.z, shape.z,
                                          shape.y, cudaMemcpyHostToDevice));
        }

        state.exec(nvbench::exec_tag::sync, [&op, &src, &points, &counts](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, points, counts);
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

using FindContoursTypes = nvbench::type_list<uint8_t>;

NVBENCH_BENCH_TYPES(FindContours, NVBENCH_TYPE_AXES(FindContoursTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_int64_axis("numPoints", {1024});
