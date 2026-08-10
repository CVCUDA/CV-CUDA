/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchFindHomographyConfig.hpp"

#include <cvcuda/OpFindHomography.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
void fill_src_grid_replicated(std::vector<T> &vec, std::size_t numSamples, std::size_t numPoints)
{
    std::size_t gridSide = 1;
    while (gridSide * gridSide < numPoints)
    {
        ++gridSide;
    }

    const T           gridScale  = static_cast<T>(2) / static_cast<T>(gridSide - 1);
    const std::size_t sampleSize = numPoints * 2;
    for (std::size_t j = 0; j < numPoints; j++)
    {
        // A two-dimensional grid makes the four-point workload the corners of a square.
        T           x       = static_cast<T>(-1) + static_cast<T>(j % gridSide) * gridScale;
        T           y       = static_cast<T>(-1) + static_cast<T>(j / gridSide) * gridScale;
        std::size_t baseIdx = j * 2;

        for (std::size_t i = 0; i < numSamples; i++)
        {
            vec[i * sampleSize + baseIdx]     = x;
            vec[i * sampleSize + baseIdx + 1] = y;
        }
    }
}

template<typename T>
void fill_dst_projective_replicated(const std::vector<T> &srcVec, const std::vector<T> &transform,
                                    std::vector<T> &dstVec, std::size_t numSamples, std::size_t numPoints)
{
    const T    *model      = transform.data();
    std::size_t sampleSize = numPoints * 2;

    for (std::size_t j = 0; j < numPoints; j++)
    {
        std::size_t baseIdx = j * 2;
        T           x       = srcVec[baseIdx];
        T           y       = srcVec[baseIdx + 1];

        // For x,y in [-1,1], this fixed transform keeps w in [0.965,1.035].
        T w             = model[6] * x + model[7] * y + model[8];
        T x_transformed = (model[0] * x + model[1] * y + model[2]) / w;
        T y_transformed = (model[3] * x + model[4] * y + model[5]) / w;

        for (std::size_t i = 0; i < numSamples; i++)
        {
            dstVec[i * sampleSize + baseIdx]     = x_transformed;
            dstVec[i * sampleSize + baseIdx + 1] = y_transformed;
        }
    }
}

template<typename T>
void fill_tensor(nvcv::Tensor &tensor, const std::vector<T> &vec)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    CVCUDA_CHECK_DATA(tensorData);

    long3 strides{tensorData->stride(0), tensorData->stride(1)};
    long3 shape{tensorData->shape(0), tensorData->shape(1)};
    long  bufSize{nvcv::cuda::GetElement(strides, 0) * nvcv::cuda::GetElement(shape, 0)};
    CVCUDA_CHECK_DATA((bufSize == static_cast<long>(vec.size() * sizeof(T))));

    CUDA_CHECK_ERROR(cudaMemcpy(tensorData->basePtr(), vec.data(), bufSize, cudaMemcpyHostToDevice));
}

template<typename T>
inline void findhomography(nvbench::state &state, nvbench::type_list<T>)
try
{
    int2                        shape     = benchutils::GetShape<2, int2>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    std::vector<T> srcVec(2 * shape.x * shape.y);
    std::vector<T> dstVec(2 * shape.x * shape.y);
    std::vector<T> transform = {static_cast<T>(1.05),  static_cast<T>(0.08),   static_cast<T>(0.15),
                                static_cast<T>(-0.04), static_cast<T>(0.97),   static_cast<T>(-0.10),
                                static_cast<T>(0.015), static_cast<T>(-0.020), static_cast<T>(1.0)};

    fill_src_grid_replicated(srcVec, shape.x, shape.y);
    fill_dst_projective_replicated(srcVec, transform, dstVec, shape.x, shape.y);

    state.add_global_memory_reads(shape.x * shape.y * 4 * sizeof(T));
    state.add_global_memory_writes(shape.x * 3 * 3 * sizeof(T));

    cvcuda::FindHomography op(shape.x, shape.y);

    if (inputKind == benchutils::InputKind::Tensor)
    {
        // clang-format off
        nvcv::Tensor src({{shape.x, shape.y}, "NW"}, nvcv::TYPE_2F32);
        nvcv::Tensor dst({{shape.x, shape.y}, "NW"}, nvcv::TYPE_2F32);
        nvcv::Tensor models({{shape.x, 3, 3}, "NHW"}, benchutils::GetDataType<T>());
        // clang-format on

        fill_tensor(src, srcVec);
        fill_tensor(dst, dstVec);

        benchutils::warmup_and_exec(state, BENCH_FINDHOMOGRAPHY_WARMUP_ITERATIONS,
                                    [&op, &src, &dst, &models](cudaStream_t s) { op(s, src, dst, models); });
    }
    else
    {
        // FindHomography's variable-shape API uses TensorBatch rather than ImageBatchVarShape.
        nvcv::TensorBatch src(shape.x);
        nvcv::TensorBatch dst(shape.x);
        nvcv::TensorBatch models(shape.x);

        const std::size_t    sampleElements = 2 * static_cast<std::size_t>(shape.y);
        const std::vector<T> srcSample(srcVec.begin(), srcVec.begin() + sampleElements);
        const std::vector<T> dstSample(dstVec.begin(), dstVec.begin() + sampleElements);

        for (int i = 0; i < shape.x; ++i)
        {
            // clang-format off
            nvcv::Tensor srcTensor({{1, shape.y}, "NW"}, nvcv::TYPE_2F32);
            nvcv::Tensor dstTensor({{1, shape.y}, "NW"}, nvcv::TYPE_2F32);
            nvcv::Tensor modelTensor({{1, 3, 3}, "NHW"}, benchutils::GetDataType<T>());
            // clang-format on

            fill_tensor(srcTensor, srcSample);
            fill_tensor(dstTensor, dstSample);
            src.pushBack(srcTensor);
            dst.pushBack(dstTensor);
            models.pushBack(modelTensor);
        }

        benchutils::warmup_and_exec(state, BENCH_FINDHOMOGRAPHY_WARMUP_ITERATIONS,
                                    [&op, &src, &dst, &models](cudaStream_t s) { op(s, src, dst, models); });
    }
}

CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

using FindHomographyTypes = nvbench::type_list<float>;

NVBENCH_BENCH_TYPES(findhomography, NVBENCH_TYPE_AXES(FindHomographyTypes))
BENCH_FINDHOMOGRAPHY_AXES;
