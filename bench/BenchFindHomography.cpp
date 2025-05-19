/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpFindHomography.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
void fill_vector(std::vector<T> &vec)
{
    auto random_val = benchutils::RandomValues<T>();
    for (std::size_t i = 0; i < vec.size(); i++)
    {
        vec[i] = random_val();
    }
}

template<typename T>
void fill_dst(const std::vector<T> &srcVec, const std::vector<T> &modelsVec, std::vector<T> &dstVec, std::size_t numSamples)
{
    for (std::size_t i = 0; i < numSamples; i++) 
    {
        const T* model   = &modelsVec[i * 9];
        std::size_t numPoints = srcVec.size() / (2 * numSamples);
        for (std::size_t j = 0; j < numPoints; j++) 
        {
            T x = srcVec[i * numPoints * 2 + j * 2];
            T y = srcVec[i * numPoints * 2 + j * 2 + 1];
            
            // Apply homography transformation:
            T w = model[6] * x + model[7] * y + model[8];
            
            if (std::abs(w) > 1e-10) {
                w = 1.0 / w;
                T x_transformed = (model[0] * x + model[1] * y + model[2]) * w;
                T y_transformed = (model[3] * x + model[4] * y + model[5]) * w;
                
                dstVec[i * numPoints * 2 + j * 2] = x_transformed;
                dstVec[i * numPoints * 2 + j * 2 + 1] = y_transformed;
            } else {
                dstVec[i * numPoints * 2 + j * 2] = 0;
                dstVec[i * numPoints * 2 + j * 2 + 1] = 0;
            }
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
    CVCUDA_CHECK_DATA(bufSize == vec.size());

    CUDA_CHECK_ERROR(cudaMemcpy(tensorData->basePtr(), vec.data(), bufSize, cudaMemcpyHostToDevice));
}

template<typename T>
inline void FindHomography(nvbench::state &state, nvbench::type_list<T>)
try
{
    long2 shape      = benchutils::GetShape<2>(state.get_string("shape"));
    bool  batch      = state.get_int64("batch");
    auto  numSamples = batch ? 1 : shape.x;

    nvcv::Tensor src({{numSamples, shape.y}, "NW"}, nvcv::TYPE_2F32);
    nvcv::Tensor dst({{numSamples, shape.y}, "NW"}, nvcv::TYPE_2F32);
    nvcv::Tensor models({{numSamples, 3, 3}, "NHW"}, benchutils::GetDataType<T>());

    std::vector<T> srcVec(2 * numSamples * shape.y);
    std::vector<T> dstVec(2 * numSamples * shape.y);
    std::vector<T> modelsVec(9);

    fill_vector(srcVec);
    fill_vector(modelsVec);
    fill_dst(srcVec, modelsVec, dstVec, numSamples);
    fill_tensor(src, srcVec);
    fill_tensor(dst, dstVec);

    state.add_global_memory_reads(shape.x * shape.y * 4 * sizeof(T));
    state.add_global_memory_writes(shape.x * 3 * 3 * sizeof(T));

    cvcuda::FindHomography op(shape.x, shape.y);

    if (!batch) // negative var shape means use Tensor
    {
        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &models](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, models);
        });
    }
    else
    {
        nvcv::TensorBatch srcTensors(shape.x);
        nvcv::TensorBatch dstTensors(shape.x);
        nvcv::TensorBatch modelsTensors(shape.x);
        srcTensors.pushBack(src);
        dstTensors.pushBack(dst);
        modelsTensors.pushBack(models);

        state.exec(nvbench::exec_tag::sync, [&op, &srcTensors, &dstTensors, &modelsTensors](nvbench::launch &launch)
        {
            op(launch.get_stream(), srcTensors, dstTensors, modelsTensors);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using FindHomographyTypes = nvbench::type_list<float>;

NVBENCH_BENCH_TYPES(FindHomography, NVBENCH_TYPE_AXES(FindHomographyTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1024"})
    .add_int64_axis("batch", {false, true});
