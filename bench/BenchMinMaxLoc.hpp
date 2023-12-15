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

#ifndef CVCUDA_BENCH_MINMAXLOC_HPP
#define CVCUDA_BENCH_MINMAXLOC_HPP

#include "BenchUtils.hpp"

namespace benchutils {

template<typename VT, typename R = Randomizer<VT>, typename RE = typename R::RE, typename UD = typename R::UD,
         class = nvcv::cuda::Require<nvcv::cuda::NumComponents<VT> == 0 && std::is_integral_v<VT>>>
inline auto RandomValuesWithoutMinMax(VT min = nvcv::cuda::TypeTraits<VT>::min,
                                      VT max = nvcv::cuda::TypeTraits<VT>::max, RE rng = DefaultGenerator())
{
    return R{UD(min + 1, max - 1), rng};
}

template<typename VT, typename R = Randomizer<VT>, typename RE = typename R::RE>
inline void RandomMinMax(std::vector<uint8_t> &srcVec, const long3 &shape, const long3 &strides, long locs,
                         RE randomGenerator = DefaultGenerator())
{
    long locsPerHeight = static_cast<int>(std::ceil(locs / shape.y));
    if (locsPerHeight * 2 >= shape.z)
    {
        throw std::runtime_error("Locations is bigger than available pixels");
    }

    for (long x = 0; x < shape.x; ++x)
    {
        long countLocs = 0;
        for (long y = 0; y < shape.y; ++y)
        {
            long numLocs = (countLocs + locsPerHeight > locs) ? (locs - countLocs) : locsPerHeight;
            for (long z = 0; z < numLocs; ++z)
            {
                ValueAt<VT>(srcVec, strides, long3{x, y, z}) = nvcv::cuda::TypeTraits<VT>::min;
            }
            for (long z = numLocs; z < numLocs * 2; ++z)
            {
                ValueAt<VT>(srcVec, strides, long3{x, y, z}) = nvcv::cuda::TypeTraits<VT>::max;
            }
            std::shuffle(&ValueAt<VT>(srcVec, strides, long3{x, y, 0}),
                         &ValueAt<VT>(srcVec, strides, long3{x, y, shape.z}), randomGenerator);

            countLocs += locsPerHeight;
        }
    }
}

template<typename VT>
inline void FillTensorWithMinMax(const nvcv::Tensor &tensor, long locations)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    CVCUDA_CHECK_DATA(tensorData);

    if (tensor.rank() != 3 && tensor.rank() != 4)
    {
        throw std::invalid_argument("Tensor rank is not 3 or 4");
    }

    long3 strides{tensorData->stride(0), tensorData->stride(1), tensorData->stride(2)};
    long3 shape{tensorData->shape(0), tensorData->shape(1), tensorData->shape(2)};
    long  bufSize{nvcv::cuda::GetElement(strides, 0) * nvcv::cuda::GetElement(shape, 0)};

    std::vector<uint8_t> tensorVec(bufSize);

    FillBuffer<VT>(tensorVec, shape, strides, RandomValuesWithoutMinMax<VT>());

    RandomMinMax<VT>(tensorVec, shape, strides, locations);

    CUDA_CHECK_ERROR(cudaMemcpy(tensorData->basePtr(), tensorVec.data(), bufSize, cudaMemcpyHostToDevice));
}

template<typename VT>
inline void FillImageBatchWithMinMax(nvcv::ImageBatchVarShape &imageBatch, long2 size, long2 varSize, long locations)
{
    auto randomWidth  = RandomValues<int>(static_cast<int>(size.x - varSize.x), static_cast<int>(size.x));
    auto randomHeight = RandomValues<int>(static_cast<int>(size.y - varSize.y), static_cast<int>(size.y));

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        nvcv::Image image(nvcv::Size2D{randomWidth(), randomHeight()}, GetFormat<VT>());

        auto data = image.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);

        long2 strides{data->plane(0).rowStride, sizeof(VT)};
        long2 shape{data->plane(0).height, data->plane(0).width};
        long  bufSize{strides.x * shape.x};

        std::vector<uint8_t> imageBuffer(bufSize);

        FillBuffer<VT>(imageBuffer, shape, strides, RandomValuesWithoutMinMax<VT>());

        RandomMinMax<VT>(imageBuffer, long3{1, shape.x, shape.y}, long3{bufSize, strides.x, strides.y}, locations);

        CUDA_CHECK_ERROR(cudaMemcpy2D(data->plane(0).basePtr, strides.x, imageBuffer.data(), strides.x, strides.x,
                                      data->plane(0).height, cudaMemcpyHostToDevice));

        imageBatch.pushBack(image);
    }
}

} // namespace benchutils

#endif // CVCUDA_BENCH_MINMAXLOC_HPP
