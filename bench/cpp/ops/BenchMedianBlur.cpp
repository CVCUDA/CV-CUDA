/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchMedianBlurConfig.hpp"

#include <cvcuda/OpMedianBlur.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T, class VG>
inline void FillPlanarImageBatch(nvcv::ImageBatchVarShape &imageBatch, long2 size, long2 varSize, VG valuesGenerator)
{
    using BT = typename nvcv::cuda::BaseType<T>;

    nvcv::ImageFormat format = benchutils::GetPlanarFormat<T>();
    auto randomWidth  = benchutils::LcgValues<int>(static_cast<int>(size.x - varSize.x), static_cast<int>(size.x));
    auto randomHeight = benchutils::LcgValues<int>(static_cast<int>(size.y - varSize.y), static_cast<int>(size.y));

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        nvcv::Image image(nvcv::Size2D{randomWidth(), randomHeight()}, format);

        auto data = image.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);

        for (int p = 0; p < format.numPlanes(); ++p)
        {
            long2 strides{data->plane(p).rowStride, sizeof(BT)};
            long2 shape{data->plane(p).height, data->plane(p).width};

            std::vector<uint8_t> imageBuffer(strides.x * shape.x);

            benchutils::FillBuffer<BT>(imageBuffer, shape, strides, valuesGenerator);
            CUDA_CHECK_ERROR(cudaMemcpy2D(data->plane(p).basePtr, strides.x, imageBuffer.data(), strides.x, strides.x,
                                          data->plane(p).height, cudaMemcpyHostToDevice));
        }

        imageBatch.pushBack(image);
    }
}

template<typename T>
inline void medianblur(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    int2 kernelSize = nvcv::cuda::StaticCast<int>(benchutils::GetShape<2>(state.get_string("kernelSize")));

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("MedianBlur benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) MedianBlur benchmark is tensor-only");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    nvcv::Size2D kernelSize2d{kernelSize.x, kernelSize.y};

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    if (isFakePlanar)
    {
        state.add_global_memory_reads(3 * bytes);
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes);
        state.add_global_memory_writes(bytes);
    }

    cvcuda::MedianBlur op(shape.x);

    // clang-format off

    // Use horizontal DECREASING gradient pattern for deterministic timing.
    // Random data causes the quickselect median algorithm to have variable performance.
    // Decreasing gradient: pixel value = 255 - (column_index % 256)
    // Note: For Tensor (4D NHWC), column is c.z; for ImageBatch (2D HW), column is c.y
    auto gradientValue = [](long col) -> BT
    {
        if constexpr (std::is_floating_point_v<BT>)
            return static_cast<BT>(255 - (col % 256)) / BT{255};
        else
            return static_cast<BT>(255 - (col % 256));
    };
    auto gradientPatternTensor = [&gradientValue](const long4_16a &c) -> BT
    {
        return gradientValue(c.z); // c.z = W for NHWC
    };
    auto gradientPatternTensorPlanar = [&gradientValue](const long4_16a &c) -> BT
    {
        return gradientValue(c.w); // c.w = W for NCHW
    };
    auto gradientPatternImage = [&gradientValue](const long4_16a &c) -> T
    {
        const BT val = gradientValue(c.y); // c.y = W for HW (FillBuffer uses x=H, y=W)
        T        ret;
        for (int i = 0; i < nvcv::cuda::NumElements<T>; ++i)
        {
            nvcv::cuda::GetElement(ret, i) = val;
        }
        return ret;
    };
    auto gradientPatternPlane = [&gradientValue](const long4_16a &c) -> BT
    {
        return gradientValue(c.y); // c.y = W for HW planes
    };

    if (isFakePlanar) // tensor-only: planar→interleaved→medianblur→interleaved→planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, gradientPatternTensorPlanar);

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_MEDIANBLUR_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &kernelSize2d](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, kernelSize2d);
                reformatOp(s, interDst, dst);
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        if (isPlanar)
        {
            benchutils::FillTensor<BT>(src, gradientPatternTensorPlanar);
        }
        else
        {
            benchutils::FillTensor<BT>(src, gradientPatternTensor);
        }

        benchutils::warmup_and_exec(state, BENCH_MEDIANBLUR_WARMUP_ITERATIONS,
            [&op, &src, &dst, &kernelSize2d](cudaStream_t s) { op(s, src, dst, kernelSize2d); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        if (isPlanar)
        {
            FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0}, gradientPatternPlane);
            benchutils::FillPlanarImageBatchLike<T>(dst, src);
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          gradientPatternImage);
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        // Kernel size tensor must be 1D tensor of int2 (TYPE_2S32), matching TestOpMedianBlur.cpp
        nvcv::Tensor kernelSizeTensor(nvcv::TensorShape({shape.x}, "N"), nvcv::TYPE_2S32);

        benchutils::FillTensor<int2>(kernelSizeTensor,
                                     [&kernelSize](const long4_16a &) { return kernelSize; });

        benchutils::warmup_and_exec(state, BENCH_MEDIANBLUR_WARMUP_ITERATIONS,
            [&op, &src, &dst, &kernelSizeTensor](cudaStream_t s) { op(s, src, dst, kernelSizeTensor); });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(medianblur, NVBENCH_TYPE_AXES(BENCH_MEDIANBLUR_TYPES))
BENCH_MEDIANBLUR_AXES;
