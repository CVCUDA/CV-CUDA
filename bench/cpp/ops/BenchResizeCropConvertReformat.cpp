/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchResizeCropConvertReformatConfig.hpp"

#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void resizecropconvertreformat(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                       srcShape  = benchutils::GetShape<3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    const std::string           layout    = state.get_string("layout");

    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    using BT = nvcv::cuda::BaseType<T>;
    long nc  = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";

    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("ResizeCropConvertReformat benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }

    if (isFakePlanar && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Fake-planar (NCHW_FAKE) ResizeCropConvertReformat benchmark is tensor-only");
        return;
    }

    // Guard against degenerate shapes that would cause divide-by-zero or invalid crops
    if (srcShape.y <= 1 || srcShape.z <= 1)
    {
        state.skip("Height and width must be > 1 for resize/crop operations");
        return;
    }

    // Resize to 0.5x (shrink by half) and crop
    // NVCVSize2D: {width, height} - srcShape.z is width, srcShape.y is height
    auto resize = NVCVSize2D{static_cast<int>(srcShape.z / 2), static_cast<int>(srcShape.y / 2)};

    // Crop region at origin, clamped to valid range
    int2 cropPos{0, 0};
    int  crop_w = std::max(1, std::min(512, resize.w - 1));
    int  crop_h = std::max(1, std::min(512, resize.h - 1));
    int2 cropSize{crop_w, crop_h};

    // NO_OP channel manipulation
    NVCVChannelManip manip = NVCV_CHANNEL_NO_OP;

    long3 dstShape{srcShape.x, cropSize.y, cropSize.x};

    // Calculate actual source region read (crop maps back to a smaller region in source)
    // scale = src_size / resize_size, src_region = (crop_pos + crop_size) * scale + margin
    float scale_x      = static_cast<float>(srcShape.z) / static_cast<float>(resize.w);
    float scale_y      = static_cast<float>(srcShape.y) / static_cast<float>(resize.h);
    int   src_region_w = std::min(static_cast<int>(std::ceil(static_cast<float>(cropPos.x + crop_w) * scale_x)) + 2,
                                  static_cast<int>(srcShape.z));
    int   src_region_h = std::min(static_cast<int>(std::ceil(static_cast<float>(cropPos.y + crop_h) * scale_y)) + 2,
                                  static_cast<int>(srcShape.y));

    const long srcBytes = srcShape.x * srcShape.y * srcShape.z * sizeof(BT) * nc;
    const long dstBytes = dstShape.x * dstShape.y * dstShape.z * sizeof(BT) * nc;

    // Memory: read source region, write destination crop. Fake planar also reformats the full source.
    state.add_global_memory_reads(srcShape.x * src_region_h * src_region_w * sizeof(BT) * nc
                                  + (isFakePlanar ? srcBytes : 0));
    state.add_global_memory_writes(dstBytes + (isFakePlanar ? srcBytes : 0));

    cvcuda::ResizeCropConvertReformat op;

    // clang-format off

    if (inputKind == benchutils::InputKind::Tensor)
    {
        if (isPlanar || isFakePlanar)
        {
            nvcv::Tensor src({{srcShape.x, nc, srcShape.y, srcShape.z}, "NCHW"}, benchutils::GetDataType<BT>());
            nvcv::Tensor dst({{dstShape.x, nc, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>());

            benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

            if (isFakePlanar)
            {
                nvcv::Tensor     tmp({{srcShape.x, srcShape.y, srcShape.z, nc}, "NHWC"}, benchutils::GetDataType<BT>());
                cvcuda::Reformat reformatOp;

                benchutils::warmup_and_exec(state, BENCH_RESIZECROPCONVERTREFORMAT_WARMUP_ITERATIONS,
                    [&op, &reformatOp, &src, &tmp, &dst, &resize, &interpType, &cropPos, &manip](cudaStream_t s) {
                        reformatOp(s, src, tmp);
                        op(s, tmp, dst, resize, interpType, cropPos, manip);
                    });
            }
            else
            {
                benchutils::warmup_and_exec(state, BENCH_RESIZECROPCONVERTREFORMAT_WARMUP_ITERATIONS,
                    [&op, &src, &dst, &resize, &interpType, &cropPos, &manip](cudaStream_t s) {
                        op(s, src, dst, resize, interpType, cropPos, manip);
                    });
            }
        }
        else
        {
            nvcv::Tensor src({{srcShape.x, srcShape.y, srcShape.z, nc}, "NHWC"}, benchutils::GetDataType<BT>());
            nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, nc}, "NHWC"}, benchutils::GetDataType<BT>());

            benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

            benchutils::warmup_and_exec(state, BENCH_RESIZECROPCONVERTREFORMAT_WARMUP_ITERATIONS,
                [&op, &src, &dst, &resize, &interpType, &cropPos, &manip](cudaStream_t s) {
                    op(s, src, dst, resize, interpType, cropPos, manip);
                });
        }
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(static_cast<int32_t>(srcShape.x));
        nvcv::Tensor dst = isPlanar
            ? nvcv::Tensor({{dstShape.x, nc, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>())
            : nvcv::Tensor({{dstShape.x, dstShape.y, dstShape.z, nc}, "NHWC"}, benchutils::GetDataType<BT>());

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0});
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
        }

        benchutils::warmup_and_exec(state, BENCH_RESIZECROPCONVERTREFORMAT_WARMUP_ITERATIONS,
            [&op, &src, &dst, &resize, &interpType, &cropPos, &manip](cudaStream_t s) {
                op(s, src, dst, resize, interpType, cropPos, manip);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(resizecropconvertreformat, NVBENCH_TYPE_AXES(BENCH_RESIZECROPCONVERTREFORMAT_TYPES))
BENCH_RESIZECROPCONVERTREFORMAT_AXES;
