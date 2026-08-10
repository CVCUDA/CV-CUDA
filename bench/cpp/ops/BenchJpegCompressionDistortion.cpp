/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchJpegCompressionDistortionConfig.hpp"

#include <cvcuda/OpJpegCompressionDistortion.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

// Quality only scales the quantization tables (built once per CUDA block from constant data), so
// it does not affect throughput; use a fixed mid-range value.
static constexpr int kQuality = 50;

template<typename T>
inline void jpegcompressiondistortion(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("JpegCompressionDistortion benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // NCHW_FAKE ("fake planar") is a tensor-only comparison path: planar data is reformatted to
    // interleaved, distorted with the interleaved kernel, and reformatted back — all timed together —
    // so the native planar path (NCHW) can be shown to beat the naive convert->op->convert pipeline.
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) jpegcompressiondistortion benchmark is tensor-only");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    // JpegCompressionDistortion preserves size, so src and dst hold the same number of bytes.
    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    if (isFakePlanar)
    {
        // reformat(NCHW->NHWC) + distort + reformat(NHWC->NCHW): reads src + interleaved src + dst,
        // writes interleaved src + interleaved dst + dst.
        state.add_global_memory_reads(3 * bytes);
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes);
        state.add_global_memory_writes(bytes);
    }

    cvcuda::JpegCompressionDistortion op;

    // clang-format off
    if (isFakePlanar) // tensor-only: planar->interleaved->distort->interleaved->planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_JPEGCOMPRESSIONDISTORTION_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst](cudaStream_t s) {
                reformatOp(s, src, interSrc);                  // NCHW -> NHWC
                op(s, interSrc, interDst, kQuality);           // interleaved distortion
                reformatOp(s, interDst, dst);                  // NHWC -> NCHW
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

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_JPEGCOMPRESSIONDISTORTION_WARMUP_ITERATIONS,
            [&op, &src, &dst](cudaStream_t s) { op(s, src, dst, kQuality); });
    }
    else // ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
            benchutils::FillPlanarImageBatch<T>(dst, long2{shape.z, shape.y}, long2{0, 0});
        }
        else
        {
            // The 1-channel (grayscale/luma) profile uses the non-color U8 format; only the
            // 3-channel profile carries the RGB color model the operator requires.
            if constexpr (nvcv::cuda::NumElements<T> == 3)
            {
                benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                              benchutils::CheckerboardValues<T>(), benchutils::GetRGBFormat<T>());
            }
            else
            {
                benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                              benchutils::CheckerboardValues<T>(), nvcv::FMT_U8);
            }
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        benchutils::warmup_and_exec(state, BENCH_JPEGCOMPRESSIONDISTORTION_WARMUP_ITERATIONS,
            [&op, &src, &dst](cudaStream_t s) { op(s, src, dst, kQuality); });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(jpegcompressiondistortion, NVBENCH_TYPE_AXES(BENCH_JPEGCOMPRESSIONDISTORTION_TYPES))
BENCH_JPEGCOMPRESSIONDISTORTION_AXES;
