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
#include "ops/generated/BenchLaplacianConfig.hpp"

#include <cvcuda/OpLaplacian.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void laplacian(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    int                         ksize     = benchutils::GetIntParam<int>(state, "ksize");
    auto                        scale     = static_cast<float>(state.get_float64("scale"));

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("Laplacian benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) Laplacian benchmark is tensor-only");
        return;
    }

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Laplacian op;

    // clang-format off

    if (isFakePlanar) // tensor-only: planar->interleaved->laplacian->interleaved->planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_LAPLACIAN_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &ksize, &scale, &borderType](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, ksize, scale, borderType);
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

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_LAPLACIAN_WARMUP_ITERATIONS,
            [&op, &src, &dst, &ksize, &scale, &borderType](cudaStream_t s) {
                op(s, src, dst, ksize, scale, borderType);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
            benchutils::FillPlanarImageBatchLike<T>(dst, src);
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        nvcv::Tensor ksizeTensor({{shape.x}, "N"}, nvcv::TYPE_S32);
        nvcv::Tensor scaleTensor({{shape.x}, "N"}, nvcv::TYPE_F32);

        benchutils::FillTensor<int>(ksizeTensor, [&ksize](const long4_16a &){ return ksize; });
        benchutils::FillTensor<float>(scaleTensor, [&scale](const long4_16a &){ return scale; });

        benchutils::warmup_and_exec(state, BENCH_LAPLACIAN_WARMUP_ITERATIONS,
            [&op, &src, &dst, &ksizeTensor, &scaleTensor, &borderType](cudaStream_t s) {
                op(s, src, dst, ksizeTensor, scaleTensor, borderType);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(laplacian, NVBENCH_TYPE_AXES(BENCH_LAPLACIAN_TYPES))
BENCH_LAPLACIAN_AXES;
