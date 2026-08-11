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
#include "ops/generated/BenchBndBoxConfig.hpp"

#include <cvcuda/OpBndBox.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/Types.h>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class BenchBndBoxError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

template<typename T>
inline void bndbox(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         numBoxes  = benchutils::GetIntParam<int>(state, "numBoxes");
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("BndBox benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if ((isPlanar || isFakePlanar) && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Planar BndBox benchmark is tensor-only");
        return;
    }

    NVCVBndBoxI bndBox{
        {43, 21, 12,  34}, // box x, y position w, h size
        2, // box thickness
        { 0,  0,  0, 255}, // box border color
        { 0,  0,  0,   0}  // box fill color
    };

    std::vector<NVCVBndBoxI> flatBoxes(static_cast<size_t>(shape.x) * numBoxes, bndBox);
    std::vector<int32_t>     numBoxesPerBatch(shape.x, numBoxes);

    NVCVBndBoxesI bndBoxes = nullptr;
    if (nvcvBndBoxesIConstruct(&bndBoxes, flatBoxes.data(), numBoxesPerBatch.data(), shape.x) != NVCV_SUCCESS)
    {
        throw BenchBndBoxError("nvcvBndBoxesIConstruct failed");
    }
    auto bndBoxesGuard = std::unique_ptr<NVCVBndBoxesIRec, void (*)(NVCVBndBoxesI)>(
        bndBoxes, [](NVCVBndBoxesI h) { nvcvBndBoxesIDestroy(h); });

    const long imageBytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const long boxBytes   = static_cast<long>(shape.x) * numBoxes * sizeof(NVCVBndBoxI);
    if (isPlanar || isFakePlanar)
    {
        // Account for source, interleaved temporary, and destination image traffic.
        state.add_global_memory_reads(3 * imageBytes + boxBytes);
        state.add_global_memory_writes(3 * imageBytes);
    }
    else
    {
        state.add_global_memory_reads(imageBytes + boxBytes);
        state.add_global_memory_writes(imageBytes);
    }

    cvcuda::BndBox op;

    // clang-format off

    if (isFakePlanar)
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_BNDBOX_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &bndBoxes](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, bndBoxes);
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

        benchutils::warmup_and_exec(state, BENCH_BNDBOX_WARMUP_ITERATIONS,
            [&op, &src, &dst, &bndBoxes](cudaStream_t s) { op(s, src, dst, bndBoxes); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(bndbox, NVBENCH_TYPE_AXES(BENCH_BNDBOX_TYPES))
BENCH_BNDBOX_AXES;
