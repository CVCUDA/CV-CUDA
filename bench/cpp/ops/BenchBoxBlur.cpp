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
#include "ops/generated/BenchBoxBlurConfig.hpp"

#include <cvcuda/OpBoxBlur.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/Types.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class BenchBoxBlurError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

inline NVCVBlurBoxI MakeBoxBlurBox(const std::string &boxPattern, int2 boxSize, int3 shape, int kernelSize,
                                   int numBoxes, int boxIdx)
{
    int boxWidth  = std::max(3, std::min<int>(boxSize.x, shape.z));
    int boxHeight = std::max(3, std::min<int>(boxSize.y, shape.y));
    int x         = 43;
    int y         = 21;

    if (boxPattern == "grid")
    {
        int gridCols = 1;
        while (gridCols * gridCols < numBoxes)
        {
            ++gridCols;
        }
        int gridRows = (numBoxes + gridCols - 1) / gridCols;
        int col      = boxIdx % gridCols;
        int row      = boxIdx / gridCols;
        int maxX     = std::max(0, shape.z - boxWidth);
        int maxY     = std::max(0, shape.y - boxHeight);
        x            = gridCols <= 1 ? maxX / 2 : col * maxX / (gridCols - 1);
        y            = gridRows <= 1 ? maxY / 2 : row * maxY / (gridRows - 1);
    }
    else if (boxPattern != "fixed")
    {
        throw std::invalid_argument("Unexpected boxPattern = " + boxPattern);
    }

    return NVCVBlurBoxI{
        {x, y, boxWidth, boxHeight},
        kernelSize
    };
}

template<typename T>
inline void boxblur(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape      = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind  = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         numBoxes   = benchutils::GetIntParam<int>(state, "numBoxes");
    int                         kernelSize = benchutils::GetIntParam<int>(state, "kernelSize");
    int2                        boxSize    = benchutils::GetShape<2, int2>(state.get_string("boxSize"));
    auto                        boxPattern = state.get_string("boxPattern");
    auto                        layout     = benchutils::GetStringParam(state, "layout", "NHWC");

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("BoxBlur benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Fake-planar (NCHW_FAKE) BoxBlur benchmark is tensor-only");
        return;
    }

    std::vector<NVCVBlurBoxI> flatBoxes;
    flatBoxes.reserve(static_cast<size_t>(shape.x) * numBoxes);
    for (long n = 0; n < shape.x; ++n)
    {
        for (int i = 0; i < numBoxes; ++i)
        {
            flatBoxes.push_back(MakeBoxBlurBox(boxPattern, boxSize, shape, kernelSize, numBoxes, i));
        }
    }
    std::vector<int32_t> numBoxesPerBatch(shape.x, numBoxes);

    NVCVBlurBoxesI blurBoxes = nullptr;
    if (nvcvBlurBoxesIConstruct(&blurBoxes, flatBoxes.data(), numBoxesPerBatch.data(), shape.x) != NVCV_SUCCESS)
    {
        throw BenchBoxBlurError("nvcvBlurBoxesIConstruct failed");
    }
    auto blurBoxesGuard = std::unique_ptr<NVCVBlurBoxesIRec, void (*)(NVCVBlurBoxesI)>(
        blurBoxes, [](NVCVBlurBoxesI h) { nvcvBlurBoxesIDestroy(h); });

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) + shape.x * numBoxes * sizeof(NVCVBlurBoxI));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::BoxBlur op;

    // clang-format off

    if (isFakePlanar)
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_BOXBLUR_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &blurBoxes](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, blurBoxes);
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

        benchutils::warmup_and_exec(state, BENCH_BOXBLUR_WARMUP_ITERATIONS,
            [&op, &src, &dst, &blurBoxes](cudaStream_t s) { op(s, src, dst, blurBoxes); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(boxblur, NVBENCH_TYPE_AXES(BENCH_BOXBLUR_TYPES))
BENCH_BOXBLUR_AXES;
