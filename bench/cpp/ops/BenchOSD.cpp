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
#include "ops/generated/BenchOSDConfig.hpp"

#include <cvcuda/OpOSD.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/Types.h>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class BenchOSDError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

template<typename T>
inline void osd(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         numElem   = benchutils::GetIntParam<int>(state, "numElem");
    auto                        elemType  = state.get_string("elementType");
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");

    int ch = nvcv::cuda::NumElements<T>;

    using BT = nvcv::cuda::BaseType<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("OSD benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if ((isPlanar || isFakePlanar) && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Planar OSD benchmark is tensor-only");
        return;
    }

    NVCVPoint point;
    point.centerPos.x = shape.z / 2;
    point.centerPos.y = shape.y / 2;
    point.radius      = std::min(shape.z, shape.y) / 2;
    point.color       = {0, 0, 0, 255};

    NVCVBndBoxI bndBox;
    bndBox.box.x       = shape.z / 4;
    bndBox.box.y       = shape.y / 4;
    bndBox.box.width   = shape.z / 2;
    bndBox.box.height  = shape.y / 2;
    bndBox.thickness   = 3;
    bndBox.fillColor   = {0, 128, 255, 0};
    bndBox.borderColor = {255, 255, 0, 255};

    NVCVLine line;
    line.pos0.x        = shape.z / 8;
    line.pos0.y        = shape.y / 8;
    line.pos1.x        = 7 * shape.z / 8;
    line.pos1.y        = 7 * shape.y / 8;
    line.thickness     = 4;
    line.color         = {255, 0, 0, 255};
    line.interpolation = true;

    NVCVCircle circle;
    circle.centerPos.x = shape.z / 2;
    circle.centerPos.y = shape.y / 2;
    circle.radius      = std::min(shape.z, shape.y) / 4;
    circle.thickness   = 4;
    circle.borderColor = {0, 255, 255, 255};
    circle.bgColor     = {255, 0, 255, 0};

    NVCVOSDType osdType = NVCV_OSD_POINT;
    const void *payload = &point;
    if (elemType == "RECT")
    {
        osdType = NVCV_OSD_RECT;
        payload = &bndBox;
    }
    else if (elemType == "LINE")
    {
        osdType = NVCV_OSD_LINE;
        payload = &line;
    }
    else if (elemType == "CIRCLE")
    {
        osdType = NVCV_OSD_CIRCLE;
        payload = &circle;
    }
    else if (elemType != "POINT")
    {
        throw std::invalid_argument("Unsupported OSD elementType: " + elemType);
    }

    const int32_t             totalElems = shape.x * numElem;
    std::vector<NVCVOSDType>  types(totalElems, osdType);
    std::vector<const void *> payloads(totalElems, payload);
    std::vector<int32_t>      numElementsPerBatch(shape.x, numElem);

    NVCVElements ctxHandle = nullptr;
    if (nvcvElementsConstruct(&ctxHandle, types.data(), payloads.data(), numElementsPerBatch.data(), shape.x)
        != NVCV_SUCCESS)
    {
        throw BenchOSDError("nvcvElementsConstruct failed");
    }
    auto ctxGuard = std::unique_ptr<NVCVElementsRec, void (*)(NVCVElements)>(
        ctxHandle, [](NVCVElements h) { nvcvElementsDestroy(h); });

    const long imageBytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const long elemBytes  = static_cast<long>(numElem) * sizeof(int) * 16;
    if (isPlanar || isFakePlanar)
    {
        state.add_global_memory_reads(3 * imageBytes + elemBytes);
        state.add_global_memory_writes(3 * imageBytes);
    }
    else
    {
        state.add_global_memory_reads(imageBytes + elemBytes);
        state.add_global_memory_writes(imageBytes);
    }

    cvcuda::OSD op;

    // clang-format off

    if (isFakePlanar)
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_OSD_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &ctxHandle](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, ctxHandle);
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

        benchutils::warmup_and_exec(state, BENCH_OSD_WARMUP_ITERATIONS,
            [&op, &src, &dst, &ctxHandle](cudaStream_t s) { op(s, src, dst, ctxHandle); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(osd, NVBENCH_TYPE_AXES(BENCH_OSD_TYPES))
BENCH_OSD_AXES;
