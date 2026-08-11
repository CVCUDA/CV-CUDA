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
#include "ops/generated/BenchRotateConfig.hpp"

#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpRotate.hpp>

#include <nvbench/nvbench.cuh>

#include <cmath>
#include <vector>

namespace {

// Sutherland-Hodgman half-plane clip.  axis=0 clips on x, axis=1 on y.
// keepAbove=true keeps points whose coord >= bound; false keeps coord <= bound.
inline std::vector<double2> clipHalfPlane(const std::vector<double2> &in, int axis, double bound, bool keepAbove)
{
    std::vector<double2> out;
    if (in.empty())
        return out;
    auto coord = [axis](double2 p)
    {
        return axis == 0 ? p.x : p.y;
    };
    auto inside = [&coord, &keepAbove, &bound](double2 p)
    {
        return keepAbove ? coord(p) >= bound : coord(p) <= bound;
    };
    auto inter = [&coord, &bound](double2 a, double2 b) -> double2
    {
        double t = (bound - coord(a)) / (coord(b) - coord(a));
        return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)};
    };
    for (size_t i = 0; i < in.size(); ++i)
    {
        double2 cur  = in[i];
        double2 prev = in[(i + in.size() - 1) % in.size()];
        bool    ci   = inside(cur);
        bool    pi   = inside(prev);
        if (ci)
        {
            if (!pi)
                out.push_back(inter(prev, cur));
            out.push_back(cur);
        }
        else if (pi)
        {
            out.push_back(inter(prev, cur));
        }
    }
    return out;
}

// Fraction of dst pixels in [0,W]x[0,H] whose inverse rotation maps inside source.
// The rotate kernel skips both read and write outside this region, so the byte model
// must be scaled by this fraction to track HBM traffic instead of nominal coverage.
inline double rotateInBoundsFraction(double angleDeg, double xShift, double yShift, double W, double H)
{
    const double th = angleDeg * M_PI / 180.0;
    const double c  = std::cos(th);
    const double s  = std::sin(th);
    // Kernel mapping is src = R(+θ) * (dst - shift), so the in-bounds dst region is
    // the image of [0,W]x[0,H] under  dst = R(-θ) * src + shift.
    auto         map = [&c, &s, &xShift, &yShift](double x, double y)
    {
        return double2{c * x + s * y + xShift, -s * x + c * y + yShift};
    };
    std::vector<double2> p = {map(0.0, 0.0), map(W, 0.0), map(W, H), map(0.0, H)};
    p                      = clipHalfPlane(p, 0, 0.0, true);
    p                      = clipHalfPlane(p, 0, W, false);
    p                      = clipHalfPlane(p, 1, 0.0, true);
    p                      = clipHalfPlane(p, 1, H, false);
    if (p.size() < 3)
        return 0.0;
    double area = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
    {
        size_t j = (i + 1) % p.size();
        area += p[i].x * p[j].y - p[j].x * p[i].y;
    }
    return std::abs(area) * 0.5 / (W * H);
}

} // namespace

template<typename T>
inline void rotate(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("Rotate benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // NCHW_FAKE ("fake planar") is a tensor-only comparison path: planar data is reformatted to
    // interleaved, rotated with the interleaved kernel, and reformatted back to planar — all timed
    // together — so the native planar path (NCHW) can be shown to be faster than this naive
    // convert→rotate→convert pipeline.
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) rotate benchmark is tensor-only");
        return;
    }

    // Rotation around image center: keeps the bulk of dst pixels inside source so
    // the kernel actually exercises the read+interp+write path rather than the
    // out-of-bounds early-exit guard at rotate.cu:61.
    const double angleDeg = 30.0;
    const double th       = angleDeg * M_PI / 180.0;
    const double cx       = static_cast<double>(shape.z) * 0.5;
    const double cy       = static_cast<double>(shape.y) * 0.5;
    const double xShift   = cx * (1.0 - std::cos(th)) - cy * std::sin(th);
    const double yShift   = cy * (1.0 - std::cos(th)) + cx * std::sin(th);
    double2      shift{xShift, yShift};

    const double inBoundsFrac
        = rotateInBoundsFraction(angleDeg, xShift, yShift, static_cast<double>(shape.z), static_cast<double>(shape.y));
    const double fullBytes = static_cast<double>(shape.x) * shape.y * shape.z * sizeof(T);
    // Native rotate only touches the in-bounds region; the two reformats in the fake-planar path move
    // the full tensor each way (reformat is full-coverage), so add 2*fullBytes on top of the rotate.
    const auto   bytesIO = static_cast<size_t>((isFakePlanar ? 2.0 * fullBytes : 0.0) + inBoundsFrac * fullBytes);
    state.add_global_memory_reads(bytesIO);
    state.add_global_memory_writes(bytesIO);

    cvcuda::Rotate op(shape.x);

    // clang-format off

    if (isFakePlanar) // tensor-only: planar→interleaved→rotate→interleaved→planar
    {
        using BT = typename nvcv::cuda::BaseType<T>;
        int  ch  = nvcv::cuda::NumElements<T>;

        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_ROTATE_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &angleDeg, &shift, &interpType](cudaStream_t s) {
                reformatOp(s, src, interSrc);                       // NCHW → NHWC
                op(s, interSrc, interDst, angleDeg, shift, interpType); // interleaved rotate
                reformatOp(s, interDst, dst);                       // NHWC → NCHW
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        using BT = typename nvcv::cuda::BaseType<T>;
        int  ch  = nvcv::cuda::NumElements<T>;

        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_ROTATE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &angleDeg, &shift, &interpType](cudaStream_t s) {
                op(s, src, dst, angleDeg, shift, interpType);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
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
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          benchutils::LcgValues<T>());
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        nvcv::Tensor angleDegTensor({{shape.x}, "N"}, nvcv::TYPE_F64);
        nvcv::Tensor shiftTensor({{shape.x, 2}, "NW"}, nvcv::TYPE_F64);

        benchutils::FillTensor<double>(angleDegTensor, [&angleDeg](const long4_16a &){ return angleDeg; });
        benchutils::FillTensor<double>(
            shiftTensor, [&shift](const long4_16a &c) { return nvcv::cuda::GetElement(shift, static_cast<int>(c.y)); });

        benchutils::warmup_and_exec(state, BENCH_ROTATE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &angleDegTensor, &shiftTensor, &interpType](cudaStream_t s) {
                op(s, src, dst, angleDegTensor, shiftTensor, interpType);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(rotate, NVBENCH_TYPE_AXES(BENCH_ROTATE_TYPES))
BENCH_ROTATE_AXES;
