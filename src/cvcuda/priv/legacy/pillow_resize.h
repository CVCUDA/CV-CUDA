/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <nvcv/Rect.h>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

#define work_type float
#define M_PI      3.14159265358979323846 /* pi */

namespace nvcv::legacy::cuda_op {

static constexpr float        bilinear_filter_support = 1.f;
static constexpr float        box_filter_support      = 0.5f;
static constexpr float        hamming_filter_support  = 1.f;
static constexpr float        bicubic_filter_support  = 2.f;
static constexpr float        lanczos_filter_support  = 3.f;
static constexpr unsigned int precision_bits          = 32 - 8 - 2;

class BilinearFilter
{
public:
    __host__ __device__ BilinearFilter()
        : _support(bilinear_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        if (x < 0.0)
        {
            x = -x;
        }
        if (x < 1.0)
        {
            return 1.0 - x;
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class BoxFilter
{
public:
    __host__ __device__ BoxFilter()
        : _support(box_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        const float half_pixel = 0.5;
        if (x > -half_pixel && x <= half_pixel)
        {
            return 1.0;
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class HammingFilter
{
public:
    __host__ __device__ HammingFilter()
        : _support(hamming_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        if (x < 0.0)
        {
            x = -x;
        }
        if (x == 0.0)
        {
            return 1.0;
        }
        if (x >= 1.0)
        {
            return 0.0;
        }
        x = x * M_PI;
        return sin(x) / x * (0.54f + 0.46f * cos(x));
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class BicubicFilter
{
public:
    __host__ __device__ BicubicFilter()
        : _support(bicubic_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        const float a = -0.5f;
        if (x < 0.0)
        {
            x = -x;
        }
        if (x < 1.0)
        {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        }
        if (x < 2.0)
        {
            return (((x - 5) * x + 8) * x - 4) * a;
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class LanczosFilter
{
public:
    __host__ __device__ LanczosFilter()
        : _support(lanczos_filter_support){};

    __host__ __device__ work_type _sincFilter(work_type x)
    {
        if (x == 0.0)
        {
            return 1.0;
        }
        x = x * M_PI;
        return sin(x) / x;
    }

    __host__ __device__ work_type filter(work_type x)
    {
        const float lanczos_a_param = 3.0;
        if (-lanczos_a_param <= x && x < lanczos_a_param)
        {
            return _sincFilter(x) * _sincFilter(x / lanczos_a_param);
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

} // namespace nvcv::legacy::cuda_op
