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

#include "Definitions.hpp"
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpInpaint.hpp>
#include <cvcuda/OpReformat.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace std;

constexpr uint8_t KNOWN  = 0; // known outside narrow band
constexpr uint8_t BAND   = 1; // narrow band (known)
constexpr uint8_t INSIDE = 2; // unknown
constexpr uint8_t CHANGE = 3; // service

namespace {

const nvcv::ImageFormat kFmt2U8Planar{NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X8)};

} // namespace

struct Point2f
{
    float x;
    float y;
};

inline static float VectorScalMult(const Point2f &v1, const Point2f &v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

inline static float VectorLength(const Point2f &v1)
{
    return v1.x * v1.x + v1.y * v1.y;
}

inline float min4(float a, float b, float c, float d)
{
    a = min(a, b);
    c = min(c, d);
    return min(a, c);
}

struct HeapElem
{
    float     T;
    int       i;
    int       j;
    HeapElem *prev;
    HeapElem *next;
};

class PriorityQueueFloat
{
private:
    PriorityQueueFloat(const PriorityQueueFloat &)            = delete;
    PriorityQueueFloat &operator=(const PriorityQueueFloat &) = delete;

    std::vector<HeapElem> storage;
    HeapElem             *mem   = nullptr;
    HeapElem             *empty = nullptr;
    HeapElem             *head  = nullptr;
    HeapElem             *tail  = nullptr;
    int                   num   = 0;
    int                   in    = 0;

public:
    bool Init(const vector<uint8_t> &f, int height, int width)
    {
        num = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (f[i * width + j] != 0)
                {
                    num++;
                }
            }
        }
        if (num <= 0)
        {
            storage.clear();
            mem = empty = head = tail = nullptr;
            return false;
        }
        storage.resize(num + 2);
        mem = storage.data();

        head    = mem;
        head->i = head->j = -1;
        head->prev        = nullptr;
        head->next        = mem + 1;
        head->T           = -FLT_MAX;
        empty             = mem + 1;
        int i             = 1;
        for (; i <= num; i++)
        {
            mem[i].prev = mem + i - 1;
            mem[i].next = mem + i + 1;
            mem[i].i    = -1;
            mem[i].T    = FLT_MAX;
        }
        tail    = mem + i;
        tail->i = tail->j = -1;
        tail->prev        = mem + i - 1;
        tail->next        = nullptr;
        tail->T           = FLT_MAX;
        return true;
    }

    bool Add(const vector<uint8_t> &f, int height, int width)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (f[i * width + j] == 0)
                    continue;
                if (!Push(i, j, 0))
                    return false;
            }
        }
        return true;
    }

    bool Push(int i, int j, float T)
    {
        HeapElem *tmp = empty;
        HeapElem *add = empty;
        if (empty == tail)
            return false;
        while (tmp->prev->T > T) tmp = tmp->prev;
        if (tmp != empty)
        {
            add->prev->next = add->next;
            add->next->prev = add->prev;
            empty           = add->next;
            add->prev       = tmp->prev;
            add->next       = tmp;
            add->prev->next = add;
            add->next->prev = add;
        }
        else
        {
            empty = empty->next;
        }
        add->i = i;
        add->j = j;
        add->T = T;
        in++;
        return true;
    }

    bool Pop(int *i, int *j)
    {
        HeapElem *tmp = head->next;
        if (empty == tmp)
            return false;
        *i              = tmp->i;
        *j              = tmp->j;
        tmp->prev->next = tmp->next;
        tmp->next->prev = tmp->prev;
        tmp->prev       = empty->prev;
        tmp->next       = empty;
        tmp->prev->next = tmp;
        tmp->next->prev = tmp;
        empty           = tmp;
        in--;
        return true;
    }

    bool Pop(int *i, int *j, float *T)
    {
        HeapElem *tmp = head->next;
        if (empty == tmp)
            return false;
        *i              = tmp->i;
        *j              = tmp->j;
        *T              = tmp->T;
        tmp->prev->next = tmp->next;
        tmp->next->prev = tmp->prev;
        tmp->prev       = empty->prev;
        tmp->next       = empty;
        tmp->prev->next = tmp;
        tmp->next->prev = tmp;
        empty           = tmp;
        in--;
        return true;
    }

    PriorityQueueFloat() = default;

    ~PriorityQueueFloat() = default;
};

struct IntPoint2D
{
    int y;
    int x;
};

struct FastMarchingData
{
    const vector<uint8_t> &f;
    const vector<float>   &t;
    const vector<uint8_t> &out;
    int                    height;
    int                    width;
};

struct InpaintTarget
{
    int     y;
    int     x;
    int     range;
    Point2f gradT;
};

struct InpaintAccumulator
{
    float ia = 0.0f;
    float jx = 0.0f;
    float jy = 0.0f;
    float s  = 1.0e-20f;
};

static int GridIndex(int y, int x, int width)
{
    return y * width + x;
}

static int OutputWidth(int width)
{
    return width - 2;
}

static int OutputIndex(int y, int x, int width)
{
    return y * OutputWidth(width) + x;
}

static int ScaleDimension(int value, double scale)
{
    return static_cast<int>(static_cast<double>(value) * scale);
}

static float PixelCount(int height, int width)
{
    return static_cast<float>(height) * static_cast<float>(width);
}

static int AbsDiff(uint8_t lhs, uint8_t rhs)
{
    return std::abs(static_cast<int>(lhs) - static_cast<int>(rhs));
}

static float PixelDiff(uint8_t lhs, uint8_t rhs)
{
    return static_cast<float>(static_cast<int>(lhs) - static_cast<int>(rhs));
}

static float SelectDifference(float center, float next, float prev, bool hasNext, bool hasPrev, float centeredScale)
{
    if (hasNext && hasPrev)
    {
        return (next - prev) * centeredScale;
    }
    if (hasNext)
    {
        return next - center;
    }
    if (hasPrev)
    {
        return center - prev;
    }
    return 0.0f;
}

static float FastMarching_solve(int i1, int j1, int i2, int j2, const vector<uint8_t> &f, const vector<float> &t,
                                int width)
{
    double a11 = t[GridIndex(i1, j1, width)];
    double a22 = t[GridIndex(i2, j2, width)];
    double m12 = min(a11, a22);

    bool firstKnown  = f[GridIndex(i1, j1, width)] != INSIDE;
    bool secondKnown = f[GridIndex(i2, j2, width)] != INSIDE;

    if (firstKnown && secondKnown)
    {
        double diff = a11 - a22;
        if (fabs(diff) >= 1.0)
        {
            return static_cast<float>(1 + m12);
        }
        return static_cast<float>((a11 + a22 + sqrt(2.0 - diff * diff)) * 0.5);
    }

    if (firstKnown)
    {
        return static_cast<float>(1 + a11);
    }
    if (secondKnown)
    {
        return static_cast<float>(1 + a22);
    }
    return static_cast<float>(1 + m12);
}

static Point2f FastMarchingGradientAt(int y, int x, const vector<uint8_t> &f, const vector<float> &t, int width)
{
    int center = GridIndex(y, x, width);

    return {SelectDifference(t[center], t[center + 1], t[center - 1], f[center + 1] != INSIDE, f[center - 1] != INSIDE,
                             0.5f),
            SelectDifference(t[center], t[center + width], t[center - width], f[center + width] != INSIDE,
                             f[center - width] != INSIDE, 0.5f)};
}

static float HorizontalImageGradient(const FastMarchingData &data, int y, int x, int outY, int leftX, int rightX)
{
    bool hasRight = data.f[GridIndex(y, x + 1, data.width)] != INSIDE;
    bool hasLeft  = data.f[GridIndex(y, x - 1, data.width)] != INSIDE;

    if (hasRight && hasLeft)
    {
        return PixelDiff(data.out[OutputIndex(outY, rightX + 1, data.width)],
                         data.out[OutputIndex(outY, leftX - 1, data.width)])
             * 2.0f;
    }
    if (hasRight)
    {
        return PixelDiff(data.out[OutputIndex(outY, rightX + 1, data.width)],
                         data.out[OutputIndex(outY, leftX, data.width)]);
    }
    if (hasLeft)
    {
        return PixelDiff(data.out[OutputIndex(outY, rightX, data.width)],
                         data.out[OutputIndex(outY, leftX - 1, data.width)]);
    }
    return 0.0f;
}

static float VerticalImageGradient(const FastMarchingData &data, int y, int x, int outX, int topY, int bottomY)
{
    bool hasBottom = data.f[GridIndex(y + 1, x, data.width)] != INSIDE;
    bool hasTop    = data.f[GridIndex(y - 1, x, data.width)] != INSIDE;

    if (hasBottom && hasTop)
    {
        return PixelDiff(data.out[OutputIndex(bottomY + 1, outX, data.width)],
                         data.out[OutputIndex(topY - 1, outX, data.width)])
             * 2.0f;
    }
    if (hasBottom)
    {
        return PixelDiff(data.out[OutputIndex(bottomY + 1, outX, data.width)],
                         data.out[OutputIndex(topY, outX, data.width)]);
    }
    if (hasTop)
    {
        return PixelDiff(data.out[OutputIndex(bottomY, outX, data.width)],
                         data.out[OutputIndex(topY - 1, outX, data.width)]);
    }
    return 0.0f;
}

static bool IsInteriorPoint(int y, int x, int height, int width)
{
    return y > 0 && x > 0 && y < height - 1 && x < width - 1;
}

static bool IsUsableSourcePixel(const FastMarchingData &data, const InpaintTarget &target, int y, int x)
{
    if (!IsInteriorPoint(y, x, data.height, data.width))
    {
        return false;
    }

    int dy = y - target.y;
    int dx = x - target.x;
    return data.f[GridIndex(y, x, data.width)] != INSIDE && dx * dx + dy * dy <= target.range * target.range;
}

static void AccumulateSourcePixel(const FastMarchingData &data, const InpaintTarget &target, int y, int x,
                                  InpaintAccumulator &acc)
{
    if (!IsUsableSourcePixel(data, target, y, x))
    {
        return;
    }

    int outYBottom = y - 1 - (y == data.height - 2 ? 1 : 0);
    int outYTop    = y - 1 + (y == 1 ? 1 : 0);
    int outXLeft   = x - 1 + (x == 1 ? 1 : 0);
    int outXRight  = x - 1 - (x == data.width - 2 ? 1 : 0);

    Point2f r{static_cast<float>(target.x - x), static_cast<float>(target.y - y)};

    float vectorLength = VectorLength(r);
    auto  dst          = static_cast<float>(1.0 / (vectorLength * sqrt(vectorLength)));
    auto  lev          = static_cast<float>(
        1.0 / (1 + fabs(data.t[GridIndex(y, x, data.width)] - data.t[GridIndex(target.y, target.x, data.width)])));
    float dir = VectorScalMult(r, target.gradT);
    if (fabs(dir) <= 0.01)
    {
        dir = 0.000001f;
    }

    auto    weight = std::fabs(dst * lev * dir);
    Point2f gradI{HorizontalImageGradient(data, y, x, outYTop, outXLeft, outXRight),
                  VerticalImageGradient(data, y, x, outXLeft, outYTop, outYBottom)};
    int     outIndex = OutputIndex(outYTop, outXLeft, data.width);

    acc.ia += weight * static_cast<float>(data.out[outIndex]);
    acc.jx -= weight * gradI.x * r.x;
    acc.jy -= weight * gradI.y * r.y;
    acc.s += weight;
}

static uint8_t SaturateToU8(float value)
{
    auto rounded = static_cast<int>(std::round(value));
    if (rounded <= 0)
    {
        return 0;
    }
    if (rounded >= UCHAR_MAX)
    {
        return UCHAR_MAX;
    }
    return static_cast<uint8_t>(rounded);
}

static void InpaintTargetPixel(const FastMarchingData &data, const InpaintTarget &target, vector<uint8_t> &out)
{
    InpaintAccumulator acc;

    for (int y = target.y - target.range; y <= target.y + target.range; y++)
    {
        for (int x = target.x - target.range; x <= target.x + target.range; x++)
        {
            AccumulateSourcePixel(data, target, y, x, acc);
        }
    }

    auto sat = acc.ia / acc.s + (acc.jx + acc.jy) / (sqrt(acc.jx * acc.jx + acc.jy * acc.jy) + 1.0e-20f) + 0.5f;
    out[OutputIndex(target.y - 1, target.x - 1, data.width)] = SaturateToU8(sat);
}

static float NeighborDistance(int y, int x, const vector<uint8_t> &f, const vector<float> &t, int width)
{
    return min4(
        FastMarching_solve(y - 1, x, y, x - 1, f, t, width), FastMarching_solve(y + 1, x, y, x - 1, f, t, width),
        FastMarching_solve(y - 1, x, y, x + 1, f, t, width), FastMarching_solve(y + 1, x, y, x + 1, f, t, width));
}

static void InpaintFMM(vector<uint8_t> &f, vector<float> &t, vector<uint8_t> &out, int range,
                       shared_ptr<PriorityQueueFloat> Heap, int height, int width)
{
    constexpr array<IntPoint2D, 4> kNeighbors{
        {{-1, 0}, {0, -1}, {1, 0}, {0, 1}}
    };

    int ii = 0;
    int jj = 0;
    while (Heap->Pop(&ii, &jj))
    {
        f[GridIndex(ii, jj, width)] = KNOWN;
        for (const IntPoint2D &offset : kNeighbors)
        {
            int i = ii + offset.y;
            int j = jj + offset.x;
            if ((i <= 0) || (j <= 0) || (i > height - 1) || (j > width - 1))
            {
                continue;
            }

            if (f[GridIndex(i, j, width)] != INSIDE)
            {
                continue;
            }

            float dist                = NeighborDistance(i, j, f, t, width);
            t[GridIndex(i, j, width)] = dist;

            FastMarchingData data{f, t, out, height, width};
            InpaintTarget    target{i, j, range, FastMarchingGradientAt(i, j, f, t, width)};
            InpaintTargetPixel(data, target, out);

            f[GridIndex(i, j, width)] = BAND;
            Heap->Push(i, j, dist);
        }
    }
}

template<typename T>
static void CopyMaskWithBorder(const vector<T> &orgMask, vector<uint8_t> &mask, int height, int width, int ecols)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (!orgMask[GridIndex(i, j, width)])
            {
                continue;
            }
            mask[GridIndex(i + 1, j + 1, ecols)] = INSIDE;
        }
    }
}

static void SetKnownBorder(vector<uint8_t> &mask, int erows, int ecols)
{
    for (int i = 0; i < ecols; i++)
    {
        mask[i]                              = KNOWN;
        mask[GridIndex(erows - 1, i, ecols)] = KNOWN;
    }
    for (int i = 0; i < erows; i++)
    {
        mask[GridIndex(i, 0, ecols)]         = KNOWN;
        mask[GridIndex(i, ecols - 1, ecols)] = KNOWN;
    }
}

static uint8_t DilatedMaskValue(const vector<uint8_t> &mask, const array<uint8_t, 9> &kernel, int y, int x, int ecols)
{
    int     kernelIndex = 0;
    uint8_t result      = 0;
    for (int dy = 0; dy < 3; dy++)
    {
        for (int dx = 0; dx < 3; dx++)
        {
            if (kernel[kernelIndex] == 0)
            {
                kernelIndex++;
                continue;
            }
            result = max(result, mask[GridIndex(y - 1 + dy, x - 1 + dx, ecols)]);
            kernelIndex++;
        }
    }
    return result;
}

static void DilateMask(const vector<uint8_t> &mask, vector<uint8_t> &band, int erows, int ecols)
{
    constexpr array<uint8_t, 9> kKernel{0, 1, 0, 1, 1, 1, 0, 1, 0};

    for (int i = 1; i < erows - 1; i++)
    {
        for (int j = 1; j < ecols - 1; j++)
        {
            band[GridIndex(i, j, ecols)] = DilatedMaskValue(mask, kKernel, i, j, ecols);
        }
    }
}

static void SubtractMaskFromBand(vector<uint8_t> &band, const vector<uint8_t> &mask, int erows, int ecols)
{
    for (int i = 1; i < erows - 1; i++)
    {
        for (int j = 1; j < ecols - 1; j++)
        {
            band[GridIndex(i, j, ecols)] -= mask[GridIndex(i, j, ecols)];
        }
    }
}

static void ApplyBandAndMask(vector<uint8_t> &f, vector<float> &t, const vector<uint8_t> &band,
                             const vector<uint8_t> &mask, int erows, int ecols)
{
    for (int i = 0; i < erows; i++)
    {
        for (int j = 0; j < ecols; j++)
        {
            int index = GridIndex(i, j, ecols);
            if (band[index])
            {
                f[index] = BAND;
                t[index] = 0;
            }
            if (mask[index])
            {
                f[index] = INSIDE;
            }
        }
    }
}

//test FMT_U8
template<typename T>
void Inpaint(std::vector<T> &src, std::vector<T> &dst, std::vector<T> &org_mask, double radius, int height, int width)
{
    auto range = static_cast<int>(std::round(radius));
    range      = std::max(range, 1);
    range      = std::min(range, 100);

    int             erows = height + 2;
    int             ecols = width + 2;
    vector<uint8_t> f(erows * ecols, KNOWN);
    vector<float>   t(erows * ecols, 1.0e6f);
    vector<uint8_t> band(erows * ecols, KNOWN);
    vector<uint8_t> mask(erows * ecols, KNOWN);

    dst.assign(src.begin(), src.end());
    CopyMaskWithBorder(org_mask, mask, height, width, ecols);
    SetKnownBorder(mask, erows, ecols);
    DilateMask(mask, band, erows, ecols);

    auto heap = make_shared<PriorityQueueFloat>();
    if (!heap->Init(band, erows, ecols))
        return;

    SubtractMaskFromBand(band, mask, erows, ecols);
    if (!heap->Add(band, erows, ecols))
        return;

    ApplyBandAndMask(f, t, band, mask, erows, ecols);

    InpaintFMM(f, t, dst, range, heap, erows, ecols);
}

namespace {

enum class StrictInpaintRoute
{
    NHWC,
    NCHW,
    NCHW_FAKE,
    HWC,
    CHW,
};

const char *StrictRouteName(StrictInpaintRoute route)
{
    switch (route)
    {
    case StrictInpaintRoute::NHWC:
        return "NHWC";
    case StrictInpaintRoute::NCHW:
        return "NCHW";
    case StrictInpaintRoute::NCHW_FAKE:
        return "NCHW_FAKE";
    case StrictInpaintRoute::HWC:
        return "HWC";
    case StrictInpaintRoute::CHW:
        return "CHW";
    }
    return "unknown";
}

bool StrictRouteIsPlanar(StrictInpaintRoute route)
{
    return route == StrictInpaintRoute::NCHW || route == StrictInpaintRoute::NCHW_FAKE
        || route == StrictInpaintRoute::CHW;
}

bool StrictRouteIsRank3(StrictInpaintRoute route)
{
    return route == StrictInpaintRoute::HWC || route == StrictInpaintRoute::CHW;
}

size_t StrictOutputIndex(int y, int x, int channel, int height, int width, int channels, bool planar)
{
    if (planar)
    {
        return (static_cast<size_t>(channel) * height + y) * width + x;
    }
    return (static_cast<size_t>(y) * width + x) * channels + channel;
}

template<typename T>
struct StrictFastMarchingData
{
    const vector<uint8_t> &f;
    const vector<float>   &t;
    const vector<T>       &out;
    int                    paddedHeight;
    int                    paddedWidth;
    int                    channels;
    bool                   planar;
};

template<typename T>
float StrictPixelDifference(T lhs, T rhs)
{
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

template<typename T>
float StrictHorizontalImageGradient(const StrictFastMarchingData<T> &data, int y, int x, int outY, int leftX,
                                    int rightX, int channel)
{
    const bool hasRight = data.f[GridIndex(y, x + 1, data.paddedWidth)] != INSIDE;
    const bool hasLeft  = data.f[GridIndex(y, x - 1, data.paddedWidth)] != INSIDE;
    const int  height   = data.paddedHeight - 2;
    const int  width    = data.paddedWidth - 2;

    if (hasRight && hasLeft)
    {
        return StrictPixelDifference(
                   data.out[StrictOutputIndex(outY, rightX + 1, channel, height, width, data.channels, data.planar)],
                   data.out[StrictOutputIndex(outY, leftX - 1, channel, height, width, data.channels, data.planar)])
             * 2.0f;
    }
    if (hasRight)
    {
        return StrictPixelDifference(
            data.out[StrictOutputIndex(outY, rightX + 1, channel, height, width, data.channels, data.planar)],
            data.out[StrictOutputIndex(outY, leftX, channel, height, width, data.channels, data.planar)]);
    }
    if (hasLeft)
    {
        return StrictPixelDifference(
            data.out[StrictOutputIndex(outY, rightX, channel, height, width, data.channels, data.planar)],
            data.out[StrictOutputIndex(outY, leftX - 1, channel, height, width, data.channels, data.planar)]);
    }
    return 0.0f;
}

template<typename T>
float StrictVerticalImageGradient(const StrictFastMarchingData<T> &data, int y, int x, int outX, int topY, int bottomY,
                                  int channel)
{
    const bool hasBottom = data.f[GridIndex(y + 1, x, data.paddedWidth)] != INSIDE;
    const bool hasTop    = data.f[GridIndex(y - 1, x, data.paddedWidth)] != INSIDE;
    const int  height    = data.paddedHeight - 2;
    const int  width     = data.paddedWidth - 2;

    if (hasBottom && hasTop)
    {
        return StrictPixelDifference(
                   data.out[StrictOutputIndex(bottomY + 1, outX, channel, height, width, data.channels, data.planar)],
                   data.out[StrictOutputIndex(topY - 1, outX, channel, height, width, data.channels, data.planar)])
             * 2.0f;
    }
    if (hasBottom)
    {
        return StrictPixelDifference(
            data.out[StrictOutputIndex(bottomY + 1, outX, channel, height, width, data.channels, data.planar)],
            data.out[StrictOutputIndex(topY, outX, channel, height, width, data.channels, data.planar)]);
    }
    if (hasTop)
    {
        return StrictPixelDifference(
            data.out[StrictOutputIndex(bottomY, outX, channel, height, width, data.channels, data.planar)],
            data.out[StrictOutputIndex(topY - 1, outX, channel, height, width, data.channels, data.planar)]);
    }
    return 0.0f;
}

template<typename T>
bool StrictIsUsableSourcePixel(const StrictFastMarchingData<T> &data, const InpaintTarget &target, int y, int x)
{
    if (!IsInteriorPoint(y, x, data.paddedHeight, data.paddedWidth) || (y == target.y && x == target.x))
    {
        return false;
    }

    const int dy = y - target.y;
    const int dx = x - target.x;
    return data.f[GridIndex(y, x, data.paddedWidth)] != INSIDE && dx * dx + dy * dy <= target.range * target.range;
}

template<typename T>
void StrictAccumulateSourcePixel(const StrictFastMarchingData<T> &data, const InpaintTarget &target, int y, int x,
                                 int channel, InpaintAccumulator &acc)
{
    if (!StrictIsUsableSourcePixel(data, target, y, x))
    {
        return;
    }

    const int outYBottom = y - 1 - (y == data.paddedHeight - 2 ? 1 : 0);
    const int outYTop    = y - 1 + (y == 1 ? 1 : 0);
    const int outXLeft   = x - 1 + (x == 1 ? 1 : 0);
    const int outXRight  = x - 1 - (x == data.paddedWidth - 2 ? 1 : 0);

    const Point2f r{static_cast<float>(target.x - x), static_cast<float>(target.y - y)};

    const float vectorLength  = VectorLength(r);
    const auto distanceWeight = static_cast<float>(1.0 / (vectorLength * std::sqrt(static_cast<double>(vectorLength))));
    const auto levelWeight    = static_cast<float>(
        1.0
        / (1.0
           + std::fabs(static_cast<double>(data.t[GridIndex(y, x, data.paddedWidth)]
                                           - data.t[GridIndex(target.y, target.x, data.paddedWidth)]))));
    float direction = VectorScalMult(r, target.gradT);
    if (std::fabs(direction) <= 0.01f)
    {
        direction = 0.000001f;
    }

    const float   weight = std::fabs(distanceWeight * levelWeight * direction);
    const Point2f gradI{
        StrictHorizontalImageGradient(data, y, x, outYTop, outXLeft, outXRight, channel),
        StrictVerticalImageGradient(data, y, x, outXLeft, outYTop, outYBottom, channel),
    };
    const int height = data.paddedHeight - 2;
    const int width  = data.paddedWidth - 2;
    const T source = data.out[StrictOutputIndex(outYTop, outXLeft, channel, height, width, data.channels, data.planar)];

    acc.ia += weight * static_cast<float>(source);
    acc.jx -= weight * gradI.x * r.x;
    acc.jy -= weight * gradI.y * r.y;
    acc.s += weight;
}

// CUDA's cvt.rni.sat.u8.f32 is round-to-nearest-even. Keep this host oracle independent from
// SaturateCast so a change to the production conversion cannot change the expected result too.
uint8_t StrictSaturateToU8(float value)
{
    if (std::isnan(value) || value <= 0.0f)
    {
        return 0;
    }
    if (value >= 255.0f)
    {
        return 255;
    }

    float rounded = std::floor(value);
    if (const float fraction = value - rounded;
        fraction > 0.5f || (fraction == 0.5f && (static_cast<int>(rounded) & 1) != 0))
    {
        rounded += 1.0f;
    }
    return static_cast<uint8_t>(rounded);
}

template<typename T>
void StrictInpaintTargetPixel(const StrictFastMarchingData<T> &data, const InpaintTarget &target, vector<T> &out)
{
    const int height = data.paddedHeight - 2;
    const int width  = data.paddedWidth - 2;

    for (int channel = 0; channel < data.channels; channel++)
    {
        InpaintAccumulator acc;
        for (int y = target.y - target.range; y <= target.y + target.range; y++)
        {
            for (int x = target.x - target.range; x <= target.x + target.range; x++)
            {
                StrictAccumulateSourcePixel(data, target, y, x, channel, acc);
            }
        }

        const double correctionDenominator
            = std::sqrt(static_cast<double>(acc.jx * acc.jx + acc.jy * acc.jy)) + 1.0e-20;
        const auto value = static_cast<float>(acc.ia / acc.s + (acc.jx + acc.jy) / correctionDenominator + 0.5);
        out[StrictOutputIndex(target.y - 1, target.x - 1, channel, height, width, data.channels, data.planar)]
            = static_cast<T>(StrictSaturateToU8(value));
    }
}

template<typename T>
void StrictInpaintFMM(vector<uint8_t> &f, vector<float> &t, vector<T> &out, int range,
                      const shared_ptr<PriorityQueueFloat> &heap, int paddedHeight, int paddedWidth, int channels,
                      bool planar)
{
    constexpr array<IntPoint2D, 4> kNeighbors{
        {{-1, 0}, {0, -1}, {1, 0}, {0, 1}}
    };

    int y = 0;
    int x = 0;
    while (heap->Pop(&y, &x))
    {
        f[GridIndex(y, x, paddedWidth)] = KNOWN;
        for (const IntPoint2D &offset : kNeighbors)
        {
            const int targetY = y + offset.y;
            const int targetX = x + offset.x;
            if (targetY <= 0 || targetX <= 0 || targetY >= paddedHeight - 1 || targetX >= paddedWidth - 1
                || f[GridIndex(targetY, targetX, paddedWidth)] != INSIDE)
            {
                continue;
            }

            const float distance                        = NeighborDistance(targetY, targetX, f, t, paddedWidth);
            t[GridIndex(targetY, targetX, paddedWidth)] = distance;

            const StrictFastMarchingData<T> data{f, t, out, paddedHeight, paddedWidth, channels, planar};
            const InpaintTarget             target{targetY, targetX, range,
                                       FastMarchingGradientAt(targetY, targetX, f, t, paddedWidth)};
            StrictInpaintTargetPixel(data, target, out);

            f[GridIndex(targetY, targetX, paddedWidth)] = BAND;
            heap->Push(targetY, targetX, distance);
        }
    }
}

template<typename T>
vector<T> StrictInpaintReference(const vector<T> &src, const vector<uint8_t> &originalMask, double radius, int height,
                                 int width, int channels, bool planar)
{
    const int range        = std::clamp(static_cast<int>(std::round(radius)), 1, 100);
    const int paddedHeight = height + 2;
    const int paddedWidth  = width + 2;

    vector<uint8_t> f(paddedHeight * paddedWidth, KNOWN);
    vector<float>   t(paddedHeight * paddedWidth, 1.0e6f);
    vector<uint8_t> band(paddedHeight * paddedWidth, KNOWN);
    vector<uint8_t> mask(paddedHeight * paddedWidth, KNOWN);
    vector<T>       out = src;

    vector<uint8_t> originalMaskCopy = originalMask;
    CopyMaskWithBorder(originalMaskCopy, mask, height, width, paddedWidth);
    SetKnownBorder(mask, paddedHeight, paddedWidth);
    DilateMask(mask, band, paddedHeight, paddedWidth);

    auto heap = make_shared<PriorityQueueFloat>();
    if (!heap->Init(band, paddedHeight, paddedWidth))
    {
        return out;
    }

    SubtractMaskFromBand(band, mask, paddedHeight, paddedWidth);
    if (!heap->Add(band, paddedHeight, paddedWidth))
    {
        return out;
    }

    ApplyBandAndMask(f, t, band, mask, paddedHeight, paddedWidth);
    StrictInpaintFMM(f, t, out, range, heap, paddedHeight, paddedWidth, channels, planar);
    return out;
}

template<typename T>
T StrictInputValue(int sample, int y, int x, int channel)
{
    const int base = (sample * 61 + y * 29 + x * 17 + channel * 43 + (x * y) % 19) % 241;
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        return static_cast<uint8_t>(base + 7);
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        return base - 73;
    }
    else
    {
        static_assert(std::is_same_v<T, float>);
        return static_cast<float>(base - 61) + static_cast<float>((x + 2 * y + 3 * channel + sample) % 8) * 0.125f;
    }
}

template<typename T>
vector<T> MakeStrictInput(int sample, int height, int width, int channels, bool planar)
{
    vector<T> input(static_cast<size_t>(height) * width * channels);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int channel = 0; channel < channels; channel++)
            {
                input[StrictOutputIndex(y, x, channel, height, width, channels, planar)]
                    = StrictInputValue<T>(sample, y, x, channel);
            }
        }
    }
    return input;
}

vector<uint8_t> MakeStrictMask(int sample, int height, int width)
{
    vector<uint8_t> mask(static_cast<size_t>(height) * width, 0);

    // Isolated holes make the expected result independent of the order in which parallel band
    // threads reach a target, while the second hole exercises the bottom/right launch tail.
    const array<IntPoint2D, 2> holes = sample % 2 == 0
        ? array<IntPoint2D, 2>{IntPoint2D{4, 3}, IntPoint2D{height - 4, width - 4}}
        : array<IntPoint2D, 2>{IntPoint2D{3, width - 5}, IntPoint2D{height - 4, 4}};
    mask[GridIndex(holes[0].y, holes[0].x, width)] = 1;
    mask[GridIndex(holes[1].y, holes[1].x, width)] = 255;
    return mask;
}

nvcv::Tensor MakeStrictImageTensor(StrictInpaintRoute route, int batch, int height, int width, int channels,
                                   nvcv::DataType dtype)
{
    if (route == StrictInpaintRoute::HWC)
    {
        return nvcv::Tensor(
            {
                {height, width, channels},
                "HWC"
        },
            dtype);
    }
    if (route == StrictInpaintRoute::CHW)
    {
        return nvcv::Tensor(
            {
                {channels, height, width},
                "CHW"
        },
            dtype);
    }
    if (StrictRouteIsPlanar(route))
    {
        return nvcv::Tensor(
            {
                {batch, channels, height, width},
                "NCHW"
        },
            dtype);
    }
    return nvcv::Tensor(
        {
            {batch, height, width, channels},
            "NHWC"
    },
        dtype);
}

nvcv::Tensor MakeStrictMaskTensor(StrictInpaintRoute route, int batch, int height, int width)
{
    // The public contract allows only HWC/NHWC masks, including when input/output are planar.
    if (StrictRouteIsRank3(route))
    {
        return nvcv::Tensor(
            {
                {height, width, 1},
                "HWC"
        },
            nvcv::TYPE_U8);
    }
    return nvcv::Tensor(
        {
            {batch, height, width, 1},
            "NHWC"
    },
        nvcv::TYPE_U8);
}

template<typename T>
void RunStrictInpaintTensorCase(nvcv::DataType dtype, int channels, StrictInpaintRoute route, int batch, double radius)
{
    constexpr int kHeight = 17;
    constexpr int kWidth  = 19;

    ASSERT_FALSE(StrictRouteIsRank3(route) && batch != 1);
    const bool planar = StrictRouteIsPlanar(route);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src  = MakeStrictImageTensor(route, batch, kHeight, kWidth, channels, dtype);
    nvcv::Tensor dst  = MakeStrictImageTensor(route, batch, kHeight, kWidth, channels, dtype);
    nvcv::Tensor mask = MakeStrictMaskTensor(route, batch, kHeight, kWidth);

    auto srcData  = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData  = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto maskData = mask.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData && maskData);

    vector<vector<T>>       source(batch);
    vector<vector<T>>       expected(batch);
    vector<vector<uint8_t>> masks(batch);
    for (int sample = 0; sample < batch; sample++)
    {
        source[sample] = MakeStrictInput<T>(sample, kHeight, kWidth, channels, planar);
        masks[sample]  = MakeStrictMask(sample, kHeight, kWidth);
        expected[sample]
            = StrictInpaintReference(source[sample], masks[sample], radius, kHeight, kWidth, channels, planar);
        nvcv::util::SetImageTensorFromVector<T>(*srcData, source[sample], sample);
        nvcv::util::SetImageTensorFromVector<uint8_t>(*maskData, masks[sample], sample);
    }

    cvcuda::Inpaint op(batch, nvcv::Size2D{kWidth, kHeight});
    if (route == StrictInpaintRoute::NCHW_FAKE)
    {
        nvcv::Tensor interSrc(
            {
                {batch, kHeight, kWidth, channels},
                "NHWC"
        },
            dtype);
        nvcv::Tensor interDst(
            {
                {batch, kHeight, kWidth, channels},
                "NHWC"
        },
            dtype);
        cvcuda::Reformat reformat;
        EXPECT_NO_THROW(reformat(stream, src, interSrc));
        EXPECT_NO_THROW(op(stream, interSrc, mask, interDst, radius));
        EXPECT_NO_THROW(reformat(stream, interDst, dst));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }
    else
    {
        EXPECT_NO_THROW(op(stream, src, mask, dst, radius));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    for (int sample = 0; sample < batch; sample++)
    {
        SCOPED_TRACE(::testing::Message() << "route=" << StrictRouteName(route) << " sample=" << sample);
        vector<T> actual;
        nvcv::util::GetImageVectorFromTensor<T>(*dstData, sample, actual);
        EXPECT_EQ(expected[sample], actual);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpInpaintStrictTensor,
                  nvcv::test::ValueList<nvcv::DataType, int, StrictInpaintRoute, int, double>
{
    // dtype, channels, route, batch, radius
    {nvcv::TYPE_U8,  1, StrictInpaintRoute::NHWC,      1, 2.0},
    {nvcv::TYPE_U8,  2, StrictInpaintRoute::NHWC,      1, 5.0},
    {nvcv::TYPE_U8,  3, StrictInpaintRoute::NHWC,      2, 2.0},
    {nvcv::TYPE_U8,  4, StrictInpaintRoute::NHWC,      1, 5.0},
    {nvcv::TYPE_S32, 1, StrictInpaintRoute::NHWC,      1, 5.0},
    {nvcv::TYPE_S32, 2, StrictInpaintRoute::NHWC,      1, 2.0},
    {nvcv::TYPE_S32, 3, StrictInpaintRoute::NHWC,      1, 5.0},
    {nvcv::TYPE_S32, 4, StrictInpaintRoute::NHWC,      1, 2.0},
    {nvcv::TYPE_F32, 1, StrictInpaintRoute::NHWC,      1, 2.0},
    {nvcv::TYPE_F32, 2, StrictInpaintRoute::NHWC,      1, 5.0},
    {nvcv::TYPE_F32, 3, StrictInpaintRoute::NHWC,      1, 2.0},
    {nvcv::TYPE_F32, 4, StrictInpaintRoute::NHWC,      1, 5.0},

    {nvcv::TYPE_U8,  1, StrictInpaintRoute::NCHW,      1, 5.0},
    {nvcv::TYPE_U8,  3, StrictInpaintRoute::NCHW,      1, 2.0},
    {nvcv::TYPE_U8,  4, StrictInpaintRoute::NCHW,      1, 5.0},
    {nvcv::TYPE_S32, 1, StrictInpaintRoute::NCHW,      1, 2.0},
    {nvcv::TYPE_S32, 3, StrictInpaintRoute::NCHW,      1, 5.0},
    {nvcv::TYPE_S32, 4, StrictInpaintRoute::NCHW,      1, 2.0},
    {nvcv::TYPE_F32, 1, StrictInpaintRoute::NCHW,      1, 5.0},
    {nvcv::TYPE_F32, 3, StrictInpaintRoute::NCHW,      1, 2.0},
    {nvcv::TYPE_F32, 4, StrictInpaintRoute::NCHW,      2, 5.0},

    {nvcv::TYPE_U8,  1, StrictInpaintRoute::NCHW_FAKE, 1, 2.0},
    {nvcv::TYPE_U8,  2, StrictInpaintRoute::NCHW_FAKE, 1, 5.0},
    {nvcv::TYPE_U8,  3, StrictInpaintRoute::NCHW_FAKE, 1, 2.0},
    {nvcv::TYPE_U8,  4, StrictInpaintRoute::NCHW_FAKE, 1, 5.0},
    {nvcv::TYPE_S32, 1, StrictInpaintRoute::NCHW_FAKE, 1, 5.0},
    {nvcv::TYPE_S32, 2, StrictInpaintRoute::NCHW_FAKE, 2, 2.0},
    {nvcv::TYPE_S32, 3, StrictInpaintRoute::NCHW_FAKE, 1, 5.0},
    {nvcv::TYPE_S32, 4, StrictInpaintRoute::NCHW_FAKE, 1, 2.0},
    {nvcv::TYPE_F32, 1, StrictInpaintRoute::NCHW_FAKE, 1, 2.0},
    {nvcv::TYPE_F32, 2, StrictInpaintRoute::NCHW_FAKE, 1, 5.0},
    {nvcv::TYPE_F32, 3, StrictInpaintRoute::NCHW_FAKE, 1, 2.0},
    {nvcv::TYPE_F32, 4, StrictInpaintRoute::NCHW_FAKE, 1, 5.0},

    {nvcv::TYPE_U8,  3, StrictInpaintRoute::HWC,       1, 2.0},
    {nvcv::TYPE_F32, 4, StrictInpaintRoute::CHW,       1, 5.0},
});

// clang-format on

TEST_P(OpInpaintStrictTensor, matches_independent_cpu_gold)
{
    const nvcv::DataType     dtype    = GetParamValue<0>();
    const int                channels = GetParamValue<1>();
    const StrictInpaintRoute route    = GetParamValue<2>();
    const int                batch    = GetParamValue<3>();
    const double             radius   = GetParamValue<4>();

    if (dtype == nvcv::TYPE_U8)
    {
        RunStrictInpaintTensorCase<uint8_t>(dtype, channels, route, batch, radius);
    }
    else if (dtype == nvcv::TYPE_S32)
    {
        RunStrictInpaintTensorCase<int32_t>(dtype, channels, route, batch, radius);
    }
    else if (dtype == nvcv::TYPE_F32)
    {
        RunStrictInpaintTensorCase<float>(dtype, channels, route, batch, radius);
    }
    else
    {
        FAIL() << "Unexpected strict Inpaint dtype";
    }
}

} // namespace

static void ExpectInpaintOutputClose(const std::vector<uint8_t> &testVec, const std::vector<uint8_t> &goldVec,
                                     int height, int width)
{
    int   count   = 0;
    float diffsum = 0.0f;
    for (size_t idx = 0; idx < testVec.size(); idx++)
    {
        int diff = AbsDiff(testVec[idx], goldVec[idx]);
        if (diff > 1)
        {
            count++;
        }
        diffsum += static_cast<float>(diff);
    }

    float pixels = PixelCount(height, width);
    EXPECT_LE(static_cast<float>(count) / pixels, 5e-2f);

    diffsum /= 255.0f;
    diffsum /= pixels;
    EXPECT_LE(diffsum, 5e-3f);
}

// clang-format off
NVCV_TEST_SUITE_P(OpInpaint, nvcv::test::ValueList<int, int, int, double>
{
    //batch,    height,     width,      radius
    {     1,       480,       360,       5.0},
    {     4,       100,       101,       5.0},
    {     3,       360,       480,       5.0},
});

// clang-format on

TEST_P(OpInpaint, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int    batch         = GetParamValue<0>();
    int    height        = GetParamValue<1>();
    int    width         = GetParamValue<2>();
    double inpaintRadius = GetParamValue<3>();

    nvcv::Tensor imgIn   = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgMask = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgOut  = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);

    auto inData = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);
    ASSERT_EQ(batch, inAccess->numSamples());

    auto maskData = imgMask.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, maskData);
    auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData);
    ASSERT_TRUE(maskAccess);
    ASSERT_EQ(batch, maskAccess->numSamples());

    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);
    ASSERT_EQ(batch, outAccess->numSamples());

    int64_t outSampleStride = outAccess->sampleStride();

    if (outData->rank() == 3)
    {
        outSampleStride = outAccess->numRows() * outAccess->rowStride();
    }

    int64_t outBufferSize = outSampleStride * outAccess->numSamples();

    // Set output buffer to dummy value
    EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));

    //Generate input and mask
    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::vector<std::vector<uint8_t>> maskVec(batch);
    std::default_random_engine        randEng;
    int                               rowStride = width * nvcv::FMT_U8.planePixelStrideBytes(0);

    for (int i = 0; i < batch; i++)
    {
        srcVec[i].assign(height * rowStride / sizeof(uint8_t), 255);
        int h  = height / 2;
        int w1 = ScaleDimension(width, 0.2);
        int w2 = ScaleDimension(width, 0.8);
        for (int hi = h - 10; hi < h + 10; hi++)
        {
            for (int wi = w1; wi <= w2; wi++)
            {
                srcVec[i][hi * width + wi] = 0;
            }
        }
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));

        maskVec[i].assign(height * rowStride / sizeof(uint8_t), 0);
        for (int hi = h - 10; hi < h + 10; hi++)
        {
            for (int wi = w1; wi <= w2; wi++)
            {
                maskVec[i][hi * width + wi] = 1;
            }
        }
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskAccess->sampleData(i), maskAccess->rowStride(), maskVec[i].data(),
                                            rowStride, rowStride, height, cudaMemcpyHostToDevice));
    }

    // Call operator
    int             maxBatch = 4;
    nvcv::Size2D    maxsize{480, 480};
    cvcuda::Inpaint InpainOp(maxBatch, maxsize);
    EXPECT_NO_THROW(InpainOp(stream, imgIn, imgMask, imgOut, inpaintRadius));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(height * rowStride / sizeof(uint8_t));
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(i), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * rowStride / sizeof(uint8_t));
        Inpaint<uint8_t>(srcVec[i], goldVec, maskVec[i], inpaintRadius, height, width);

        ExpectInpaintOutputClose(testVec, goldVec, height, width);
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpInpaint, varshape_correct_shape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int    batch         = GetParamValue<0>();
    int    height        = GetParamValue<1>();
    int    width         = GetParamValue<2>();
    double inpaintRadius = GetParamValue<3>();

    nvcv::ImageFormat fmt = nvcv::FMT_U8;

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndWidth(ScaleDimension(width, 0.8), ScaleDimension(width, 1.1));
    std::uniform_int_distribution rndHeight(ScaleDimension(height, 0.8), ScaleDimension(height, 1.1));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    std::vector<nvcv::Image> imgMask;
    for (int i = 0; i < batch; ++i)
    {
        int rw = rndWidth(randEng);
        int rh = rndHeight(randEng);
        imgSrc.emplace_back(nvcv::Size2D{rw, rh}, fmt);
        imgMask.emplace_back(nvcv::Size2D{rw, rh}, fmt);
        imgDst.emplace_back(nvcv::Size2D{rw, rh}, fmt);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchMask(batch);
    batchMask.pushBack(imgMask.begin(), imgMask.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    //Generate input
    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::vector<std::vector<uint8_t>> maskVec(batch);

    for (int i = 0; i < batch; i++)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        const auto maskData = imgMask[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(maskData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        srcVec[i].assign(srcHeight * srcRowStride / sizeof(uint8_t), 255);
        int h  = srcHeight / 2;
        int w1 = ScaleDimension(srcWidth, 0.2);
        int w2 = ScaleDimension(srcWidth, 0.8);
        for (int hi = h - 10; hi < h + 10; hi++)
        {
            for (int wi = w1; wi <= w2; wi++)
            {
                srcVec[i][hi * srcWidth + wi] = 0;
            }
        }

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));

        maskVec[i].assign(srcHeight * srcRowStride / sizeof(uint8_t), 0);
        for (int hi = h - 10; hi < h + 10; hi++)
        {
            for (int wi = w1; wi <= w2; wi++)
            {
                maskVec[i][hi * srcWidth + wi] = 1;
            }
        }

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskData->plane(0).basePtr, maskData->plane(0).rowStride, maskVec[i].data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));
    }

    // Call operator
    int             maxBatch = 4;
    nvcv::Size2D    maxsize{ScaleDimension(480, 1.1), ScaleDimension(480, 1.1)};
    cvcuda::Inpaint InpaintOp(maxBatch, maxsize);
    EXPECT_NO_THROW(InpaintOp(stream, batchSrc, batchMask, batchDst, inpaintRadius));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride / sizeof(uint8_t));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride / sizeof(uint8_t));
        Inpaint<uint8_t>(srcVec[i], goldVec, maskVec[i], inpaintRadius, dstHeight, dstWidth);

        ExpectInpaintOutputClose(testVec, goldVec, dstHeight, dstWidth);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint, test_grad_corner_condition)
{
    int    batch         = 1;
    int    height        = 5;
    int    width         = 5;
    double inpaintRadius = 2.0;

    std::vector<std::vector<uint8_t>> maskVecCases;
    // case 1:
    // mask: 0 0 0 0 0
    //      0 1 1 1 0
    //      0 1 0 1 0
    //      0 1 1 1 0
    //      0 0 0 0 0
    {
        std::vector<uint8_t> maskVec(height * width, 0);

        maskVec[1 * width + 1] = 1;
        maskVec[1 * width + 2] = 1;
        maskVec[1 * width + 3] = 1;
        maskVec[2 * width + 1] = 1;
        maskVec[2 * width + 3] = 1;
        maskVec[3 * width + 1] = 1;
        maskVec[3 * width + 2] = 1;
        maskVec[3 * width + 3] = 1;
        maskVecCases.emplace_back(maskVec);
    }
    // case 2
    // mask: 1 0 0 0 0
    //      0 0 0 0 0
    //      0 0 0 0 0
    //      0 0 0 0 0
    //      0 0 0 0 1
    {
        std::vector<uint8_t> maskVec(height * width, 0);
        maskVec[0 * width + 0] = 1;
        maskVec[0 * width + 4] = 1;
        maskVecCases.emplace_back(maskVec);
    }

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // run tensor op
    for (auto &maskVec : maskVecCases) // NOSONAR
    {
        nvcv::Tensor imgIn   = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
        nvcv::Tensor imgMask = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
        nvcv::Tensor imgOut  = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);

        auto inData   = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto maskData = imgMask.exportData<nvcv::TensorDataStridedCuda>();
        auto outData  = imgOut.exportData<nvcv::TensorDataStridedCuda>();

        auto inAccess   = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData);
        auto outAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);

        int rowStride = width * nvcv::FMT_U8.planePixelStrideBytes(0);

        std::vector<uint8_t> srcVec(height * width, 255);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(0), inAccess->rowStride(), srcVec.data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskAccess->sampleData(0), maskAccess->rowStride(), maskVec.data(),
                                            rowStride, rowStride, height, cudaMemcpyHostToDevice));

        int             maxBatch = 1;
        nvcv::Size2D    maxsize{width, height};
        cvcuda::Inpaint InpaintOp(maxBatch, maxsize);
        EXPECT_NO_THROW(InpaintOp(stream, imgIn, imgMask, imgOut, inpaintRadius));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        std::vector<uint8_t> testVec(height * rowStride / sizeof(uint8_t));
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(0), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * rowStride / sizeof(uint8_t));
        Inpaint<uint8_t>(srcVec, goldVec, maskVec, inpaintRadius, height, width);

        ExpectInpaintOutputClose(testVec, goldVec, height, width);
    }

    // run varshape op
    for (auto &maskVec : maskVecCases) // NOSONAR
    {
        auto fmt = nvcv::FMT_U8;

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;
        std::vector<nvcv::Image> imgMask;
        imgSrc.emplace_back(nvcv::Size2D{width, height}, fmt);
        imgMask.emplace_back(nvcv::Size2D{width, height}, fmt);
        imgDst.emplace_back(nvcv::Size2D{width, height}, fmt);

        nvcv::ImageBatchVarShape batchSrc(batch);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchMask(batch);
        batchMask.pushBack(imgMask.begin(), imgMask.end());

        nvcv::ImageBatchVarShape batchDst(batch);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        std::vector<uint8_t> srcVec(height * width, 255);
        const auto           srcData  = imgSrc[0].exportData<nvcv::ImageDataStridedCuda>();
        const auto           maskData = imgMask[0].exportData<nvcv::ImageDataStridedCuda>();

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec.data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskData->plane(0).basePtr, maskData->plane(0).rowStride, maskVec.data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));

        int             maxBatch = 4;
        nvcv::Size2D    maxsize{ScaleDimension(480, 1.1), ScaleDimension(480, 1.1)};
        cvcuda::Inpaint InpaintOp(maxBatch, maxsize);
        EXPECT_NO_THROW(InpaintOp(stream, batchSrc, batchMask, batchDst, inpaintRadius));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        const auto dstData = imgDst[0].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride / sizeof(uint8_t));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride / sizeof(uint8_t));
        Inpaint<uint8_t>(srcVec, goldVec, maskVec, inpaintRadius, dstHeight, dstWidth);

        ExpectInpaintOutputClose(testVec, goldVec, dstHeight, dstWidth);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint, tensor_uses_large_initial_fmm_distance)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int    batch         = 1;
    constexpr int    height        = 7;
    constexpr int    width         = 7;
    constexpr double inpaintRadius = 3.0;

    const std::array<uint8_t, height * width> srcVec{
        24,  65,  60,  53,  84,  117, 133, 56,  67,  95,  104, 136, 129, 165, 70,  97,  104,
        161, 136, 186, 180, 127, 155, 165, 153, 180, 184, 241, 128, 184, 179, 205, 227, 215,
        248, 182, 192, 228, 246, 236, 19,  31,  214, 236, 236, 240, 33,  15,  37,
    };
    const std::array<uint8_t, height * width> maskVec{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    nvcv::Tensor imgIn   = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgMask = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgOut  = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);

    auto inData   = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto maskData = imgMask.exportData<nvcv::TensorDataStridedCuda>();
    auto outData  = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    auto inAccess   = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData);
    auto outAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);

    int rowStride = width * nvcv::FMT_U8.planePixelStrideBytes(0);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(0), inAccess->rowStride(), srcVec.data(), rowStride,
                                        rowStride, height, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskAccess->sampleData(0), maskAccess->rowStride(), maskVec.data(), rowStride,
                                        rowStride, height, cudaMemcpyHostToDevice));

    cvcuda::Inpaint op(batch, nvcv::Size2D{width, height});
    EXPECT_NO_THROW(op(stream, imgIn, imgMask, imgOut, inpaintRadius));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> testVec(height * rowStride / sizeof(uint8_t));
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(0), outAccess->rowStride(),
                                        rowStride, height, cudaMemcpyDeviceToHost));

    // These pixels depend on FMM distances that keep the initial t value until a later expansion step.
    EXPECT_EQ(195, testVec[3 * width + 3]);
    EXPECT_EQ(201, testVec[4 * width + 3]);
    EXPECT_EQ(209, testVec[4 * width + 4]);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void FillInpaintStripe(std::vector<uint8_t> &src, std::vector<uint8_t> &mask, int width, int height,
                              int channels, int elemSize)
{
    const int pixelStride = channels * elemSize;

    src.assign(static_cast<size_t>(height) * width * pixelStride, 255);
    mask.assign(static_cast<size_t>(height) * width, 0);

    const int h  = height / 2;
    const int w1 = ScaleDimension(width, 0.2);
    const int w2 = ScaleDimension(width, 0.8);
    for (int wi = w1; wi <= w2; wi++)
    {
        auto pixel = src.begin() + static_cast<std::ptrdiff_t>((static_cast<size_t>(h) * width + wi) * pixelStride);
        std::fill(pixel, pixel + pixelStride, 0);
        mask[static_cast<size_t>(h) * width + wi] = 1;
    }
}

static void RunTensorInpaintPlanarParity(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width,
                                         int height, int numImages, double inpaintRadius)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels      = planarFmt.numChannels();
    const int elemSize      = planarFmt.planePixelStrideBytes(0);
    const int maskElemSize  = nvcv::FMT_U8.planePixelStrideBytes(0);
    const int rowStride     = width * interleavedFmt.planePixelStrideBytes(0);
    const int maskRowStride = width * maskElemSize;

    nvcv::Tensor srcI  = nvcv::util::CreateTensor(numImages, width, height, interleavedFmt);
    nvcv::Tensor dstI  = nvcv::util::CreateTensor(numImages, width, height, interleavedFmt);
    nvcv::Tensor srcP  = nvcv::util::CreateTensor(numImages, width, height, planarFmt);
    nvcv::Tensor dstP  = nvcv::util::CreateTensor(numImages, width, height, planarFmt);
    nvcv::Tensor maskI = nvcv::util::CreateTensor(numImages, width, height, nvcv::FMT_U8);
    nvcv::Tensor maskP(
        {
            {numImages, 1, height, width},
            "NCHW"
    },
        nvcv::TYPE_U8);

    auto srcIData  = srcI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData  = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPData  = srcP.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData  = dstP.exportData<nvcv::TensorDataStridedCuda>();
    auto maskIData = maskI.exportData<nvcv::TensorDataStridedCuda>();
    auto maskPData = maskP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcIData && dstIData && srcPData && dstPData && maskIData && maskPData);

    auto srcIAcc  = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcIData);
    auto dstIAcc  = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstIData);
    auto srcPAcc  = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPData);
    auto dstPAcc  = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPData);
    auto maskIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskIData);
    auto maskPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskPData);
    ASSERT_TRUE(srcIAcc && dstIAcc && srcPAcc && dstPAcc && maskIAcc && maskPAcc);

    for (int i = 0; i < numImages; i++)
    {
        std::vector<uint8_t> src;
        std::vector<uint8_t> mask;
        FillInpaintStripe(src, mask, width, height, channels, elemSize);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcIAcc->sampleData(i), srcIAcc->rowStride(), src.data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
        nvcv::test::planar::UploadPlanarSample(
            *srcPAcc, i, nvcv::test::planar::DeinterleaveToPlanes(src, width, height, channels, elemSize), width,
            height, channels, elemSize);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskIAcc->sampleData(i), maskIAcc->rowStride(), mask.data(), maskRowStride,
                                            maskRowStride, height, cudaMemcpyHostToDevice));
        nvcv::test::planar::UploadPlanarSample(*maskPAcc, i, mask, width, height, 1, maskElemSize);
    }

    cvcuda::Inpaint op(numImages, nvcv::Size2D{width, height});
    EXPECT_NO_THROW(op(stream, srcI, maskI, dstI, inpaintRadius));
    EXPECT_NO_THROW(op(stream, srcP, maskP, dstP, inpaintRadius));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; i++)
    {
        SCOPED_TRACE(i);

        auto gpuInter    = nvcv::test::planar::DownloadInterleavedSample(*dstIAcc, i, width, height, rowStride);
        auto planesOut   = nvcv::test::planar::DownloadPlanarSample(*dstPAcc, i, width, height, channels, elemSize);
        auto planarInter = nvcv::test::planar::InterleaveFromPlanes(planesOut, width, height, channels, elemSize);

        if (gpuInter != planarInter)
        {
            const auto [interIt, planarIt] = std::mismatch(gpuInter.begin(), gpuInter.end(), planarInter.begin());
            ADD_FAILURE() << "sample=" << i << " offset=" << std::distance(gpuInter.begin(), interIt)
                          << " interleaved=" << static_cast<int>(*interIt) << " planar=" << static_cast<int>(*planarIt);
        }
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunVarShapeInpaintPlanarParity(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width,
                                           int height, int numImages, double inpaintRadius)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels     = planarFmt.numChannels();
    const int elemSize     = planarFmt.planePixelStrideBytes(0);
    const int maskElemSize = nvcv::FMT_U8.planePixelStrideBytes(0);

    std::vector<nvcv::Image> srcI;
    std::vector<nvcv::Image> dstI;
    std::vector<nvcv::Image> maskI;
    std::vector<nvcv::Image> srcP;
    std::vector<nvcv::Image> dstP;
    std::vector<nvcv::Image> maskP;
    for (int i = 0; i < numImages; i++)
    {
        nvcv::Size2D size{width + i * 3, height + i * 2};
        srcI.emplace_back(size, interleavedFmt);
        dstI.emplace_back(size, interleavedFmt);
        maskI.emplace_back(size, nvcv::FMT_U8);
        srcP.emplace_back(size, planarFmt);
        dstP.emplace_back(size, planarFmt);
        maskP.emplace_back(size, nvcv::FMT_U8);
    }

    nvcv::ImageBatchVarShape batchSrcI(numImages);
    nvcv::ImageBatchVarShape batchDstI(numImages);
    nvcv::ImageBatchVarShape batchMaskI(numImages);
    nvcv::ImageBatchVarShape batchSrcP(numImages);
    nvcv::ImageBatchVarShape batchDstP(numImages);
    nvcv::ImageBatchVarShape batchMaskP(numImages);
    batchSrcI.pushBack(srcI.begin(), srcI.end());
    batchDstI.pushBack(dstI.begin(), dstI.end());
    batchMaskI.pushBack(maskI.begin(), maskI.end());
    batchSrcP.pushBack(srcP.begin(), srcP.end());
    batchDstP.pushBack(dstP.begin(), dstP.end());
    batchMaskP.pushBack(maskP.begin(), maskP.end());

    for (int i = 0; i < numImages; i++)
    {
        const int sampleW       = srcI[i].size().w;
        const int sampleH       = srcI[i].size().h;
        const int rowStride     = sampleW * interleavedFmt.planePixelStrideBytes(0);
        const int widthBytes    = sampleW * elemSize;
        const int maskRowStride = sampleW * maskElemSize;

        std::vector<uint8_t> src;
        std::vector<uint8_t> mask;
        FillInpaintStripe(src, mask, sampleW, sampleH, channels, elemSize);

        auto srcIData  = srcI[i].exportData<nvcv::ImageDataStridedCuda>();
        auto srcPData  = srcP[i].exportData<nvcv::ImageDataStridedCuda>();
        auto maskIData = maskI[i].exportData<nvcv::ImageDataStridedCuda>();
        auto maskPData = maskP[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcIData && srcPData && maskIData && maskPData);
        ASSERT_EQ(srcPData->numPlanes(), channels);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcIData->plane(0).basePtr, srcIData->plane(0).rowStride, src.data(),
                                            rowStride, rowStride, sampleH, cudaMemcpyHostToDevice));
        auto      planes     = nvcv::test::planar::DeinterleaveToPlanes(src, sampleW, sampleH, channels, elemSize);
        const int planeBytes = sampleW * sampleH * elemSize;
        for (int c = 0; c < channels; c++)
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcPData->plane(c).basePtr, srcPData->plane(c).rowStride,
                                                planes.data() + c * planeBytes, widthBytes, widthBytes, sampleH,
                                                cudaMemcpyHostToDevice));
        }
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskIData->plane(0).basePtr, maskIData->plane(0).rowStride, mask.data(),
                                            maskRowStride, maskRowStride, sampleH, cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskPData->plane(0).basePtr, maskPData->plane(0).rowStride, mask.data(),
                                            maskRowStride, maskRowStride, sampleH, cudaMemcpyHostToDevice));
    }

    cvcuda::Inpaint op(numImages, nvcv::Size2D{width + (numImages - 1) * 3, height + (numImages - 1) * 2});
    EXPECT_NO_THROW(op(stream, batchSrcI, batchMaskI, batchDstI, inpaintRadius));
    EXPECT_NO_THROW(op(stream, batchSrcP, batchMaskP, batchDstP, inpaintRadius));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; i++)
    {
        SCOPED_TRACE(i);

        const int sampleW    = dstI[i].size().w;
        const int sampleH    = dstI[i].size().h;
        const int rowStride  = sampleW * interleavedFmt.planePixelStrideBytes(0);
        const int widthBytes = sampleW * elemSize;

        std::vector<uint8_t> gpuInter(static_cast<size_t>(sampleH) * rowStride);
        auto                 dstIData = dstI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(gpuInter.data(), rowStride, dstIData->plane(0).basePtr,
                                            dstIData->plane(0).rowStride, rowStride, sampleH, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> planesOut(static_cast<size_t>(sampleW) * sampleH * channels * elemSize);
        auto                 dstPData   = dstP[i].exportData<nvcv::ImageDataStridedCuda>();
        const int            planeBytes = sampleW * sampleH * elemSize;
        for (int c = 0; c < channels; c++)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(planesOut.data() + c * planeBytes, widthBytes, dstPData->plane(c).basePtr,
                                   dstPData->plane(c).rowStride, widthBytes, sampleH, cudaMemcpyDeviceToHost));
        }
        auto planarInter = nvcv::test::planar::InterleaveFromPlanes(planesOut, sampleW, sampleH, channels, elemSize);

        if (gpuInter != planarInter)
        {
            const auto [interIt, planarIt] = std::mismatch(gpuInter.begin(), gpuInter.end(), planarInter.begin());
            ADD_FAILURE() << "sample=" << i << " offset=" << std::distance(gpuInter.begin(), interIt)
                          << " interleaved=" << static_cast<int>(*interIt) << " planar=" << static_cast<int>(*planarIt);
        }
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpInpaintPlanar,
                  nvcv::test::ValueList<int, int, int, double, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // width, height, batch, radius, planarFmt, interleavedFmt
    {    64,     48,     2,    5.0,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {    80,     60,     1,    3.0,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
});

NVCV_TEST_SUITE_P(OpInpaintPlanarVarShape,
                  nvcv::test::ValueList<int, int, int, double, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // width, height, batch, radius, planarFmt, interleavedFmt
    {    64,     48,     2,    5.0, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
});

// clang-format on

TEST_P(OpInpaintPlanar, tensor_matches_interleaved)
{
    RunTensorInpaintPlanarParity(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(),
                                 GetParamValue<2>(), GetParamValue<3>());
}

TEST_P(OpInpaintPlanarVarShape, varshape_matches_interleaved)
{
    RunVarShapeInpaintPlanarParity(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(),
                                   GetParamValue<2>(), GetParamValue<3>());
}

// clang-format off
NVCV_TEST_SUITE_P(OpInpaint_Negative, nvcv::test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat>
    {
        //fmtSrc, fmtDst, fmtMask
        // invalid input, output
        {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_U8},
        {nvcv::FMT_RGBf16, nvcv::FMT_RGB8, nvcv::FMT_U8},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_U8},
        {nvcv::FMT_RGB8, nvcv::FMT_RGBf32, nvcv::FMT_U8},
        {nvcv::FMT_RGBA8, nvcv::FMT_RGB8, nvcv::FMT_U8},
        // invalid mask
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_F32},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8},
    });

// clang-format on

TEST(OpInpaint_Negative, create_will_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaInpaintCreate(nullptr, 1, 1, 1));
}

TEST(OpInpaint_Negative, create_rejects_non_positive_limits)
{
    const std::array<std::tuple<int32_t, int32_t, int32_t>, 6> invalidLimits{
        std::make_tuple(-1, 4, 4), std::make_tuple(0, 4, 4),  std::make_tuple(1, -1, 4),
        std::make_tuple(1, 0, 4),  std::make_tuple(1, 4, -1), std::make_tuple(1, 4, 0),
    };

    for (auto [maxBatchSize, maxHeight, maxWidth] : invalidLimits)
    {
        NVCVOperatorHandle handle = nullptr;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaInpaintCreate(&handle, maxBatchSize, maxHeight, maxWidth))
            << "maxBatchSize=" << maxBatchSize << ", maxHeight=" << maxHeight << ", maxWidth=" << maxWidth;
        EXPECT_EQ(nullptr, handle);
    }
}

TEST(OpInpaint_Negative, tensor_batch_exceeds_maxBatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int    maxBatch      = 1;
    constexpr int    batch         = 2;
    constexpr int    width         = 4;
    constexpr int    height        = 4;
    constexpr double inpaintRadius = 1.0;

    nvcv::Tensor imgIn   = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgMask = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgOut  = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);

    cvcuda::Inpaint op(maxBatch, nvcv::Size2D{width, height});
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &imgIn, &imgMask, &imgOut, &inpaintRadius]
                                                             { op(stream, imgIn, imgMask, imgOut, inpaintRadius); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint_Negative, tensor_shape_exceeds_maxShape)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int    maxBatch      = 1;
    constexpr int    maxWidth      = 4;
    constexpr int    maxHeight     = 4;
    constexpr int    width         = maxWidth + 1;
    constexpr int    height        = maxHeight + 1;
    constexpr double inpaintRadius = 1.0;

    nvcv::Tensor imgIn   = nvcv::util::CreateTensor(maxBatch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgMask = nvcv::util::CreateTensor(maxBatch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgOut  = nvcv::util::CreateTensor(maxBatch, width, height, nvcv::FMT_U8);

    cvcuda::Inpaint op(maxBatch, nvcv::Size2D{maxWidth, maxHeight});
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, imgIn, imgMask, imgOut, inpaintRadius); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint_Negative, varshape_batch_exceeds_maxBatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int    maxBatch      = 1;
    constexpr int    batch         = 2;
    constexpr int    width         = 4;
    constexpr int    height        = 4;
    constexpr double inpaintRadius = 1.0;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgMask;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batch; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
        imgMask.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
        imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchMask(batch);
    batchMask.pushBack(imgMask.begin(), imgMask.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Inpaint op(maxBatch, nvcv::Size2D{width, height});
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &batchSrc, &batchMask, &batchDst, &inpaintRadius]
                                { op(stream, batchSrc, batchMask, batchDst, inpaintRadius); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint_Negative, varshape_shape_exceeds_maxShape)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int    maxBatch      = 1;
    constexpr int    maxWidth      = 4;
    constexpr int    maxHeight     = 4;
    constexpr int    width         = maxWidth + 1;
    constexpr int    height        = maxHeight + 1;
    constexpr double inpaintRadius = 1.0;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgMask;
    std::vector<nvcv::Image> imgDst;
    imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
    imgMask.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
    imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);

    nvcv::ImageBatchVarShape batchSrc(maxBatch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchMask(maxBatch);
    batchMask.pushBack(imgMask.begin(), imgMask.end());

    nvcv::ImageBatchVarShape batchDst(maxBatch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Inpaint op(maxBatch, nvcv::Size2D{maxWidth, maxHeight});
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, batchSrc, batchMask, batchDst, inpaintRadius); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint_Negative, varshape_batch_count_mismatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int    maxBatch      = 3;
    constexpr int    inputBatch    = 2;
    constexpr int    width         = 4;
    constexpr int    height        = 4;
    constexpr double inpaintRadius = 1.0;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgMask;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < inputBatch; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
        imgMask.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
    }
    for (int i = 0; i < maxBatch; ++i)
    {
        imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
    }

    nvcv::ImageBatchVarShape batchSrc(maxBatch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchMask(maxBatch);
    batchMask.pushBack(imgMask.begin(), imgMask.end());

    nvcv::ImageBatchVarShape batchDst(maxBatch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Inpaint op(maxBatch, nvcv::Size2D{width, height});
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, batchSrc, batchMask, batchDst, inpaintRadius); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint_Negative, tensor_planar_2channel_rejected)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn(
        {
            {1, 2, 32, 32},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgMask(
        {
            {1, 1, 32, 32},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgOut(
        {
            {1, 2, 32, 32},
            "NCHW"
    },
        nvcv::TYPE_U8);

    cvcuda::Inpaint op(1, nvcv::Size2D{32, 32});
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, imgIn, imgMask, imgOut, 1.0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInpaint_Negative, varshape_planar_2channel_rejected)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const nvcv::ImageFormat  twoChannelPlanar = kFmt2U8Planar;
    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgMask;
    std::vector<nvcv::Image> imgDst;
    imgSrc.emplace_back(nvcv::Size2D{32, 32}, twoChannelPlanar);
    imgMask.emplace_back(nvcv::Size2D{32, 32}, nvcv::FMT_U8);
    imgDst.emplace_back(nvcv::Size2D{32, 32}, twoChannelPlanar);

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchMask(1);
    batchMask.pushBack(imgMask.begin(), imgMask.end());
    nvcv::ImageBatchVarShape batchDst(1);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Inpaint op(1, nvcv::Size2D{32, 32});
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, batchSrc, batchMask, batchDst, 1.0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpInpaint_Negative, invalid_parameters)
{
    const int    maxBatch      = 4;
    const int    maxWidth      = 32;
    const int    maxHeight     = 32;
    const double inpaintRadius = 1.0;
    nvcv::Size2D maxsize{maxWidth, maxHeight};

    const int batch  = 2;
    const int width  = 16;
    const int height = 16;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmtSrc  = GetParamValue<0>();
    nvcv::ImageFormat fmtDst  = GetParamValue<1>();
    nvcv::ImageFormat fmtMask = GetParamValue<2>();

    nvcv::Tensor imgIn   = nvcv::util::CreateTensor(batch, width, height, fmtSrc);
    nvcv::Tensor imgMask = nvcv::util::CreateTensor(batch, width, height, fmtMask);
    nvcv::Tensor imgOut  = nvcv::util::CreateTensor(batch, width, height, fmtDst);

    cvcuda::Inpaint InpaintOp(maxBatch, maxsize);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&InpaintOp, &stream, &imgIn, &imgMask, &imgOut, &inpaintRadius]
                                { InpaintOp(stream, imgIn, imgMask, imgOut, inpaintRadius); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpInpaint_Negative, invalid_parameters_varshape)
{
    const int    maxBatch      = 4;
    const int    maxWidth      = 32;
    const int    maxHeight     = 32;
    const double inpaintRadius = 1.0;
    nvcv::Size2D maxsize{maxWidth, maxHeight};

    const int batch  = 2;
    const int width  = 16;
    const int height = 16;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmtSrc  = GetParamValue<0>();
    nvcv::ImageFormat fmtDst  = GetParamValue<1>();
    nvcv::ImageFormat fmtMask = GetParamValue<2>();

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    std::vector<nvcv::Image> imgMask;
    for (int i = 0; i < batch; i++)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, fmtSrc);
        imgMask.emplace_back(nvcv::Size2D{width, height}, fmtMask);
        imgDst.emplace_back(nvcv::Size2D{width, height}, fmtDst);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchMask(batch);
    batchMask.pushBack(imgMask.begin(), imgMask.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Inpaint InpaintOp(maxBatch, maxsize);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&InpaintOp, &stream, &batchSrc, &batchMask, &batchDst, &inpaintRadius]
                                { InpaintOp(stream, batchSrc, batchMask, batchDst, inpaintRadius); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
