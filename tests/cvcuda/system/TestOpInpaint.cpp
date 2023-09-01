/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/ValueTests.hpp>
#include <cvcuda/OpInpaint.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <cmath>
#include <iostream>
#include <random>

using namespace std;

#define KNOWN  0 //known outside narrow band
#define BAND   1 //narrow band (known)
#define INSIDE 2 //unknown
#define CHANGE 3 //servise

struct Point2f
{
    float x, y;
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

typedef struct HeapElem
{
    float            T;
    int              i, j;
    struct HeapElem *prev;
    struct HeapElem *next;
} HeapElem;

class PriorityQueueFloat
{
private:
    PriorityQueueFloat(const PriorityQueueFloat &);            // copy disabled
    PriorityQueueFloat &operator=(const PriorityQueueFloat &); // assign disabled

protected:
    HeapElem *mem, *empty, *head, *tail;
    int       num, in;

public:
    bool Init(const vector<uint8_t> &f, int height, int width)
    {
        int i, j;
        for (i = num = 0; i < height; i++)
        {
            for (j = 0; j < width; j++) num += (f[i * width + j] != 0);
        }
        if (num <= 0)
            return false;
        mem = (HeapElem *)malloc((num + 2) * sizeof(HeapElem));
        if (mem == NULL)
            return false;

        head    = mem;
        head->i = head->j = -1;
        head->prev        = NULL;
        head->next        = mem + 1;
        head->T           = -FLT_MAX;
        empty             = mem + 1;
        for (i = 1; i <= num; i++)
        {
            mem[i].prev = mem + i - 1;
            mem[i].next = mem + i + 1;
            mem[i].i    = -1;
            mem[i].T    = FLT_MAX;
        }
        tail    = mem + i;
        tail->i = tail->j = -1;
        tail->prev        = mem + i - 1;
        tail->next        = NULL;
        tail->T           = FLT_MAX;
        return true;
    }

    bool Add(const vector<uint8_t> &f, int height, int width)
    {
        int i, j;
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                if (f[i * width + j] != 0)
                {
                    if (!Push(i, j, 0))
                        return false;
                }
            }
        }
        return true;
    }

    bool Push(int i, int j, float T)
    {
        HeapElem *tmp = empty, *add = empty;
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

    PriorityQueueFloat(void)
    {
        num = in = 0;
        mem = empty = head = tail = NULL;
    }

    ~PriorityQueueFloat(void)
    {
        free(mem);
    }
};

static float FastMarching_solve(int i1, int j1, int i2, int j2, const vector<uint8_t> &f, const vector<float> &t,
                                int width)
{
    double sol, a11, a22, m12;
    a11 = t[i1 * width + j1];
    a22 = t[i2 * width + j2];
    m12 = min(a11, a22);

    if (f[i1 * width + j1] != INSIDE)
        if (f[i2 * width + j2] != INSIDE)
            if (fabs(a11 - a22) >= 1.0)
                sol = 1 + m12;
            else
                sol = (a11 + a22 + sqrt((double)(2 - (a11 - a22) * (a11 - a22)))) * 0.5;
        else
            sol = 1 + a11;
    else if (f[i2 * width + j2] != INSIDE)
        sol = 1 + a22;
    else
        sol = 1 + m12;

    return (float)sol;
}

static void InpaintFMM(vector<uint8_t> &f, vector<float> &t, vector<uint8_t> &out, int range,
                       shared_ptr<PriorityQueueFloat> Heap, int height, int width)
{
    int   i = 0, j = 0, ii = 0, jj = 0, k, l, q, color = 0;
    float dist;

    while (Heap->Pop(&ii, &jj))
    {
        f[ii * width + jj] = KNOWN;
        for (q = 0; q < 4; q++)
        {
            if (q == 0)
            {
                i = ii - 1;
                j = jj;
            }
            else if (q == 1)
            {
                i = ii;
                j = jj - 1;
            }
            else if (q == 2)
            {
                i = ii + 1;
                j = jj;
            }
            else if (q == 3)
            {
                i = ii;
                j = jj + 1;
            }
            if ((i <= 0) || (j <= 0) || (i > height - 1) || (j > width - 1))
                continue;

            if (f[i * width + j] == INSIDE)
            {
                dist             = min4(FastMarching_solve(i - 1, j, i, j - 1, f, t, width),
                                        FastMarching_solve(i + 1, j, i, j - 1, f, t, width),
                                        FastMarching_solve(i - 1, j, i, j + 1, f, t, width),
                                        FastMarching_solve(i + 1, j, i, j + 1, f, t, width));
                t[i * width + j] = dist;

                for (color = 0; color <= 0; color++)
                {
                    Point2f gradI, gradT, r;
                    float   Ia = 0, Jx = 0, Jy = 0, s = 1.0e-20f, w, dst, lev, dir, sat;

                    if (f[i * width + j + 1] != INSIDE)
                    {
                        if (f[i * width + j - 1] != INSIDE)
                        {
                            gradT.x = (float)((t[i * width + j + 1] - t[i * width + j - 1])) * 0.5f;
                        }
                        else
                        {
                            gradT.x = (float)((t[i * width + j + 1] - t[i * width + j]));
                        }
                    }
                    else
                    {
                        if (f[i * width + j - 1] != INSIDE)
                        {
                            gradT.x = (float)((t[i * width + j] - t[i * width + j - 1]));
                        }
                        else
                        {
                            gradT.x = 0;
                        }
                    }
                    if (f[(i + 1) * width + j] != INSIDE)
                    {
                        if (f[(i - 1) * width + j] != INSIDE)
                        {
                            gradT.y = (float)((t[(i + 1) * width + j] - t[(i - 1) * width + j])) * 0.5f;
                        }
                        else
                        {
                            gradT.y = (float)((t[(i + 1) * width + j] - t[i * width + j]));
                        }
                    }
                    else
                    {
                        if (f[(i - 1) * width + j] != INSIDE)
                        {
                            gradT.y = (float)((t[i * width + j] - t[(i - 1) * width + j]));
                        }
                        else
                        {
                            gradT.y = 0;
                        }
                    }
                    for (k = i - range; k <= i + range; k++)
                    {
                        int km = k - 1 + (k == 1), kp = k - 1 - (k == height - 2);
                        for (l = j - range; l <= j + range; l++)
                        {
                            int lm = l - 1 + (l == 1), lp = l - 1 - (l == width - 2);
                            if (k > 0 && l > 0 && k < height - 1 && l < width - 1)
                            {
                                if ((f[k * width + l] != INSIDE)
                                    && ((l - j) * (l - j) + (k - i) * (k - i) <= range * range))
                                {
                                    r.y = (float)(i - k);
                                    r.x = (float)(j - l);

                                    dst = (float)(1. / (VectorLength(r) * sqrt(VectorLength(r))));
                                    lev = (float)(1. / (1 + fabs(t[k * width + l] - t[i * width + j])));

                                    dir = VectorScalMult(r, gradT);
                                    if (fabs(dir) <= 0.01)
                                        dir = 0.000001f;
                                    w = (float)fabs(dst * lev * dir);

                                    if (f[k * width + l + 1] != INSIDE)
                                    {
                                        if (f[k * width + l - 1] != INSIDE)
                                        {
                                            gradI.x = (float)((out[km * (width - 2) + lp + 1]
                                                               - out[km * (width - 2) + lm - 1]))
                                                    * 2.0f;
                                        }
                                        else
                                        {
                                            gradI.x = (float)((out[km * (width - 2) + lp + 1]
                                                               - out[km * (width - 2) + lm]));
                                        }
                                    }
                                    else
                                    {
                                        if (f[k * width + l - 1] != INSIDE)
                                        {
                                            gradI.x = (float)((out[km * (width - 2) + lp]
                                                               - out[km * (width - 2) + lm - 1]));
                                        }
                                        else
                                        {
                                            gradI.x = 0;
                                        }
                                    }
                                    if (f[(k + 1) * width + l] != INSIDE)
                                    {
                                        if (f[(k - 1) * width + l] != INSIDE)
                                        {
                                            gradI.y = (float)((out[(kp + 1) * (width - 2) + lm]
                                                               - out[(km - 1) * (width - 2) + lm]))
                                                    * 2.0f;
                                        }
                                        else
                                        {
                                            gradI.y = (float)((out[(kp + 1) * (width - 2) + lm]
                                                               - out[km * (width - 2) + lm]));
                                        }
                                    }
                                    else
                                    {
                                        if (f[(k - 1) * width + l] != INSIDE)
                                        {
                                            gradI.y = (float)((out[kp * (width - 2) + lm]
                                                               - out[(km - 1) * (width - 2) + lm]));
                                        }
                                        else
                                        {
                                            gradI.y = 0;
                                        }
                                    }
                                    Ia += (float)w * (float)(out[km * (width - 2) + lm]);
                                    Jx -= (float)w * (float)(gradI.x * r.x);
                                    Jy -= (float)w * (float)(gradI.y * r.y);
                                    s += w;
                                }
                            }
                        }
                    }
                    sat = (float)((Ia / s + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20f) + 0.5f));
                    {
                        int v                                = round(sat);
                        out[(i - 1) * (width - 2) + (j - 1)] = (uint8_t)((unsigned)v <= UCHAR_MAX ? v
                                                                         : v > 0                  ? UCHAR_MAX
                                                                                                  : 0);
                        ;
                    }
                }

                f[i * width + j] = BAND;
                Heap->Push(i, j, dist);
            }
        }
    }
}

//test FMT_U8
template<typename T>
void Inpaint(std::vector<T> &src, std::vector<T> &dst, std::vector<T> &org_mask, double radius, int height, int width)
{
    int range = (int)std::round(radius);
    range     = std::max(range, 1);
    range     = std::min(range, 100);

    int             erows = height + 2, ecols = width + 2;
    vector<uint8_t> f(erows * ecols, KNOWN);
    vector<float>   t(erows * ecols, 1.0e6f);
    vector<uint8_t> band(erows * ecols, KNOWN);
    vector<uint8_t> mask(erows * ecols, KNOWN);
    vector<uint8_t> kernel = {0, 1, 0, 1, 1, 1, 0, 1, 0};

    // cvCopy(input_img, output_img);
    dst.assign(src.begin(), src.end());
    // COPY_MASK_BORDER1_C1(inpaint_mask, mask, uchar);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            if (org_mask[i * width + j])
                mask[(i + 1) * ecols + j + 1] = INSIDE;
    // SET_BORDER1_C1(mask, uchar, 0);
    for (int i = 0; i < ecols; i++)
    {
        mask[i]                       = KNOWN;
        mask[(erows - 1) * ecols + i] = KNOWN;
    }
    for (int i = 0; i < erows; i++)
    {
        mask[i * ecols]             = KNOWN;
        mask[i * ecols + ecols - 1] = KNOWN;
    }
    // cvDilate(mask, band, el_cross, 1);
    for (int i = 1; i < erows - 1; i++)
        for (int j = 1; j < ecols - 1; j++)
        {
            int     k   = 0;
            uint8_t res = 0;
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    if (kernel[k++])
                        res = max(res, mask[(i - 1 + ii) * ecols + j - 1 + jj]);
            band[i * ecols + j] = res;
        }
    // Heap = cv::makePtr<CvPriorityQueueFloat>();
    // if (!Heap->Init(band))
    // return;
    shared_ptr<PriorityQueueFloat> heap = make_shared<PriorityQueueFloat>();
    if (!heap->Init(band, erows, ecols))
        return;
    // cvSub(band, mask, band, NULL);
    // SET_BORDER1_C1(band, uchar, 0);
    for (int i = 1; i < erows - 1; i++)
        for (int j = 1; j < ecols - 1; j++) band[i * ecols + j] -= mask[i * ecols + j];
    // if (!Heap->Add(band))
    //     return;
    if (!heap->Add(band, erows, ecols))
        return;
    // cvSet(f, cvScalar(BAND, 0, 0, 0), band);
    // cvSet(f, cvScalar(INSIDE, 0, 0, 0), mask);
    // cvSet(t, cvScalar(0, 0, 0, 0), band);
    for (int i = 0; i < erows; i++)
        for (int j = 0; j < ecols; j++)
        {
            if (band[i * ecols + j])
            {
                f[i * ecols + j] = BAND;
                t[i * ecols + j] = 0;
            }
            if (mask[i * ecols + j])
                f[i * ecols + j] = INSIDE;
        }

    InpaintFMM(f, t, dst, range, heap, erows, ecols);
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
        srcVec[i].resize(height * rowStride / sizeof(uint8_t));
        fill(srcVec[i].begin(), srcVec[i].end(), 255);
        int h  = height / 2;
        int w1 = width * 0.2, w2 = width * 0.8;
        for (int hi = h - 10; hi < h + 10; hi++)
            for (int wi = w1; wi <= w2; wi++) srcVec[i][hi * width + wi] = 0;
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));

        maskVec[i].assign(height * rowStride / sizeof(uint8_t), 0);
        for (int hi = h - 10; hi < h + 10; hi++)
            for (int wi = w1; wi <= w2; wi++) maskVec[i][hi * width + wi] = 1;
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

        //ratio = count(abs(diff) > 1) / size
        //mean(abs(diff/255))
        int   count   = 0;
        float diffsum = 0.f;
        for (int x = 0; x < (int)testVec.size(); x++)
        {
            if (abs(testVec[x] - goldVec[x]) > 1)
            {
                count++;
            }
            diffsum += abs(testVec[x] - goldVec[x]);
        }
        float ratio = (float)count / (height * width);
        EXPECT_LE(ratio, 5e-2);

        diffsum /= 255;
        diffsum /= (height * width);
        EXPECT_LE(diffsum, 5e-3);
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
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> rndHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc, imgDst, imgMask;
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
    std::vector<std::vector<uint8_t>> srcVec(batch), maskVec(batch);

    for (int i = 0; i < batch; i++)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        const auto maskData = imgMask[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(maskData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        srcVec[i].resize(srcHeight * srcRowStride / sizeof(uint8_t));
        fill(srcVec[i].begin(), srcVec[i].end(), 255);
        int h  = srcHeight / 2;
        int w1 = srcWidth * 0.2, w2 = srcWidth * 0.8;
        for (int hi = h - 10; hi < h + 10; hi++)
            for (int wi = w1; wi <= w2; wi++) srcVec[i][hi * srcWidth + wi] = 0;

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));

        maskVec[i].assign(srcHeight * srcRowStride / sizeof(uint8_t), 0);
        for (int hi = h - 10; hi < h + 10; hi++)
            for (int wi = w1; wi <= w2; wi++) maskVec[i][hi * srcWidth + wi] = 1;

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(maskData->plane(0).basePtr, maskData->plane(0).rowStride, maskVec[i].data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));
    }

    // Call operator
    int             maxBatch = 4;
    nvcv::Size2D    maxsize{(int)(480 * 1.1), (int)(480 * 1.1)};
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

        //ratio = count(abs(diff) > 1) / size
        //mean(abs(diff/255))
        int   count   = 0;
        float diffsum = 0.f;
        for (int x = 0; x < (int)testVec.size(); x++)
        {
            if (abs(testVec[x] - goldVec[x]) > 1)
            {
                count++;
            }
            diffsum += abs(testVec[x] - goldVec[x]);
        }
        float ratio = (float)count / (dstHeight * dstWidth);
        EXPECT_LE(ratio, 5e-2);

        diffsum /= 255;
        diffsum /= (dstHeight * dstWidth);
        EXPECT_LE(diffsum, 5e-3);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
