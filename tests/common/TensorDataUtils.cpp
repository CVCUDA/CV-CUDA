/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "TensorDataUtils.hpp"

#include <cmath>

namespace nvcv::test {

static void printPlane(const uint8_t *data, int width, int height, int rowStride, int bytesPC, int numC)
{
    std::cout << "[";
    printf("%02x", data[0]);
    bool endB = false;
    int  size = height * rowStride;
    int  x    = 1;
    for (int i = 1; i < size; i++)
    {
        if (x % (width * bytesPC * numC) == 0 && !endB)
        {
            std::cout << "]";
            endB = true;
        }
        else if (i % bytesPC == 0)
        {
            if (x % (bytesPC * numC) == 0 && !endB)
            {
                std::cout << " |";
            }
            else
            {
                std::cout << ",";
            }
        }
        if (i % rowStride == 0)
        {
            std::cout << "\n[";
            endB = false;
            x    = 1;
        }
        else
        {
            x++;
            std::cout << " ";
        }
        printf("%02x", data[i]);
    }

    if (!endB)
        std::cout << "]";
    else
        std::cout << ",";

    std::cout << "\n";
}

TensorImageData::TensorImageData(const ITensorData *tensorData, int sampleIndex)
    : m_planeStride(0)
{
    assert(tensorData);
    if (!nvcv::TensorDataAccessStridedImage::IsCompatible(*tensorData))
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor Data not compatible with Pitch Access");

    auto tDataAc = nvcv::TensorDataAccessStridedImage::Create(*tensorData);

    int sizeSampleBytes = tDataAc->sampleStride();
    m_rowStride         = tDataAc->rowStride();

    m_data.resize(sizeSampleBytes);

    if (cudaSuccess
        != cudaMemcpy(m_data.data(), tDataAc->sampleData(sampleIndex), tDataAc->sampleStride(), cudaMemcpyDeviceToHost))
    {
        throw Exception(Status::ERROR_INTERNAL, "cudaMemcpy filed");
    }

    m_layout    = tDataAc->infoLayout().layout().m_layout;
    m_bytesPerC = tDataAc->dtype().bitsPerPixel() / 8;
    m_planar    = false;
    if (m_layout == NVCV_TENSOR_HWC)
    {
        m_size.h = tDataAc->infoShape().shape()[0];
        m_size.w = tDataAc->infoShape().shape()[1];
        m_numC   = tDataAc->infoShape().shape()[2];
    }
    else if (m_layout == NVCV_TENSOR_NHWC)
    {
        m_size.h = tDataAc->infoShape().shape()[1];
        m_size.w = tDataAc->infoShape().shape()[2];
        m_numC   = tDataAc->infoShape().shape()[3];
    }
    else if (m_layout == NVCV_TENSOR_CHW)
    {
        m_numC   = tDataAc->infoShape().shape()[0];
        m_size.h = tDataAc->infoShape().shape()[1];
        m_size.w = tDataAc->infoShape().shape()[2];
        m_planar = true;
    }
    else if (m_layout == NVCV_TENSOR_NCHW)
    {
        m_numC   = tDataAc->infoShape().shape()[1];
        m_size.h = tDataAc->infoShape().shape()[2];
        m_size.w = tDataAc->infoShape().shape()[3];
        m_planar = true;
    }
    else
    {
        throw std::runtime_error("Tensor layout unknown");
    }

    if (m_planar)
    {
        if (!nvcv::TensorDataAccessStridedImagePlanar::IsCompatible(*tensorData))
            throw std::runtime_error("Tensor Data not compatible with Pitch Planar Access");

        auto tDataACp = nvcv::TensorDataAccessStridedImagePlanar::Create(*tensorData);
        m_planeStride = tDataACp->planeStride();
    }

    return;
}

std::ostream &operator<<(std::ostream &out, const TensorImageData &cvImageData)
{
    out << "\n[H = " << cvImageData.m_size.h << " W = " << cvImageData.m_size.w << " C = " << cvImageData.m_numC
        << "]\n[planar = " << cvImageData.m_planar << " bytesPerC = " << cvImageData.m_bytesPerC
        << " rowStride = " << cvImageData.m_rowStride << " planeStride = " << cvImageData.m_planeStride
        << " sampleStride = " << cvImageData.m_data.size() << "]\n";

    if (!cvImageData.m_planar)
    {
        printPlane(&cvImageData.m_data[0], cvImageData.m_size.w, cvImageData.m_size.h, cvImageData.m_rowStride,
                   cvImageData.m_bytesPerC, cvImageData.m_numC);
    }
    else
    {
        for (int i = 0; i < cvImageData.m_numC; i++)
        {
            out << "\nPlane = " << i << "\n";
            printPlane(&cvImageData.m_data[cvImageData.m_planeStride * i], cvImageData.m_size.w, cvImageData.m_size.h,
                       cvImageData.m_rowStride, cvImageData.m_bytesPerC, 1);
        }
    }
    return out << "\n";
}

bool TensorImageData::operator==(const TensorImageData &that) const
{
    return ((this->m_size == that.m_size) && (this->m_rowStride == that.m_rowStride) && (this->m_numC == that.m_numC)
            && (this->m_planar == that.m_planar) && (this->m_bytesPerC == that.m_bytesPerC)
            && (this->m_layout == that.m_layout) && (this->m_data.size() == that.m_data.size())
            && (memcmp(this->m_data.data(), that.m_data.data(), this->m_data.size()) == 0));
}

bool TensorImageData::operator!=(const TensorImageData &that) const
{
    return !operator==(that);
}

} // namespace nvcv::test
