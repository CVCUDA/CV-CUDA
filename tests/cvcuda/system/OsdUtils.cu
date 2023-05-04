/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OsdUtils.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

#include <fstream>
#include <vector>

namespace nvcv::test { namespace osd {

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

inline static bool check_runtime(cudaError_t e, const char *call, int line, const char *file)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e),
                cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

// Create image using size and format
Image *create_image(int width, int height, ImageFormat format)
{
    Image *output  = new Image();
    output->width  = width;
    output->height = height;
    output->format = format;

    if (format == ImageFormat::RGB)
    {
        output->stride = output->width * 3;
        checkRuntime(cudaMalloc(&output->data0, output->stride * output->height));
    }
    else if (format == ImageFormat::RGBA)
    {
        output->stride = output->width * 4;
        checkRuntime(cudaMalloc(&output->data0, output->stride * output->height));
    }
    else if (format == ImageFormat::PitchLinearNV12)
    {
        output->stride = output->width;
        if (output->width % 2 != 0 || output->height % 2 != 0)
        {
            fprintf(stderr, "Invalid image size(%d, %d) for NV12\n", output->width, output->height);
            delete output;
            return nullptr;
        }
        checkRuntime(cudaMalloc(&output->data0, output->stride * output->height));
        checkRuntime(cudaMalloc(&output->data1, output->stride * output->height / 2));
    }
    else if (format == ImageFormat::BlockLinearNV12)
    {
        output->stride = output->width;
        if (output->width % 2 != 0 || output->height % 2 != 0)
        {
            fprintf(stderr, "Invalid image size(%d, %d) for NV12\n", output->width, output->height);
            delete output;
            return nullptr;
        }
        cudaChannelFormatDesc planeDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        checkRuntime(cudaMallocArray((cudaArray_t *)&output->reserve0, &planeDesc, output->stride, height));
        checkRuntime(cudaMallocArray((cudaArray_t *)&output->reserve1, &planeDesc, output->stride, height / 2));

        cudaResourceDesc luma_desc = {};
        luma_desc.resType          = cudaResourceTypeArray;
        luma_desc.res.array.array  = (cudaArray_t)output->reserve0;
        checkRuntime(cudaCreateSurfaceObject((cudaSurfaceObject_t *)&output->data0, &luma_desc));

        cudaResourceDesc chroma_desc = {};
        chroma_desc.resType          = cudaResourceTypeArray;
        chroma_desc.res.array.array  = (cudaArray_t)output->reserve1;
        checkRuntime(cudaCreateSurfaceObject((cudaSurfaceObject_t *)&output->data1, &chroma_desc));
    }
    else
    {
        fprintf(stderr, "Unsupport format %d\n", (int)format);
        delete output;
        output = nullptr;
    }
    return output;
}

// Free image pointer
void free_image(Image *image)
{
    if (image == nullptr)
        return;

    if (image->format == ImageFormat::RGB)
    {
        if (image->data0)
            checkRuntime(cudaFree(image->data0));
    }
    else if (image->format == ImageFormat::RGBA)
    {
        if (image->data0)
            checkRuntime(cudaFree(image->data0));
    }
    else if (image->format == ImageFormat::PitchLinearNV12)
    {
        if (image->data0)
            checkRuntime(cudaFree(image->data0));
        if (image->data1)
            checkRuntime(cudaFree(image->data1));
    }
    else if (image->format == ImageFormat::BlockLinearNV12)
    {
        if (image->data0)
            checkRuntime(cudaDestroySurfaceObject((cudaSurfaceObject_t)image->data0));
        if (image->data1)
            checkRuntime(cudaDestroySurfaceObject((cudaSurfaceObject_t)image->data1));
        if (image->reserve0)
            checkRuntime(cudaFreeArray((cudaArray_t)image->reserve0));
        if (image->reserve1)
            checkRuntime(cudaFreeArray((cudaArray_t)image->reserve1));
    }
    delete image;
}

void cuosd_apply(cuOSDContext_t context, Image *image, void *_stream, bool launch)
{
    cudaStream_t stream = (cudaStream_t)_stream;

    cuOSDImageFormat format = cuOSDImageFormat::None;
    if (image->format == ImageFormat::RGB)
    {
        format = cuOSDImageFormat::RGB;
    }
    else if (image->format == ImageFormat::RGBA)
    {
        format = cuOSDImageFormat::RGBA;
    }
    else if (image->format == ImageFormat::PitchLinearNV12)
    {
        format = cuOSDImageFormat::PitchLinearNV12;
    }
    else if (image->format == ImageFormat::BlockLinearNV12)
    {
        format = cuOSDImageFormat::BlockLinearNV12;
    }
    cuosd_apply(context, image->data0, image->data1, image->width, image->stride, image->height, format, stream,
                launch);
}

void cuosd_launch(cuOSDContext_t context, Image *image, void *_stream)
{
    cudaStream_t stream = (cudaStream_t)_stream;

    cuOSDImageFormat format = cuOSDImageFormat::None;
    if (image->format == ImageFormat::RGB)
    {
        format = cuOSDImageFormat::RGB;
    }
    else if (image->format == ImageFormat::RGBA)
    {
        format = cuOSDImageFormat::RGBA;
    }
    else if (image->format == ImageFormat::PitchLinearNV12)
    {
        format = cuOSDImageFormat::PitchLinearNV12;
    }
    else if (image->format == ImageFormat::BlockLinearNV12)
    {
        format = cuOSDImageFormat::BlockLinearNV12;
    }
    cuosd_launch(context, image->data0, image->data1, image->width, image->stride, image->height, format, stream);
}
}} // namespace nvcv::test::osd
