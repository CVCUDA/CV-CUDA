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

// Include everything to check if C compiler groks them.
#include <nvcv/ColorSpec.h>
#include <nvcv/Config.h>
#include <nvcv/DataLayout.h>
#include <nvcv/DataType.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/ImageBatchData.h>
#include <nvcv/ImageData.h>
#include <nvcv/ImageFormat.h>
#include <nvcv/Rect.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <nvcv/TensorData.h>
#include <nvcv/TensorLayout.h>
#include <nvcv/TensorShape.h>
#include <nvcv/Version.h>
#include <nvcv/alloc/Allocator.h>
#include <nvcv/alloc/Fwd.h>
#include <nvcv/alloc/Requirements.h>

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

// Instantiate structs/enums to check if they are correctly defined (i.e. using typedef)

static void TestCompile_LayoutMake()
{
    NVCVTensorLayout layout = NVCV_TENSOR_LAYOUT_MAKE("NHWC");
    (void)layout;
}

static void TestCompile_ColorSpec()
{
    NVCVColorModel            cmodel   = NVCV_COLOR_MODEL_XYZ;
    NVCVColorSpace            cspace   = NVCV_COLOR_SPACE_BT601;
    NVCVWhitePoint            wpoint   = NVCV_WHITE_POINT_D65;
    NVCVYCbCrEncoding         ycbcrEnc = NVCV_YCbCr_ENC_SMPTE240M;
    NVCVColorTransferFunction xferFunc = NVCV_COLOR_XFER_BT2020;
    NVCVColorRange            crange   = NVCV_COLOR_RANGE_FULL;
    NVCVChromaLocation        cloc     = NVCV_CHROMA_LOC_EVEN;
    NVCVColorSpec             cspec    = NVCV_COLOR_SPEC_DISPLAYP3_LINEAR;

    NVCVColorSpec cspec2 = NVCV_MAKE_COLOR_SPEC(cspace, ycbcrEnc, xferFunc, crange, cloc, cloc);

    NVCVRawPattern        raw = NVCV_RAW_BAYER_CCCC;
    NVCVChromaSubsampling css = NVCV_CSS_411R;
}

static void TestCompile_DataLayout()
{
    NVCVPacking       packing = NVCV_PACKING_X256;
    NVCVDataKind      dkind   = NVCV_DATA_KIND_SIGNED;
    NVCVMemLayout     layout  = NVCV_MEM_LAYOUT_BL;
    NVCVChannel       ch      = NVCV_CHANNEL_Z;
    NVCVSwizzle       sw      = NVCV_SWIZZLE_XYW0;
    NVCVSwizzle       sw2     = NVCV_MAKE_SWIZZLE(NVCV_CHANNEL_0, NVCV_CHANNEL_Z, NVCV_CHANNEL_X, NVCV_CHANNEL_Z);
    NVCVByteOrder     byOrder = NVCV_ORDER_MSB;
    NVCVPackingParams pparams = {};
}

static void TestCompile_DataType()
{
    NVCVByte     b   = {};
    NVCVDataType dt  = NVCV_DATA_TYPE_4S8;
    NVCVDataType dt2 = NVCV_MAKE_DATA_TYPE(NVCV_DATA_KIND_SIGNED, NVCV_PACKING_X256);
}

static void TestCompile_Image()
{
    NVCVTypeImage            type   = NVCV_TYPE_IMAGE;
    NVCVImageHandle          handle = NULL;
    NVCVImageDataCleanupFunc fn     = NULL;
    NVCVImageRequirements    reqs   = {};
}

static void TestCompile_ImageBatch()
{
    NVCVTypeImageBatch                 type   = NVCV_TYPE_IMAGEBATCH_VARSHAPE;
    NVCVImageBatchHandle               handle = NULL;
    NVCVImageBatchDataCleanupFunc      fn     = NULL;
    NVCVImageBatchVarShapeRequirements reqs   = {};
}

static void TestCompile_ImageBatchData()
{
    NVCVImageBatchVarShapeBufferStrided bufvarstr;
    NVCVImageBatchTensorBufferStrided   bufstr;
    NVCVImageBatchBufferType            type = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA;
    NVCVImageBatchBuffer                buf;
    NVCVImageBatchData                  data;
}

static void TestCompile_ImageData()
{
    NVCVImagePlaneStrided    plstr;
    NVCVImageBufferStrided   bufstr;
    NVCVImageBufferCudaArray bufcudaarr;
    NVCVImageBufferType      type = NVCV_IMAGE_BUFFER_STRIDED_HOST;
    NVCVImageBuffer          buf;
    NVCVImageData            data;
}

static void TestCompile_ImageFormat()
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_U8;
}

static void TestCompile_Rect()
{
    NVCVRectI rc = {};
}

static void TestCompile_Status()
{
    NVCVStatus status = NVCV_ERROR_INTERNAL;
}

static void TestCompile_Tensor()
{
    NVCVTensorHandle          handle = NULL;
    NVCVTensorDataCleanupFunc fn     = NULL;
    NVCVTensorRequirements    reqs   = {};
}

static void TestCompile_TensorData()
{
    NVCVTensorBufferStrided bufstr  = {};
    NVCVTensorBufferType    buftype = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    NVCVTensorBuffer        tbuf    = {};
    NVCVTensorData          tdata   = {};
}

static void TestCompile_TensorLayout()
{
    NVCVTensorLayout layout = NVCV_TENSOR_NCHW;
    NVCVTensorLabel  lbl    = NVCV_TLABEL_HEIGHT;
}
