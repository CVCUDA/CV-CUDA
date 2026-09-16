/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CV_CUDA_LEGACY_H
#define CV_CUDA_LEGACY_H

#include "AdaptiveThresholdPolicy.hpp"
#include "CvCudaOSD.hpp"

#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <cvcuda/Workspace.hpp>
#include <nvcv/BorderType.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/Rect.h>
#include <nvcv/RoundMode.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <cstddef>
#include <random>
#include <vector>

namespace nvcv::legacy::cuda_op {

using cvcuda::Workspace;
using cvcuda::WorkspaceMem;
using cvcuda::WorkspaceMemRequirements;
using cvcuda::WorkspaceRequirements;

enum ErrorCode
{
    SUCCESS             = 0,
    INVALID_DATA_TYPE   = 1,
    INVALID_DATA_SHAPE  = 2,
    INVALID_DATA_FORMAT = 3,
    INVALID_PARAMETER   = 4
};

enum DataFormat
{
    kNCHW = 0,
    kNHWC = 1,
    kCHW  = 2,
    kHWC  = 3,
};

enum DataType
{
    kCV_8U  = 0,
    kCV_8S  = 1,
    kCV_16U = 2,
    kCV_16S = 3,
    kCV_32S = 4,
    kCV_32F = 5,
    kCV_64F = 6,
    kCV_16F = 7,
};

struct DataShape
{
    DataShape()
        : C(0)
        , H(0)
        , W(0){};
    DataShape(int n, int c, int h, int w)
        : N(n)
        , C(c)
        , H(h)
        , W(w){};
    DataShape(int c, int h, int w)
        : C(c)
        , H(h)
        , W(w){};

    bool operator==(const DataShape &s) const
    {
        return s.N == N && s.H == H && s.W == W && s.C == C;
    }

    bool operator!=(const DataShape &s) const
    {
        return !(*this == s);
    }

    friend std::ostream &operator<<(std::ostream &out, const DataShape &s)
    {
        out << "(N = " << s.N << ", H = " << s.H << ", W = " << s.W << ", C = " << s.C << ")";
        return out;
    }

    int N = 1; // batch
    int C;     // channel
    int H;     // height
    int W;     // width
};

inline size_t DataSize(DataType data_type)
{
    size_t size = 0;
    switch (data_type)
    {
    case kCV_8U:
    case kCV_8S:
        size = 1;
        break;
    case kCV_16U:
    case kCV_16S:
    case kCV_16F:
        size = 2;
        break;
    case kCV_32S:
    case kCV_32F:
        size = 4;
        break;
    case kCV_64F:
        size = 8;
        break;
    default:
        break;
    }
    return size;
}

struct WarpAffineTransform
{
    static __device__ __forceinline__ float2 calcCoord(const float *c_warpMat, int x, int y)
    {
        const auto  fx   = static_cast<float>(x);
        const auto  fy   = static_cast<float>(y);
        const float xcoo = c_warpMat[0] * fx + c_warpMat[1] * fy + c_warpMat[2];
        const float ycoo = c_warpMat[3] * fx + c_warpMat[4] * fy + c_warpMat[5];

        return make_float2(xcoo, ycoo);
    }

    // declare a 3x3 matrix/array to avoid conflicts in shared GPU kernel with warpPerspective
    float xform[9]; // NOSONAR: CUDA kernels consume this fixed-size transform storage.
};

struct PerspectiveTransform
{
    explicit PerspectiveTransform(const float *transMatrix)
    {
        xform[0] = transMatrix[0];
        xform[1] = transMatrix[1];
        xform[2] = transMatrix[2];
        xform[3] = transMatrix[3];
        xform[4] = transMatrix[4];
        xform[5] = transMatrix[5];
        xform[6] = transMatrix[6];
        xform[7] = transMatrix[7];
        xform[8] = transMatrix[8];
    }

    static __device__ __forceinline__ float2 calcCoord(const float *c_warpMat, int x, int y)
    {
        const auto  fx    = static_cast<float>(x);
        const auto  fy    = static_cast<float>(y);
        const float coeff = 1.0f / (c_warpMat[6] * fx + c_warpMat[7] * fy + c_warpMat[8]);

        const float xcoo = coeff * (c_warpMat[0] * fx + c_warpMat[1] * fy + c_warpMat[2]);
        const float ycoo = coeff * (c_warpMat[3] * fx + c_warpMat[4] * fy + c_warpMat[5]);

        return make_float2(xcoo, ycoo);
    }

    float xform[9]; // NOSONAR: CUDA kernels consume this fixed-size transform storage.
};

// cuda base operator class
class CudaBaseOp
{
public:
    CudaBaseOp() = default;

    CudaBaseOp(DataShape max_input_shape, DataShape max_output_shape)
        : max_input_shape_(max_input_shape)
        , max_output_shape_(max_output_shape)
    {
    }

    virtual ~CudaBaseOp() = default;

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    virtual size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
    {
        return 0;
    };

    bool checkDataShapeValid(DataShape input_shape, DataShape output_shape) const
    {
        int input_size      = input_shape.N * input_shape.C * input_shape.H * input_shape.W;
        int max_input_size  = max_input_shape_.N * max_input_shape_.C * max_input_shape_.H * max_input_shape_.W;
        int output_size     = output_shape.N * output_shape.C * output_shape.H * output_shape.W;
        int max_output_size = max_output_shape_.N * max_output_shape_.C * max_output_shape_.H * max_output_shape_.W;
        return (input_size <= max_input_size) && (output_size <= max_output_size);
    }

private:
    DataShape max_input_shape_;
    DataShape max_output_shape_;
};

class CustomCrop : public CudaBaseOp
{
public:
    CustomCrop() = delete;

    CustomCrop(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Crops the a given input image into a destination image.
     *        Destination will have the [0,0] position populated by the x,y position as
     *        defined in the ROI x,y parameters of the input data. The operator will continue to populate the
     *        output data until the destination image is populated with the size described by the ROI.
     *
     *
     * Limitations:
     *
     * ROI must be smaller than output tensor.
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | Yes
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | Yes
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     *
     * @param [in] in input tensor.
     *
     * @param [out] out output tensor.
     * @param [in]  roi region of interest, defined in pixels
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, NVCVRectI roi,
                    cudaStream_t stream);
};

class Morphology : public CudaBaseOp
{
public:
    Morphology()           = default;
    ~Morphology() override = default;

    /**
     * @brief Dilates/Erodes an image
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | No
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     *
     * @param inData gpuData to a tensor of one or more HWC images
     * @param outData gpuData a tensor hosting the outputs of the operation
     * @param morph_type Type of operation to perform on data Erode/Dilate
     * @param mask_size shape and size of the mask to use for the operation
     * @param anchor anchor to use for the kernel (-1,-1) will use center of kernel
     * @param noop if 0 this will be a copy operation
     * @param borderMode the border mode to use when accessing data outside of source
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                    NVCVMorphologyType morph_type, Size2D mask_size, int2 anchor, bool noop,
                    const NVCVBorderType borderMode, cudaStream_t stream);
};

class MorphologyVarShape : public CudaBaseOp
{
public:
    MorphologyVarShape() = default;

    ~MorphologyVarShape() override = default;
    /**
     * @brief Dilates/Erodes an image
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | No
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     *
     * @param inData gpuData to a tensor of one or more HWC images
     * @param outData gpuData a tensor hosting the outputs of the operation
     * @param morph_type Type of operation to perform on data Erode/Dilate
     * @param mask_size Tensor of the shape and sizes of the mask to use for the operation
     * @param anchor Tensor to as anchor data in the kernel (-1,-1) will use center of kernel
     * @param noop if true this will be a copy operation
     * @param borderMode the border mode to use when acessing data outside of source
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const nvcv::ImageBatchVarShape &inBatch, const nvcv::ImageBatchVarShape &outBatch,
                    NVCVMorphologyType morph_type, const TensorDataStridedCuda &masks,
                    const TensorDataStridedCuda &anchors, bool noop, NVCVBorderType borderMode,
                    bool enableGenericInterior, cudaStream_t stream);
};

class PadAndStack : public CudaBaseOp
{
public:
    PadAndStack() = delete;

    PadAndStack(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     * Top/left Tensors
     *
     *     Must be kNHWC where N=H=C=1 with W = N (N in reference to input and output tensors).
     *     Data Type must be 32bit Signed.
     */

    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                    const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                    const NVCVBorderType borderMode, const float borderValue, cudaStream_t stream);
};

class Rotate : public CudaBaseOp
{
public:
    Rotate() = delete;
    Rotate(DataShape max_input_shape, DataShape max_output_shape);

    ~Rotate() override;

    /**
     * @brief Rotates input images around the origin (0,0) and then shifts it.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param dsize size of the output images.
     * @param angle angle of rotation in degrees.
     * @param xShift shift along the horizontal axis.
     * @param yShift shift along the vertical axis.
     * @param interpolation interpolation method. Only INTER_NEAREST, INTER_LINEAR, and INTER_CUBIC are supported.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const double angleDeg,
                    const double2 shift, const NVCVInterpolationType interpolation, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type) override;

private:
    float *d_aCoeffs;
};

class MedianBlur : public CudaBaseOp
{
public:
    MedianBlur() = delete;

    MedianBlur(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Blur an image using a median kernel.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param ksize median blur kernel size.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const nvcv::Size2D ksize,
                    cudaStream_t stream);
};

class CopyMakeBorder : public CudaBaseOp
{
public:
    CopyMakeBorder() = delete;

    CopyMakeBorder(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     * @brief Forms a border around an image.
     * The function copies the source image into the middle of the destination image. The areas to the left, to the
     * right, above and below the copied source image will be filled with extrapolated pixels. This is not what
     * filtering functions based on it do (they extrapolate pixels on-fly), but what other more complex functions,
     * including your own, may do to simplify image boundary handling.
     * @param inData Input Tensor
     * @param outData Output Tensor
     * @param top the top pixels.
     * @param left the left pixels.
     * Parameter specifying how many pixels in each direction from the source image rectangle to extrapolate.
     * The src and dist size can be got from input and output tensor.
     * For example, top=1, left=1, src_w=64, src_h=64, dist_w=66, dist_h=66 mean that it builds 1 pixel-wide border.
     * @param border_type border type. See \p NVCVBorderType for details.
     * @param borderValue border value if borderType==BORDER_CONSTANT.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const int top,
                    const int left, const NVCVBorderType border_type, const float4 &borderValue, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
};

class CopyMakeBorderVarShape : public CudaBaseOp
{
public:
    CopyMakeBorderVarShape() = delete;

    CopyMakeBorderVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Forms a border around an image.
     * The function copies the source image into the middle of the destination image. The areas to the left, to the
     * right, above and below the copied source image will be filled with extrapolated pixels. This is not what
     * filtering functions based on it do (they extrapolate pixels on-fly), but what other more complex functions,
     * including your own, may do to simplify image boundary handling.
     * @param inData Input Tensor
     * @param outData Output Tensor
     * @param top the top pixels.
     * @param left the left pixels.
     * Parameter specifying how many pixels in each direction from the source image rectangle to extrapolate.
     * The src and dist size can be got from input and output tensor.
     * For example, top=1, left=1, src_w=64, src_h=64, dist_w=66, dist_h=66 mean that it builds 1 pixel-wide border of top=left=bottom=right=1.
     * @param border_type border type. See NVCVBorderType for details.
     * @param value border value if borderType==BORDER_CONSTANT.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                    const NVCVBorderType border_type, const float4 value, cudaStream_t stream);

    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                    const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                    const NVCVBorderType border_type, const float4 value, cudaStream_t stream);

private:
    template<class OutType>
    ErrorCode inferWarp(const ImageBatchVarShapeDataStridedCuda &inData, const OutType &outData,
                        const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                        const NVCVBorderType border_type, const float4 value, cudaStream_t stream);
};

class RotateVarShape : public CudaBaseOp
{
public:
    RotateVarShape() = delete;

    explicit RotateVarShape(const int maxVarShapeBatchSize);

    ~RotateVarShape() override;

    /**
     * @brief Rotates input images around the origin (0,0) and then shifts it.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variables.
     * @param batch batch_size.
     * @param buffer_size buffer size of gpu_workspace and cpu_workspace
     * @param dsize size of the output images.
     * @param angle angle of rotation in degrees.
     * @param xShift shift along the horizontal axis.
     * @param yShift shift along the vertical axis.
     * @param interpolation interpolation method. Only INTER_NEAREST, INTER_LINEAR, and INTER_CUBIC are supported.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &angleDeg, const TensorDataStridedCuda &shift,
                    const NVCVInterpolationType interpolation, cudaStream_t stream);

private:
    float    *d_aCoeffs;
    const int m_maxBatchSize;
};

class Laplacian : public CudaBaseOp
{
public:
    Laplacian() = delete;

    Laplacian(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Calculates the Laplacian of an image.
     * @param inData Input Tensor
     * @param outData Output Tensor
     * @param ksize aperture size used to compute the second-derivative filters, it can be 1 or 3.
     * @param scale optional scale factor for the computed Laplacian values. By default, no scaling is applied.
     * @param borderMode pixel extrapolation method, e.g. \p NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int ksize, float scale,
                    NVCVBorderType borderMode, cudaStream_t stream);
};

class Gaussian : public CudaBaseOp
{
public:
    Gaussian() = delete;

    Gaussian(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize);

    ~Gaussian() override;

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blurs an image using a Gaussian filter.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param sigma Gaussian kernel standard deviation in X and Y directions.
     *              If sigma.y is zero or negative, use sigma.y = sigma.x.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Size2D kernelSize,
                    double2 sigma, NVCVBorderType borderMode, cudaStream_t stream);

private:
    Size2D  m_maxKernelSize = {0, 0};
    Size2D  m_curKernelSize = {0, 0};
    double2 m_curSigma      = {-1.0, -1.0};
    float  *m_kernel        = nullptr;
};

class AverageBlur : public CudaBaseOp
{
public:
    AverageBlur() = delete;

    AverageBlur(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize);

    ~AverageBlur() override;

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blur an image using an average kernel.
     * @param ksize average blur kernel size.
     * @param anchor anchor of the kernel that indicates the relative position of a filtered point within the kernel.
     * (-1,-1) means that the anchor is at the kernel center.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Size2D kernelSize,
                    int2 kernelAnchor, NVCVBorderType borderMode, cudaStream_t stream);

private:
    Size2D m_maxKernelSize = {0, 0};
};

class Conv2DVarShape : public CudaBaseOp
{
public:
    Conv2DVarShape() = delete;

    Conv2DVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Convolves an image with the kernel. The function does actually compute correlation, not the convolution
     * (same as opencv filter2D)
     * @param inputs gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outputs gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is
     * input_shape[i] and type is data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variable.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variable.
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param ksize convolution kernel size.
     * @param kernels convolution kernels. All the kernel values are flatted into a 1d array.
     * @param anchors anchor of the kernel that indicates the relative position of a filtered point within the kernel.
     * (-1,-1) means that the anchor is at the kernel center.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
                    NVCVBorderType borderMode, cudaStream_t stream);
};

class LaplacianVarShape : public CudaBaseOp
{
public:
    LaplacianVarShape() = delete;

    LaplacianVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Calculates the Laplacian of an image.
     * @param inData Input image batch var shape
     * @param outData Output image batch var shape
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param ksize aperture size used to compute the second-derivative filters
     * @param scale optional scale factor for the computed Laplacian values. By default, no scaling is applied.
     * @param borderMode pixel extrapolation method, e.g. nvcv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &ksize, const TensorDataStridedCuda &scale, NVCVBorderType borderMode,
                    cudaStream_t stream);
};

class GammaContrastVarShape : public CudaBaseOp
{
public:
    GammaContrastVarShape() = delete;

    GammaContrastVarShape(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount);

    ~GammaContrastVarShape() override;

    /**
     * @brief Adjust image contrast by scaling pixel values to 255*((v/255)**gamma)
     * @param inputs gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outputs gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is
     * input_shape[i] and type is data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variable.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variable.
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param gammas the gamma value for each image / image channel. If per_channel is true, the length of gammas should
     * be equal to batch * channel_size.
     * @param per_channel whether to use the same value for all channels.
     * @param input_shapes shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */

    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &gammas, cudaStream_t stream);

private:
    int    m_maxBatchSize    = 0;
    int    m_maxChannelCount = 0;
    float *m_gammaArray      = nullptr;
};

// Tensor (non-var-shape) GammaContrast. Supports interleaved (kNHWC/kHWC) and planar (kNCHW/kCHW)
// layouts. The gamma tensor is per-sample or per-sample-per-channel, normalized into a dense
// [numSamples*channels] array shared with the var-shape path so results are bit-exact.
class GammaContrast : public CudaBaseOp
{
public:
    GammaContrast() = delete;

    GammaContrast(const int32_t maxBatchSize, const int32_t maxChannelCount);

    ~GammaContrast() override;

    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                    const TensorDataStridedCuda &gammas, cudaStream_t stream);

    // Scalar (host-float) gamma/gain path: out = gain * in**gamma applied with a single gamma/gain for
    // all samples/channels. Uses no gamma scratch (the scalars are passed straight into the kernel).
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float gamma, float gain,
                    NVCVRoundMode roundMode, cudaStream_t stream);

private:
    int    m_maxBatchSize    = 0;
    int    m_maxChannelCount = 0;
    float *m_gammaArray      = nullptr;
};

class GaussianVarShape : public CudaBaseOp
{
public:
    GaussianVarShape() = delete;

    GaussianVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize, int maxBatchSize);

    ~GaussianVarShape() override;

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blurs each image using a Gaussian filter.
     * @param inData Input images.
     * @param outData Output images.
     * @param kernelSize Gaussian kernel size.
     * @param sigma Gaussian kernel standard deviation in X and Y directions.
     *              If sigma.y is zero or negative, use sigma.y = sigma.x.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &kernelSize, const TensorDataStridedCuda &sigma,
                    NVCVBorderType borderMode, cudaStream_t stream);

private:
    Size2D m_maxKernelSize = {0, 0};
    int    m_maxBatchSize  = 0;
    float *m_kernel        = nullptr;
};

class AverageBlurVarShape : public CudaBaseOp
{
public:
    AverageBlurVarShape() = delete;

    AverageBlurVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize, int maxBatchSize);

    ~AverageBlurVarShape() override;

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blur each image using an average filter.
     * @param inData Input images.
     * @param outData Output images.
     * @param kernelSize Average blur kernel size.
     *                     + Must be 1D tensor of int2, NVCV_DATA_TYPE_2S32
     * @param kernelAnchor Anchor of the kernel that indicates the relative position of a filtered point within the kernel.
     * (-1,-1) means that the anchor is at the kernel center.
     *                     + Must be 1D tensor of int2, NVCV_DATA_TYPE_2S32
     * @param borderMode pixel extrapolation method, e.g. nvcv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &kernelSize, const TensorDataStridedCuda &kernelAnchor,
                    NVCVBorderType borderMode, cudaStream_t stream);

private:
    Size2D m_maxKernelSize = {0, 0};
    int    m_maxBatchSize  = 0;
};

class MedianBlurVarShape : public CudaBaseOp
{
public:
    MedianBlurVarShape() = delete;
    explicit MedianBlurVarShape(const int maxVarShapeBatchSize);

    ~MedianBlurVarShape() override;
    /**
     * @brief Blur an image using a median kernel.
     * @param inputs gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outputs gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is
     * input_shape[i] and type is data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variable.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variable.
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param ksize median blur kernel size.
     * @param input_shapes shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                    const TensorDataStridedCuda &ksize, cudaStream_t stream);

private:
    const int        m_maxBatchSize;
    std::vector<int> m_kernelSizes;
};

class BilateralFilter : public CudaBaseOp
{
public:
    BilateralFilter() = delete;

    BilateralFilter(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief apply bilateral filter on images.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as data_type.
     * @param diameter pixel neighborhood diameter that is used during filtering
     * @param sigmaColor filter sigma in the color space
     * @param sigmaSpace filter sigma in the coordinate space
     * @param borderMode pixel extrapolation method, e.g. nvcv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int diameter,
                    float sigmaColor, float sigmaSpace, NVCVBorderType borderMode, cudaStream_t stream);
};

class BilateralFilterVarShape : public CudaBaseOp
{
public:
    BilateralFilterVarShape() = delete;

    BilateralFilterVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief apply bilateral filter on images.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as data_type.
     * @param diameterData tensor of each pixel neighborhood that is used during filtering
     * @param sigmaColorData tensor filter sigmas in the color space
     * @param sigmaSpaceData tensor filter sigmas in the coordinate space
     * @param borderMode pixel extrapolation method, e.g. nvcv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &diameterData, const TensorDataStridedCuda &sigmaColorData,
                    const TensorDataStridedCuda &sigmaSpaceData, NVCVBorderType borderMode, cudaStream_t stream);
};

class JointBilateralFilter : public CudaBaseOp
{
public:
    JointBilateralFilter() = delete;

    JointBilateralFilter(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief apply joint bilateral filter on pairs of images.
     * @param inData Tensor representing batch of images
     * @param inColorData Tensor representing batch of images for color distance
     * @param outData Tensor representing batch of output images
     * @param diameter pixel neighborhood diameter that is used during filtering
     * @param sigmaColor filter sigma in the color space
     * @param sigmaSpace filter sigma in the coordinate space
     * @param borderMode pixel extrapolation method, e.g. nvcv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &inColorData,
                    const TensorDataStridedCuda &outData, int diameter, float sigmaColor, float sigmaSpace,
                    NVCVBorderType borderMode, cudaStream_t stream);
};

class JointBilateralFilterVarShape : public CudaBaseOp
{
public:
    JointBilateralFilterVarShape() = delete;

    JointBilateralFilterVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief apply joint bilateral filter on pairs of images.
     * @param inData VarShape representing batch of images
     * @param inColorData VarShape representing batch of images for color distance
     * @param outData VarShape representing batch of output images
     * @param diameterData Tensor of each pixel neighborhood that is used during filtering
     * @param sigmaColorData Tensor filter sigmas in the color space
     * @param sigmaSpaceData Tensor filter sigmas in the coordinate space
     * @param borderMode pixel extrapolation method, e.g. nvcv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData,
                    const ImageBatchVarShapeDataStridedCuda &inColorData,
                    const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &diameterData,
                    const TensorDataStridedCuda &sigmaColorData, const TensorDataStridedCuda &sigmaSpaceData,
                    NVCVBorderType borderMode, cudaStream_t stream);
};

class OSD : public CudaBaseOp
{
public:
    OSD() = delete;

    OSD(DataShape max_input_shape, DataShape max_output_shape);

    ~OSD() override;

    /**
     * @brief Draw OSD elements onto input tensor, then return back output tensor.
     * @param inData Input tensor.
     * @param outData Output tensor.
     * @param elements OSD elements, \ref NVCVElements.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, NVCVElements elements,
                    cudaStream_t stream);

    /**
     * @brief Draw BndBox elements onto input tensor, then return back output tensor.
     * @param inData Input tensor.
     * @param outData Output tensor.
     * @param boxes Bounding box rectangle, \ref NVCVBndBoxesI.
     */
    ErrorCode inferBox(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, NVCVBndBoxesI bboxes,
                       cudaStream_t stream);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type) override;

private:
    nvcv::cuda::osd::cuOSDContext_t m_context;
};

class BoxBlur : public CudaBaseOp
{
public:
    BoxBlur() = delete;

    BoxBlur(DataShape max_input_shape, DataShape max_output_shape);

    ~BoxBlur() override;

    /**
     * @brief Converts an image from one color space to another.
     * @param inData Input tensor.
     * @param outData Output tensor.
     * @param boxes Bounding boxes to blur, \ref NVCVBlurBoxesI.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, NVCVBlurBoxesI bboxes,
                    cudaStream_t stream, bool skipCopy = false);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type) override;

private:
    nvcv::cuda::osd::cuOSDContext_t m_context;
};

class WarpAffine : public CudaBaseOp
{
public:
    WarpAffine() = delete;

    WarpAffine(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /*
     * @brief Applies an affine transformation to an image. Same function as nvcv::warpAffine.
     * @param inData input tensor.
     * @param outData output tensor.
     * @param xform cpu pointer, 2x3 transformation matrix.
     * @param flags Combination of interpolation methods(NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR or NVCV_INTERP_CUBIC)
                     and the optional flag NVCV_WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation.
     * @param borderMode pixel extrapolation method(NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE).
     * @param borderValue used in case of a constant border.
     * @param stream for the asynchronous execution.
    */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const float *xform,
                    const int32_t flags, const NVCVBorderType borderMode, const float4 borderValue,
                    cudaStream_t stream);
};

class WarpPerspective : public CudaBaseOp
{
public:
    WarpPerspective() = delete;

    WarpPerspective(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /*
     * @brief Applies a perspective transformation to an image. Same function as nvcv::warpPerspective.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param trans_matrix cpu pointer, 3×3 transformation matrix.
     * @param cpu_workspace cpu pointer, storage transformation matrix or inverse transformation matrix. It has the same
     * size as trans_matrix, e.g. 3x3.
     * @param dsize size of the output images.
     * @param flags Combination of interpolation methods(INTER_NEAREST or INTER_LINEAR) and the optional flag
     * WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation ( dst→src ).
     * @param borderMode pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).
     * @param borderValue used in case of a constant border.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const float *transMatrix,
                    const int32_t flags, const NVCVBorderType borderMode, const float4 borderValue,
                    cudaStream_t stream);
};

class WarpPerspectiveVarShape : public CudaBaseOp
{
public:
    WarpPerspectiveVarShape() = delete;

    explicit WarpPerspectiveVarShape(const int32_t maxBatchSize);

    ~WarpPerspectiveVarShape() override;

    /**
     * @brief Applies a perspective transformation to an image. Same function as nvcv::warpPerspective.
     * @param inputs gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outputs gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is dsize[i]
     * and type is data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variable.
     * @param cpu_workspace cpu pointer, storage transformation matrix or inverse transformation matrix. It has the same
     * size as trans_matrix, e.g. 3x3.
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param dsize cpu pointer, sizes of the output images.
     * @param trans_matrix cpu pointer, 3×3 transformation matrix.
     * @param flags Combination of interpolation methods(INTER_NEAREST or INTER_LINEAR) and the optional flag
     * WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation ( dst→src ).
     * @param borderMode pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).
     * @param borderValue used in case of a constant border.
     * @param input_shape cpu pointer, shapes of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     *
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                    const float4 borderValue, cudaStream_t stream);

private:
    const int m_maxBatchSize;
    float    *m_transformationMatrix = nullptr;
};

class WarpAffineVarShape : public CudaBaseOp
{
public:
    WarpAffineVarShape() = delete;

    explicit WarpAffineVarShape(const int32_t maxBatchSize);

    ~WarpAffineVarShape() override;
    /**
     * @brief Applies an affine transformation to an image. Same function as nvcv::warpAffine.
     * @param inputs gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outputs gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is dsize[i]
     * and type is data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variable.
     * @param cpu_workspace cpu pointer, storage transformation matrix or inverse transformation matrix. It has the same
     * size as trans_matrix, e.g. 3x3.
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param dsize cpu pointer, sizes of the output images.
     * @param trans_matrix cpu pointer, 2×3 transformation matrix.
     * @param flags Combination of interpolation methods(INTER_NEAREST or INTER_LINEAR) and the optional flag
     * WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation ( dst→src ).
     * @param borderMode pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).
     * @param borderValue used in case of a constant border.
     * @param input_shape cpu pointer, shapes of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                    const float4 borderValue, cudaStream_t stream);

private:
    const int m_maxBatchSize;
    float    *m_transformationMatrix = nullptr;
};

class Composite : public CudaBaseOp
{
public:
    Composite() = delete;

    Composite(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /*
     * @brief Composite perform the composite operation given a foreground, background and mat images
     *
     * @param foreground gpu tensor for foreground image
     *
     * @param background gpu tensor for background image
     *
     * @param fgMask gpu tensor for mat image
     *
     * @param outData gpu tensor for the output image
     *
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &foreground, const TensorDataStridedCuda &background,
                    const TensorDataStridedCuda &fgMask, const TensorDataStridedCuda &outData, cudaStream_t stream);
};

class CompositeVarShape : public CudaBaseOp
{
public:
    CompositeVarShape() = delete;

    CompositeVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Composite perform the composite operation given a foreground, background and mat images
     *
     * @param foreground gpu tensor for foreground image
     *
     * @param background gpu tensor for background image
     *
     * @param fgMask gpu tensor for foreground mask image
     *
     * @param outData gpu tensor for the output image
     *
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &forground,
                    const ImageBatchVarShapeDataStridedCuda &background,
                    const ImageBatchVarShapeDataStridedCuda &fgMask, const ImageBatchVarShapeDataStridedCuda &outData,
                    cudaStream_t stream);
};

class PillowResize : public CudaBaseOp
{
public:
    /**
     * @brief Resizes the input images. The function resize resizes the image down to or up to the specified size.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the same type as data_type. The output
     * sizes are derived from the dsize,fx, and fy.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param dsize size of the output images.if it equals zero, it is computed as:
     * @param interpolation interpolation method. See InterpolationMethods below
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     *
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                    const NVCVInterpolationType interpolation, cudaStream_t stream, const Workspace &workspace);

    NVCVWorkspaceRequirements getWorkspaceRequirements(DataShape max_input_shape, DataShape max_output_shape,
                                                       DataType max_data_type);
};

class PillowResizeVarShape : public CudaBaseOp
{
public:
    /**
     * @brief Resizes the input images. The function resize resizes the image down to or up to the specified size.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the same type as data_type. The output
     * sizes are derived from the dsize,fx, and fy.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variables.
     * @param batch batch_size.
     * @param buffer_size buffer size of gpu_workspace and cpu_workspace
     * @param dsize size of the output images.if it equals zero, it is computed as:
     * @param interpolation interpolation method. See cv::InterpolationFlags for more detials.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     *
     */
    ErrorCode infer(const ImageBatchVarShape &inData, const ImageBatchVarShape &outData,
                    const NVCVInterpolationType interpolation, cudaStream_t stream, const Workspace &workspace);

    NVCVWorkspaceRequirements getWorkspaceRequirements(DataShape max_input_shape, DataShape max_output_shape,
                                                       DataType max_data_type);
};

class AdaptiveThreshold : public CudaBaseOp
{
public:
    AdaptiveThreshold() = delete;

    AdaptiveThreshold(DataShape maxInputShape, DataShape maxOutputShape, int32_t maxBlockSize,
                      AdaptiveThresholdKernelPolicy kernelPolicy);

    ~AdaptiveThreshold() override;

    /**
     * @brief Applies an adaptive threshold to input images.
     * @param in gpu pointer, batched input images, whose shape is input_shape and type is data_type.
     * @param out gpu pointer, batched output images that have the size dsize and the same type as data_type.
     * @param maxValue Non-zero value assigned to the pixels for which the condition is satisfied.
     * @param adaptiveMethod Adaptive thresholding algorithm to use, see NVCVAdaptiveThresholdType. The BORDER_REPLICATE | BORDER_ISOLATED is used to process boundaries.
     * @param thresholdType Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV.
     * @param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
     * @param c Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &in, const TensorDataStridedCuda &out, const double maxValue,
                    const NVCVAdaptiveThresholdType adaptiveMethod, const NVCVThresholdType thresholdType,
                    const int32_t blockSize, const double c, cudaStream_t stream);

private:
    const int                           m_maxBlockSize;
    const AdaptiveThresholdKernelPolicy m_kernelPolicy;
    int                                 m_blockSize      = -1;
    int                                 m_adaptiveMethod = -1;
    float                              *m_kernel         = nullptr;
};

class AdaptiveThresholdVarShape : public CudaBaseOp
{
public:
    AdaptiveThresholdVarShape() = delete;

    AdaptiveThresholdVarShape(DataShape maxInputShape, DataShape maxOutputShape, int32_t maxBlockSize,
                              int32_t maxVarShapeBatchSize, AdaptiveThresholdKernelPolicy kernelPolicy);

    ~AdaptiveThresholdVarShape() override;

    /**
     * @brief Applies an adaptive threshold to input images.
     * @param in gpu pointer, in[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param out gpu pointer, out[i] is output image where i ranges from 0 to batch-1, whose size is
     * input_shape[i] and type is data_type.
     * @param maxValue Non-zero value assigned to the pixels for which the condition is satisfied.
     * @param adaptiveMethod Adaptive thresholding algorithm to use, see NVCVAdaptiveThresholdType. The BORDER_REPLICATE | BORDER_ISOLATED is used to process boundaries.
     * @param thresholdType Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV.
     * @param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
     * @param c Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                    const TensorDataStridedCuda &maxValue, const NVCVAdaptiveThresholdType adaptiveMethod,
                    const NVCVThresholdType thresholdType, const TensorDataStridedCuda &blockSize,
                    const TensorDataStridedCuda &c, cudaStream_t stream);

private:
    const int                           m_maxBatchSize;
    const int                           m_maxBlockSize;
    const AdaptiveThresholdKernelPolicy m_kernelPolicy;
    float                              *m_kernel = nullptr;
};

class RandomResizedCrop : public CudaBaseOp
{
public:
    using CudaBaseOp::calBufferSize;

    RandomResizedCrop() = delete;

    RandomResizedCrop(DataShape max_input_shape, DataShape max_output_shape, const double min_scale,
                      const double max_scale, const double min_ratio, const double max_ratio, int32_t maxBatchSize,
                      uint32_t seed);

    ~RandomResizedCrop() override;

    /**
     * @brief Resize and crop images
     * @param inData gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outData gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param interpolation the interpolation used in resize implementation
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                    const NVCVInterpolationType interpolation, cudaStream_t stream);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param batch_size input batch size
     */
    size_t calBufferSize(int batch_size);

protected:
    struct CropParamBuffers
    {
        float *scaleY;
        float *scaleX;
        int   *tops;
        int   *lefts;
    };

    int32_t          maxBatchSize() const noexcept;
    CropParamBuffers hostCropParams(int batch) noexcept;
    CropParamBuffers deviceCropParams(int batch) noexcept;
    std::byte       *hostCropParamStorage() noexcept;
    std::byte       *deviceCropParamStorage() noexcept;

    void getCropParams(int input_rows, int input_cols, int *top_indices, int *left_indices, int *crop_rows,
                       int *crop_cols);

private:
    double       min_scale_;
    double       max_scale_;
    double       min_ratio_;
    double       max_ratio_;
    std::mt19937 generator_;
    int32_t      m_maxBatchSize;
    std::byte   *m_cpuCropParams = nullptr;
    std::byte   *m_gpuCropParams = nullptr;
};

class RandomResizedCropVarShape : public RandomResizedCrop
{
public:
    using RandomResizedCrop::infer;

    RandomResizedCropVarShape() = delete;

    RandomResizedCropVarShape(DataShape max_input_shape, DataShape max_output_shape, const double min_scale,
                              const double max_scale, const double min_ratio, const double max_ratio,
                              int32_t maxBatchSize, uint32_t seed);

    /**
     * @brief Resize and crop images
     * @param inData gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outData gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is dsize[i]
     * and type is data_type.
     * @param interpolation the interpolation used in resize implementation
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ImageBatchVarShape &inData, const ImageBatchVarShape &outData,
                    const NVCVInterpolationType interpolation, cudaStream_t stream);
};

class Histogram : public CudaBaseOp
{
public:
    Histogram() = default;
    /**
     * @brief Resize and crop images
     * @param inData input tensor kNHWC/HWC tensor representing the input image(s)
     * @param mask mask tensor of the same size as the input image(s). Only non-zero values are counted for histogram.
     * @param histogram output tensor of size HWC representing the histogram where each row is an image histogram.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const TensorDataStridedCuda &inData, OptionalTensorConstRef mask,
                    const TensorDataStridedCuda &histogram, cudaStream_t stream);
};

class Inpaint : public CudaBaseOp
{
public:
    Inpaint() = delete;

    Inpaint(DataShape max_input_shape, DataShape max_output_shape, int maxBatchSize, Size2D maxShape);

    ~Inpaint() override;

    /**
    * @brief Restores the selected region in an image using the region neighborhood. TELEA algorithm is used here.
    * @param inputs gpu pointer, batched input images.
    * @param masks gpu pointer, batched inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted.
    * @param outputs gpu pointer, batched output images that have the same type.
    * @param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
    * @param stream for the asynchronous execution.
    */
    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &masks,
                    const TensorDataStridedCuda &outData, double inpaintRadius, cudaStream_t stream);

private:
    bool     m_init_dilate = false; // whether kernel is initialized
    int      m_maxBatchSize;
    uint8_t *m_kernel_ptr;
    uint8_t *m_workspace;
};

class InpaintVarShape : public CudaBaseOp
{
public:
    InpaintVarShape() = delete;

    InpaintVarShape(DataShape max_input_shape, DataShape max_output_shape, int maxBatchSize, Size2D maxShape);

    ~InpaintVarShape() override;

    /**
    * @brief Restores the selected region in an image using the region neighborhood. TELEA algorithm is used here.
    * @param inputs gpu pointer, batched input images.
    * @param masks gpu pointer, batched inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted.
    * @param outputs gpu pointer, batched output images that have the same type.
    * @param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
    * @param stream for the asynchronous execution.
    */
    ErrorCode infer(const ImageBatchVarShape &inBatch, const ImageBatchVarShapeDataStridedCuda &masks,
                    const ImageBatchVarShape &outBatch, double inpaintRadius, cudaStream_t stream);

private:
    bool     m_init_dilate = false; // whether kernel is initialized
    int      m_maxBatchSize;
    uint8_t *m_kernel_ptr;
    uint8_t *m_workspace;
};

class HistogramEq : public CudaBaseOp
{
public:
    HistogramEq() = delete;

    explicit HistogramEq(int maxBatchSize);

    ~HistogramEq() override;

    ErrorCode infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, cudaStream_t stream);

private:
    int        m_maxBatchSize;
    int        m_maxChannelCount;
    int        m_sizeOfHisto;
    std::byte *m_histoArray;
};

class HistogramEqVarShape : public CudaBaseOp
{
public:
    HistogramEqVarShape() = delete;

    explicit HistogramEqVarShape(int maxBatchSize);

    ~HistogramEqVarShape() override;

    ErrorCode infer(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    cudaStream_t stream);

private:
    int        m_maxBatchSize;
    int        m_maxChannelCount;
    int        m_sizeOfHisto;
    std::byte *m_histoArray;
};

} // namespace nvcv::legacy::cuda_op

#endif // CV_CUDA_LEGACY_H
