/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor ResizeCropConvertReformatInto(Tensor &dst, Tensor &src, const std::tuple<int, int> resizeDim,
                                     NVCVInterpolationType interp, const std::tuple<int, int> cropPos,
                                     const NVCVChannelManip manip, const float scale, const float offset,
                                     const bool srcCast, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvcuda::ResizeCropConvertReformat>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {src});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*resize});

    nvcv::Size2D size_wh{std::get<0>(resizeDim), std::get<1>(resizeDim)};
    int2         crop_xy{std::get<0>(cropPos), std::get<1>(cropPos)};

    guard.run(
        [&resize, &pstream, &src, &dst, &size_wh, &interp, &crop_xy, &manip, &scale, &offset, &srcCast]()
        { resize->submit(pstream->cudaHandle(), src, dst, size_wh, interp, crop_xy, manip, scale, offset, srcCast); });

    return std::move(dst);
}

Tensor ResizeCropConvertReformat(Tensor &src, const std::tuple<int, int> resizeDim, NVCVInterpolationType interp,
                                 const NVCVRectI cropRect, const char *layout, nvcv::DataType dataType,
                                 const NVCVChannelManip manip, const float scale, const float offset,
                                 const bool srcCast, std::optional<Stream> pstream)
{
    nvcv::TensorLayout srcLayout = src.layout();

    if (srcLayout != NVCV_TENSOR_HWC && srcLayout != NVCV_TENSOR_NHWC && srcLayout != NVCV_TENSOR_CHW
        && srcLayout != NVCV_TENSOR_NCHW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT,
                              "Input tensor must have layout 'HWC', 'NHWC', 'CHW', or 'NCHW'.");
    }

    nvcv::TensorLayout dstLayout = (layout && *layout ? nvcv::TensorLayout(layout) : nvcv::TensorLayout(""));

    if (dstLayout.rank() == 0)
    {
        dstLayout = srcLayout;
    }

    if (dstLayout.rank() != srcLayout.rank())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT,
                              "Output tensor rank must match input tensor rank.");
    }
    if (dstLayout != NVCV_TENSOR_HWC && dstLayout != NVCV_TENSOR_NHWC && dstLayout != NVCV_TENSOR_CHW
        && dstLayout != NVCV_TENSOR_NCHW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT,
                              "Output tensor must have layout 'HWC', 'NHWC', 'CHW', or 'NCHW'.");
    }

    nvcv::TensorShape srcShape = Permute(src.shape(), nvcv::TensorLayout{NVCV_TENSOR_NHWC});

    nvcv::TensorShape::ShapeType shape = srcShape.shape();

    shape[2] = cropRect.width;
    shape[1] = cropRect.height;

    nvcv::TensorShape dstShape = Permute(nvcv::TensorShape(shape, nvcv::TensorLayout{NVCV_TENSOR_NHWC}), dstLayout);

    Tensor dst = Tensor::Create(dstShape, dataType);

    const std::tuple<int, int> cropPos = std::make_tuple(cropRect.x, cropRect.y);

    return ResizeCropConvertReformatInto(dst, src, resizeDim, interp, cropPos, manip, scale, offset, srcCast, pstream);
}

Tensor ResizeCropConvertReformatVarShapeInto(Tensor &dst, ImageBatchVarShape &src, const std::tuple<int, int> resizeDim,
                                             NVCVInterpolationType interp, const std::tuple<int, int> cropPos,
                                             const NVCVChannelManip manip, const float scale, const float offset,
                                             const bool srcCast, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvcuda::ResizeCropConvertReformat>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {src});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*resize});

    nvcv::Size2D size_wh(std::get<0>(resizeDim), std::get<1>(resizeDim));
    int2         crop_xy{std::get<0>(cropPos), std::get<1>(cropPos)};

    guard.run(
        [&resize, &pstream, &src, &dst, &size_wh, &interp, &crop_xy, &manip, &scale, &offset, &srcCast]()
        { resize->submit(pstream->cudaHandle(), src, dst, size_wh, interp, crop_xy, manip, scale, offset, srcCast); });

    return std::move(dst);
}

Tensor ResizeCropConvertReformatVarShape(ImageBatchVarShape &src, const std::tuple<int, int> resizeDim,
                                         NVCVInterpolationType interp, const NVCVRectI cropRect, const char *layout,
                                         nvcv::DataType dataType, const NVCVChannelManip manip, const float scale,
                                         const float offset, const bool srcCast, std::optional<Stream> pstream)
{
    const nvcv::ImageFormat srcFrmt = src.uniqueFormat();
    if (!srcFrmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have same format across all images.");
    }

    nvcv::TensorLayout dstLayout = (layout && *layout ? nvcv::TensorLayout(layout) : nvcv::TensorLayout(""));

    int channels = srcFrmt.numChannels();
    int images   = src.numImages();

    if (channels != 1 && channels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have 1 or 3 channels.");
    }

    if (srcFrmt != nvcv::FMT_RGB8 && srcFrmt != nvcv::FMT_BGR8 && srcFrmt != nvcv::FMT_RGB8p
        && srcFrmt != nvcv::FMT_BGR8p && srcFrmt != nvcv::FMT_Y8 && srcFrmt != nvcv::FMT_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must have 8-bit channels in RGB, BGR, Y8, or U8 format.");
    }

    nvcv::TensorShape shape;

    if (dstLayout.rank() == 0)
    {
        shape = nvcv::TensorShape{
            {          images, cropRect.height, cropRect.width, channels},
            nvcv::TensorLayout{NVCV_TENSOR_NHWC                }
        };
    }
    else
    {
        if (dstLayout == NVCV_TENSOR_NHWC || dstLayout == NVCV_TENSOR_HWC || dstLayout == NVCV_TENSOR_NHW
            || dstLayout == NVCV_TENSOR_HW)
            shape = nvcv::TensorShape{
                {          images, cropRect.height, cropRect.width, channels},
                nvcv::TensorLayout{NVCV_TENSOR_NHWC                }
            };
        else if (dstLayout == NVCV_TENSOR_NCHW || dstLayout == NVCV_TENSOR_CHW)
            shape = nvcv::TensorShape{
                {          images, channels, cropRect.height, cropRect.width},
                nvcv::TensorLayout{NVCV_TENSOR_NCHW         }
            };
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Destination layout must be 'HWC', 'NHWC', 'NHW', 'HW', 'NCHW', or 'CHW'.");
        }
    }

    Tensor dst = Tensor::Create(shape, dataType);

    const std::tuple<int, int> cropPos = std::make_tuple(cropRect.x, cropRect.y);

    return ResizeCropConvertReformatVarShapeInto(dst, src, resizeDim, interp, cropPos, manip, scale, offset, srcCast,
                                                 pstream);
}

} // namespace

void ExportOpResizeCropConvertReformat(py::module &m)
{
    using namespace pybind11::literals;

    m.def("resize_crop_convert_reformat", NvtxTrace("cvcuda.resize_crop_convert_reformat", &ResizeCropConvertReformat),
          "src"_a, "resize_dim"_a, "interp"_a, "crop_rect"_a, py::kw_only(), "layout"_a = "",
          "data_type"_a = NVCV_DATA_TYPE_NONE, "manip"_a = NVCV_CHANNEL_NO_OP, "scale"_a = 1.0, "offset"_a = 0.0,
          "srcCast"_a = true, "stream"_a = nullptr,
          R"pbdoc(
        Executes the ResizeCropConvertReformat operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            resize_dim (tuple[int,int]): Dimensions, width & height, of resized tensor (prior to cropping).
            interp (cvcuda.Interp): Interpolation type used for resizing. Currently, only cvcuda.Interp.NEAREST and
                                    cvcuda.Interp.LINEAR are available.
            crop_rect (cvcuda.RectI): Crop rectangle, (left, top, width, height), specifying the top-left corner and
                                   width & height dimensions of the region to crop from the resized images.
            layout(string, optional): String specifying output tensor layout (e.g., 'NHWC' or 'CHW'). Empty string
                                      (default) indicates output tensor layout copies input.
            data_type(cvcuda.Type, optional): Data type of output tensor channel (e.g., uint8 or float). 0 (default)
                                           indicates output tensor data type copies input.
            manip(cvcuda.ChannelManip, optional): Channel manipulation (e.g., shuffle RGB to BGR). NO_OP (default)
                                                  indicates output tensor channels are unchanged.
            scale(float, optional): Scale (i.e., multiply) the output values by this amount. 1.0 (default) results
                                    in no scaling of the output values.
            offset(float, optional): Offset (i.e., add to) the output values by this amount. This is applied after
                                     scaling. Let v be a resized and cropped value, then v * scale + offset is final
                                     output value. 0.0 (default) results in no offset being added to the output.
            srcCast(bool, optional): Boolean indicating whether or not the resize interpolation results are re-cast
                                     back to the input (or source) data type. Refer to the C API reference for more
                                     information. True (default) re-cast resize interpolation results back to the
                                     source data type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("resize_crop_convert_reformat_into",
          NvtxTrace("cvcuda.resize_crop_convert_reformat_into", &ResizeCropConvertReformatInto), "dst"_a, "src"_a,
          "resize_dim"_a, "interp"_a, "cropPos"_a, py::kw_only(), "manip"_a = NVCV_CHANNEL_NO_OP, "scale"_a = 1.0,
          "offset"_a = 0.0, "srcCast"_a = true, "stream"_a = nullptr, R"pbdoc(
        Executes the ResizeCropConvertReformat operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation. Output tensor also specifies the
                               crop dimensions (i.e., width & height), as well as the output data type (e.g., uchar3
                               or float) and tensor layout (e.g., 'NHWC' or 'NCHW').
            src (cvcuda.Tensor): Input tensor containing one or more images.
            resize_dim (tuple[int,int]): Dimensions, width & height, of resized tensor (prior to cropping).
            interp (cvcuda.Interp): Interpolation type used for resizing. Currently, only cvcuda.Interp.NEAREST and
                                    cvcuda.Interp.LINEAR are available.
            cropPos (tuple[int,int]): Crop position, (x, y), specifying the top-left corner of the region to crop
                                      from the resized images. The crop region's width and height is specified by the
                                      output tensor's width & height.
            manip(cvcuda.ChannelManip, optional): Channel manipulation (e.g., shuffle RGB to BGR). NO_OP (default)
                                                  indicates output tensor channels are unchanged.
            scale(float, optional): Scale (i.e., multiply) the output values by this amount. 1.0 (default) results
                                    in no scaling of the output values.
            offset(float, optional): Offset (i.e., add to) the output values by this amount. This is applied after
                                     scaling. Let v be a resized and cropped value, then v * scale + offset is final
                                     output value. 0.0 (default) results in no offset being added to the output.
            srcCast(bool, optional): Boolean indicating whether or not the resize interpolation results are re-cast
                                     back to the input (or source) data type. Refer to the C API reference for more
                                     information. True (default) re-cast resize interpolation results back to the
                                     source data type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("resize_crop_convert_reformat",
          NvtxTrace("cvcuda.resize_crop_convert_reformat", &ResizeCropConvertReformatVarShape), "src"_a, "resize_dim"_a,
          "interp"_a, "crop_rect"_a, py::kw_only(), "layout"_a = "", "data_type"_a = NVCV_DATA_TYPE_NONE,
          "manip"_a = NVCV_CHANNEL_NO_OP, "scale"_a = 1.0, "offset"_a = 0.0, "srcCast"_a = true, "stream"_a = nullptr,
          R"pbdoc(
        Executes the ResizeCropConvertReformat operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images of varying sizes, but all images
                                      must have the same data type, channels, and layout.
            resize_dim (tuple[int,int]): Dimensions, width & height, of resized tensor (prior to cropping).
            interp (cvcuda.Interp): Interpolation type used for resizing. Currently, only cvcuda.Interp.NEAREST and
                                    cvcuda.Interp.LINEAR are available.
            crop_rect (cvcuda.RectI): Crop rectangle, (left, top, width, height), specifying the top-left corner and
                                   width & height dimensions of the region to crop from the resized images.
            layout(string, optional): String specifying output tensor layout (e.g., 'NHWC' or 'CHW'). Empty string
                                      (default) indicates output tensor layout copies input.
            data_type(cvcuda.Type, optional): Data type of output tensor channel (e.g., uint8 or float). 0 (default)
                                           indicates output tensor data type copies input.
            manip(cvcuda.ChannelManip, optional): Channel manipulation (e.g., shuffle RGB to BGR). NO_OP (default)
                                                  indicates output tensor channels are unchanged.
            scale(float, optional): Scale (i.e., multiply) the output values by this amount. 1.0 (default) results
                                    in no scaling of the output values.
            offset(float, optional): Offset (i.e., add to) the output values by this amount. This is applied after
                                     scaling. Let v be a resized and cropped value, then v * scale + offset is final
                                     output value. 0.0 (default) results in no offset being added to the output.
            srcCast(bool, optional): Boolean indicating whether or not the resize interpolation results are re-cast
                                     back to the input (or source) data type. Refer to the C API reference for more
                                     information. True (default) re-cast resize interpolation results back to the
                                     source data type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("resize_crop_convert_reformat_into",
          NvtxTrace("cvcuda.resize_crop_convert_reformat_into", &ResizeCropConvertReformatVarShapeInto), "dst"_a,
          "src"_a, "resize_dim"_a, "interp"_a, "cropPos"_a, py::kw_only(), "manip"_a = NVCV_CHANNEL_NO_OP,
          "scale"_a = 1.0, "offset"_a = 0.0, "srcCast"_a = true, "stream"_a = nullptr, R"pbdoc(
        Executes the ResizeCropConvertReformat operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation. Output tensor also specifies the
                               crop dimensions (i.e., width & height), as well as the output data type (e.g., uchar3
                               or float) and tensor layout (e.g., 'NHWC' or 'NCHW').
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images of varying sizes, but all images
                                      must have the same data type, channels, and layout.
            resize_dim (tuple[int,int]): Dimensions, width & height, of resized tensor (prior to cropping).
            interp (cvcuda.Interp): Interpolation type used for resizing. Currently, only cvcuda.Interp.NEAREST and
                                    cvcuda.Interp.LINEAR are available.
            cropPos (tuple[int,int]): Crop position, (x, y), specifying the top-left corner of the region to
                                      crop from the resized images. The crop region's width and height is specified by
                                      the output tensor's width & height.
            manip(cvcuda.ChannelManip, optional): Channel manipulation (e.g., shuffle RGB to BGR). NO_OP (default)
                                                  indicates output tensor channels are unchanged.
            scale(float, optional): Scale (i.e., multiply) the output values by this amount. 1.0 (default) results
                                    in no scaling of the output values.
            offset(float, optional): Offset (i.e., add to) the output values by this amount. This is applied after
                                     scaling. Let v be a resized and cropped value, then v * scale + offset is final
                                     output value. 0.0 (default) results in no offset being added to the output.
            srcCast(bool, optional): Boolean indicating whether or not the resize interpolation results are re-cast
                                     back to the input (or source) data type. Refer to the C API reference for more
                                     information. True (default) re-cast resize interpolation results back to the
                                     source data type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
