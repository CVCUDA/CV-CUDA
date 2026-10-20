/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_PYTHON_CHANNEL_COPY_UTILS_HPP
#define CVCUDA_PYTHON_CHANNEL_COPY_UTILS_HPP

#include "NativeBinding.hpp"
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>

#include <array>
#include <memory>
#include <vector>

namespace cvcudapy::channelcopy {

inline int ChannelAxis(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_HWC || layout == nvcv::TENSOR_NHWC)
        return layout.rank() - 1;
    if (layout == nvcv::TENSOR_CHW || layout == nvcv::TENSOR_NCHW)
        return layout.rank() - 3;
    throw py::value_error("Expected HWC, NHWC, CHW or NCHW layout");
}

inline int CheckImage(const native::Tensor &tensor)
{
    const auto &image   = tensor.impl();
    const int   channel = ChannelAxis(image.layout());
    if (const auto dtype = image.dtype(); dtype != nvcv::TYPE_U8 && dtype != nvcv::TYPE_S8 && dtype != nvcv::TYPE_U16
                                          && dtype != nvcv::TYPE_S16 && dtype != nvcv::TYPE_U32
                                          && dtype != nvcv::TYPE_S32 && dtype != nvcv::TYPE_F16
                                          && dtype != nvcv::TYPE_F32 && dtype != nvcv::TYPE_F64)
        throw py::value_error("Unsupported scalar dtype");
    for (int dimension = 0; dimension < image.rank(); ++dimension)
        if (image.shape()[dimension] <= 0)
            throw py::value_error("Tensor dimensions must be positive");
    if (image.shape()[channel] < 1 || image.shape()[channel] > 4)
        throw py::value_error("Expected one to four channels");
    return channel;
}

inline nvcv::TensorShape WithChannels(const nvcv::TensorShape &shape, int64_t channels)
{
    std::array<int64_t, 4> dimensions{};
    for (int dimension = 0; dimension < shape.rank(); ++dimension) dimensions[dimension] = shape[dimension];
    dimensions[ChannelAxis(shape.layout())] = channels;
    return nvcv::TensorShape(dimensions.data(), shape.rank(), shape.layout());
}

inline void ValidateBands(const std::vector<std::shared_ptr<native::Tensor>> &tensors)
{
    if (tensors.empty() || tensors.size() > 4)
        throw py::value_error("Expected one to four band tensors");
    for (const auto &tensor : tensors)
    {
        if (!tensor)
            throw py::type_error("Band tensor sequence cannot contain None");
        if (const int channel = CheckImage(*tensor); tensor->impl().shape()[channel] != 1
                                                     || tensor->impl().shape() != tensors.front()->impl().shape()
                                                     || tensor->dtype() != tensors.front()->dtype())
            throw py::value_error("Bands must have identical shapes, layouts and dtypes, with one channel each");
    }
}

inline std::shared_ptr<native::TensorBatch> MakeBands(const std::vector<std::shared_ptr<native::Tensor>> &tensors)
{
    ValidateBands(tensors);
    auto batch = native::TensorBatch::Create(static_cast<int>(tensors.size()));
    for (const auto &tensor : tensors) batch->pushBack(*tensor);
    return batch;
}

inline nvcv::ImageFormat CheckImage(const native::ImageBatchVarShape &batch)
{
    const auto format = GetUniqueFormat(batch);
    if (!format)
        throw py::value_error("Expected a nonempty image batch with a uniform format");
    if (format->numChannels() < 1 || format->numChannels() > 4)
        throw py::value_error("Expected one to four channels");
    return *format;
}

inline void ValidateBands(const std::vector<std::shared_ptr<native::ImageBatchVarShape>> &batches)
{
    if (batches.empty() || batches.size() > 4)
        throw py::value_error("Expected one to four band batches");
    for (const auto &batch : batches)
        if (!batch)
            throw py::type_error("Band batch sequence cannot contain None");
}

inline std::vector<nvcv::ImageBatchVarShape> MakeBands(
    const std::vector<std::shared_ptr<native::ImageBatchVarShape>> &batches)
{
    ValidateBands(batches);
    std::vector<nvcv::ImageBatchVarShape> bands;
    bands.reserve(batches.size());
    for (const auto &batch : batches)
    {
        if (const auto format = CheckImage(*batch); format.numChannels() != 1 || format.numPlanes() != 1)
            throw py::value_error("Bands must have one plane and one channel");
        bands.push_back(batch->impl());
    }
    return bands;
}

inline std::shared_ptr<native::ImageBatchVarShape> CreateImageBatch(const native::ImageBatchVarShape &input,
                                                                    nvcv::ImageFormat format, native::Stream &stream)
{
    std::vector<native::ImageBatchVarShape::OutputImage> images;
    images.reserve(input.numImages());
    for (const auto &image : input) images.push_back({image->size(), format});
    return native::ImageBatchVarShape::CreateForOutput(input.capacity(), images, stream);
}

#ifdef NB_FREE_THREADED
inline std::vector<nvcvpy::priv::BatchMutex *> BatchMutexes(
    const std::vector<std::shared_ptr<native::ImageBatchVarShape>> &bands, native::ImageBatchVarShape *image = nullptr)
{
    std::vector<nvcvpy::priv::BatchMutex *> mutexes;
    mutexes.reserve(bands.size() + 1);
    if (image)
        mutexes.push_back(&image->batchMutex());
    for (const auto &band : bands) mutexes.push_back(&band->batchMutex());
    return mutexes;
}
#endif

} // namespace cvcudapy::channelcopy

#endif
