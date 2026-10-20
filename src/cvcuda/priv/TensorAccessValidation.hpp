/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_PRIV_TENSOR_ACCESS_VALIDATION_HPP
#define CVCUDA_PRIV_TENSOR_ACCESS_VALIDATION_HPP

#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstdint>
#include <vector>

namespace cvcuda::priv {

// exportData() returns an empty optional for a tensor or batch that is not CUDA-accessible and
// pitch-linear; these throw ERROR_INVALID_ARGUMENT with a message prefixed by `name`.
inline nvcv::Optional<nvcv::TensorDataStridedCuda> RequireCudaTensor(const nvcv::Tensor &tensor, const char *name)
{
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    if (!data)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be cuda-accessible, pitch-linear tensor",
                              name);
    }
    return data;
}

inline nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda> RequireCudaVarShape(
    const nvcv::ImageBatchVarShape &batch, cudaStream_t stream, const char *name)
{
    auto data = batch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!data)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s must be cuda-accessible, varshape pitch-linear image batch", name);
    }
    return data;
}

// Create() returns nullopt for a CUDA-accessible tensor that is not image-shaped. Callers then read
// numRows()/numCols() straight off the optional, so a release build would dereference nothing.
// Templated on the optional: Create() hands back nvcv::Optional, not std::optional.
template<class OptionalAccess>
inline const nvcv::TensorDataAccessStridedImagePlanar &RequireImageAccess(const OptionalAccess &access,
                                                                          const char           *name)
{
    if (!access)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be an image-shaped tensor", name);
    }
    return *access;
}

// The tensor wraps index with a 32-bit stride type, which is the whole reason the tensor paths are
// cheaper than a 64-bit addressing mode. Division rather than multiplication: sampleStride() *
// numSamples() can itself overflow int64_t for a bogus descriptor and wrap past the limit.
inline bool FitsInt32Addressing(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    constexpr int64_t kMax    = nvcv::cuda::TypeTraits<int32_t>::max;
    const int64_t     samples = access.numSamples();
    const int64_t     stride  = access.sampleStride();

    return samples <= 0 || stride <= kMax / samples;
}

inline void RequireInt32Addressable(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    if (!FitsInt32Addressing(access))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input size exceeds %d. Tensor is too large.",
                              nvcv::cuda::TypeTraits<int32_t>::max);
    }
}

inline void RequireInt32Addressable(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                    const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    if (!FitsInt32Addressing(inAccess) || !FitsInt32Addressing(outAccess))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.",
                              nvcv::cuda::TypeTraits<int32_t>::max);
    }
}

// Sizing the launch from the output while sampling the input turns a shape mismatch into an
// out-of-bounds read, so the filter paths require the extents to agree up front.
inline void RequireSameImageShape(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                  const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    if (inAccess.numSamples() != outAccess.numSamples() || inAccess.numRows() != outAccess.numRows()
        || inAccess.numCols() != outAccess.numCols() || inAccess.numChannels() != outAccess.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same shape, got %dx%dx%dx%d and %dx%dx%dx%d",
                              (int)inAccess.numSamples(), (int)inAccess.numRows(), (int)inAccess.numCols(),
                              (int)inAccess.numChannels(), (int)outAccess.numSamples(), (int)outAccess.numRows(),
                              (int)outAccess.numCols(), (int)outAccess.numChannels());
    }
}

// A var-shape submit reads one parameter per image straight out of a device tensor via
// Tensor1DWrap, so a mismatched element reinterprets the bits and a short tensor reads past the end.
//
// Two spellings reach here and both are byte-identical to the wrap: a rank-1 tensor of the packed
// dtype (TYPE_2S32), which the C++ callers use, and a rank-2 (N, components) tensor of the matching
// scalar dtype laid out "NC", which the Python bindings build. Accept either.
inline void RequirePerImageParam(const nvcv::TensorDataStridedCuda &data, nvcv::DataType packed, int64_t numImages,
                                 const char *name)
{
    const nvcv::DataType dtype      = data.dtype();
    const int            components = packed.numChannels();

    const bool packedForm = data.rank() == 1 && dtype == packed;

    const bool splitForm = data.rank() == 2 && data.shape(1) == components && dtype.numChannels() == 1
                        && dtype.dataKind() == packed.dataKind()
                        && dtype.bitsPerChannel()[0] == packed.bitsPerChannel()[0];

    if (!packedForm && !splitForm)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s must be a rank-1 %s tensor or a rank-2 (N, %d) tensor of its element type, got "
                              "rank-%d %s",
                              name, nvcvDataTypeGetName(packed), components, (int)data.rank(),
                              nvcvDataTypeGetName(dtype));
    }

    if (data.shape(0) < numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must provide one entry per image (%ld)", name,
                              (long)numImages);
    }
}

// Owns one reference per image handle for the lifetime of a validation pass.
//
// nvcvImageBatchVarShapeGetImages hands back a new reference for every handle it writes, so they
// have to be released individually; the vector is value-initialised so a throw part-way through the
// fetch still leaves the destructor with nulls rather than garbage.
class ImageHandleBatch
{
public:
    ImageHandleBatch(NVCVImageBatchHandle batch, int32_t numImages)
        : m_handles(numImages > 0 ? static_cast<size_t>(numImages) : 0u)
    {
        if (numImages > 0)
        {
            nvcv::detail::CheckThrow(nvcvImageBatchVarShapeGetImages(batch, 0, m_handles.data(), numImages));
        }
    }

    ~ImageHandleBatch()
    {
        for (NVCVImageHandle handle : m_handles)
        {
            if (handle != nullptr)
            {
                // A destructor cannot throw, and there is nothing to recover from if a release of a
                // handle we were just handed fails.
                (void)nvcvImageDecRef(handle, nullptr);
            }
        }
    }

    ImageHandleBatch(const ImageHandleBatch &)            = delete;
    ImageHandleBatch &operator=(const ImageHandleBatch &) = delete;

    nvcv::Size2D size(int32_t index) const
    {
        int32_t width = 0, height = 0;
        nvcv::detail::CheckThrow(nvcvImageGetSize(m_handles[index], &width, &height));
        return {width, height};
    }

    nvcv::ImageData exportData(int32_t index) const
    {
        nvcv::ImageData data;
        nvcv::detail::CheckThrow(nvcvImageExportData(m_handles[index], &data.cdata()));
        return data;
    }

private:
    std::vector<NVCVImageHandle> m_handles;
};

// Input/output agreement checks shared by operators that pair a source with a destination.
inline void RequireSameDtype(const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out)
{
    if (in.dtype() != out.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }
}

inline void RequireSameLayout(const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out)
{
    if (in.layout() != out.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }
}

inline void RequireUniqueFormat(const nvcv::ImageBatchVarShapeDataStridedCuda &batch, const char *side)
{
    if (!batch.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the %s batch must all have the same format", side);
    }
}

inline void RequireSameFormat(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                              const nvcv::ImageBatchVarShapeDataStridedCuda &out)
{
    if (in.uniqueFormat() != out.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }
}

inline void RequireSameNumImages(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &out)
{
    if (in.numImages() != out.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images, got %d and %d", in.numImages(),
                              out.numImages());
    }
}

// A var-shape submit has to prove input and output agree image by image: a mismatch leaves part of
// the output unwritten on one dispatch path and blurs against the border on the other. The sizes
// live in device memory, so host code can only reach them through the image handles.
//
// The obvious spelling -- `in[i].size() != out[i].size()` -- fetches one handle per call and wraps
// each in a refcounted nvcv::Image, twice per image. On a 72-image batch that is ~9 us of host time
// per submit, which measured as a 2% GPU-idle gap on a 419 us Gaussian call: the kernels were at
// parity with the legacy layer and the whole difference was this loop. The C API takes a count and
// nvcvImageGetSize reads a size straight off a handle, so fetch once per batch and skip the wrappers.
inline void RequireSameVarShapeImageSizes(const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out)
{
    const int32_t numImages = in.numImages();

    ImageHandleBatch inHandles(in.handle(), numImages);
    ImageHandleBatch outHandles(out.handle(), numImages);

    for (int32_t i = 0; i < numImages; ++i)
    {
        const nvcv::Size2D inSize = inHandles.size(i), outSize = outHandles.size(i);

        if (inSize != outSize)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and Output image %d must have the same size, got %dx%d and %dx%d", i, inSize.w,
                                  inSize.h, outSize.w, outSize.h);
        }
    }
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_TENSOR_ACCESS_VALIDATION_HPP
