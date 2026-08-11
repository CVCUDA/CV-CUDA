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

#include "../WorkspaceCache.hpp"
#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpPillowResize.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ImageFormat.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class PillowResizeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Specialized class for cvcuda::PillowResize operator with a better cache Key.
// It allows for reusing an existing operator object from cache if its payload size is >= the required size.
// It also allows to fetch the biggest payload object to be reused while removing all others.
// This is more flexible than using the generic PyOperator class and its Key class.
class PyOpPillowResize : public nvcvpy::Container // NOSONAR: operator wrappers share the Python cache hierarchy.
{
public:
    // Define a Key class to be used by the cache to fetch similar items for potential reuse.
    class Key : public nvcvpy::IKey
    {
    public:
        // Arguments of the key constructor should match the corresponding cvcuda operator arguments.
        Key() = default;

        size_t payloadSize() const
        {
            return 0;
        }

    private:
        size_t doGetHash() const override
        {
            return 0;
        }

        // The comparison of keys is based on the payload size, the one in the cache is "that" key.
        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            return dynamic_cast<const Key *>(&that_) != nullptr;
        }
    };

    // Constructor instantiate the cache key and the operator object.
    PyOpPillowResize() = default;

    inline void submit(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, nvcv::ImageFormat format,
                       NVCVInterpolationType interpolation) const
    {
        int          batch_size = getBatchSize(in);
        nvcv::Size2D in_size    = imageSize(in);
        nvcv::Size2D out_size   = imageSize(out);

        auto req = m_op.getWorkspaceRequirements(batch_size, in_size, out_size, format);
        auto ws  = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, interpolation);
    }

    static int getBatchSize(const nvcv::Tensor &tensor)
    {
        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensor.exportData());
        if (!access)
            throw PillowResizeError("Incompatible tensor layout");

        return static_cast<int>(access->numSamples());
    }

    static nvcv::Size2D imageSize(const nvcv::Tensor &tensor)
    {
        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensor.exportData());
        if (!access)
            throw PillowResizeError("Incompatible tensor layout");

        return access->size();
    }

    inline void submit(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                       const NVCVInterpolationType interpolation) const
    {
        assert(in.numImages() == out.numImages());
        auto in_sizes  = imageSizes(in);
        auto out_sizes = imageSizes(out);
        auto N         = static_cast<int>(in_sizes.size());
        auto req       = m_op.getWorkspaceRequirements(N, in_sizes.data(), out_sizes.data(), in.uniqueFormat());
        auto ws        = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, interpolation);
    }

    static std::vector<nvcv::Size2D> imageSizes(const nvcv::ImageBatchVarShape &batch)
    {
        std::vector<nvcv::Size2D> sizes(batch.numImages());

        for (size_t i = 0; i < sizes.size(); i++) sizes[i] = batch[i].size();

        return sizes;
    }

    // Required override to get the py object container.
    py::object container() const override
    {
        return py::reinterpret_borrow<py::object>(this->ptr());
    }

    // Required override to get the key as the base interface class.
    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    // The static fetch function can be used to specialize the fetch of a specific object from the cache.
    // It can be used to select the best object among a number of matched cache objects.
    // Here, it fetches the biggest payload OP among cache items (can handle any smaller request).
    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());

        // Find the operator with the largest workspace (can handle any smaller request)
        std::shared_ptr<nvcvpy::ICacheItem> retItem        = cache[0];
        size_t                              maxPayloadSize = 0;

        for (const auto &item : cache)
        {
            const auto &key            = static_cast<const Key &>(item.get()->key());
            auto        keyPayloadSize = key.payloadSize();

            if (keyPayloadSize > maxPayloadSize)
            {
                maxPayloadSize = keyPayloadSize;
                retItem        = item;
            }
        }

        // Note: Removed cache.clear() and removeAllNotInUseMatching() calls to reduce per-call overhead.
        // The cache will naturally evict unused operators when memory pressure occurs.
        // This fix matches the pattern used in OpInpaint.cpp, OpSIFT.cpp, and OpFindHomography.cpp.

        return retItem;
    }

private:
    Key                  m_key;
    cvcuda::PillowResize m_op;
};

Tensor PillowResizeInto(Tensor &output, Tensor &input, nvcv::ImageFormat format, NVCVInterpolationType interp,
                        std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }
    auto in_access = nvcv::TensorDataAccessStridedImagePlanar::Create(input.exportData());
    if (auto out_access = nvcv::TensorDataAccessStridedImagePlanar::Create(output.exportData());
        !in_access || !out_access)
    {
        throw PillowResizeError("Incompatible input/output tensor layout");
    }

    // Use CreateOperatorEx to use the extended create operator function passing the specialized PyOperator above
    // as template type, instead of the regular cvcuda::OP class used in the CreateOperator function.
    auto pillowResize = CreateOperatorEx<PyOpPillowResize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*pillowResize});

    guard.run([&pillowResize, &pstream, &input, &output, &format, &interp]()
              { pillowResize->submit(pstream->cudaHandle(), input, output, format, interp); });

    return output;
}

Tensor PillowResize(Tensor &input, const Shape &out_shape, nvcv::ImageFormat format, NVCVInterpolationType interp,
                    std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(out_shape, input.dtype(), input.layout());

    return PillowResizeInto(output, input, format, interp, pstream);
}

ImageBatchVarShape VarShapePillowResizeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                            NVCVInterpolationType interpolation, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // The same PyOpPillowResize class and CreateOperatorEx function can be used regardless of Tensors or VarShape.
    auto pillowResize = CreateOperatorEx<PyOpPillowResize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*pillowResize});

    guard.run([&pillowResize, &pstream, &input, &output, &interpolation]()
              { pillowResize->submit(pstream->cudaHandle(), input, output, interpolation); });

    return output;
}

ImageBatchVarShape VarShapePillowResize(ImageBatchVarShape &input, const std::vector<std::tuple<int, int>> &outSizes,
                                        NVCVInterpolationType interpolation, std::optional<Stream> pstream)
{
    if (static_cast<int32_t>(outSizes.size()) != input.numImages())
    {
        throw PillowResizeError("Invalid outSizes passed");
    }

    ImageBatchVarShape output = CreateSizedImageBatch(input, outSizes);

    return VarShapePillowResizeInto(output, input, interpolation, pstream);
}

} // namespace

void ExportOpPillowResize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("pillowresize", NvtxTrace("cvcuda.pillowresize", &PillowResize), "src"_a, "shape"_a, "format"_a,
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Pillow Resize operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            shape (tuple): Shape of the output image.
            format (cvcuda.Format): Format of the input and output images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("pillowresize_into", NvtxTrace("cvcuda.pillowresize_into", &PillowResizeInto), "dst"_a, "src"_a, "format"_a,
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Pillow Resize operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            shape (tuple): Shape of the output image.
            format (cvcuda.Format): Format of the input and output images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("pillowresize", NvtxTrace("cvcuda.pillowresize", &VarShapePillowResize), "src"_a, "sizes"_a,
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Pillow Resize operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            sizes (Tuple[int]): Shapes of output images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("pillowresize_into", NvtxTrace("cvcuda.pillowresize_into", &VarShapePillowResizeInto), "dst"_a, "src"_a,
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Pillow Resize operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
