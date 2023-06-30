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

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpPillowResize.hpp>
#include <cvcuda/Types.h>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ImageFormat.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {

// Specialized class for cvcuda::PillowResize operator with a better cache Key.
// It allows for reusing an existing operator object from cache if its payload size is >= the required size.
// It also allows to fetch the biggest payload object to be reused while removing all others.
// This is more flexible than using the generic PyOperator class and its Key class.
class PyOpPillowResize : public nvcvpy::Container
{
public:
    // Define a Key class to be used by the cache to fetch similar items for potential reuse.
    class Key : public nvcvpy::IKey
    {
    public:
        // Arguments of the key constructor should match the corresponding cvcuda operator arguments.
        Key(const nvcv::Size2D &maxSize, int maxBatchSize, nvcv::ImageFormat fmt)
            : m_maxSize{maxSize}
            , m_maxBatchSize{maxBatchSize}
            , m_format{fmt}
        {
        }

        // The payload size is an approximate function of the actual size of the payload.
        // There is no need to be an exact value, it is just provide ordering inside cache.
        size_t payloadSize() const
        {
            return m_maxSize.w * m_maxSize.h * m_maxBatchSize;
        }

    private:
        // The hash is based only on the image format used by the operator.
        // (In addition to the OP type as defined by IKey).
        size_t doGetHash() const override
        {
            return ComputeHash(m_format);
        }

        // The comparison of keys is based on the payload size, the one in the cache is "that" key.
        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            const Key &that = static_cast<const Key &>(that_);
            return this->payloadSize() <= that.payloadSize();
        }

        nvcv::Size2D      m_maxSize;
        int               m_maxBatchSize;
        nvcv::ImageFormat m_format;
    };

    // Constructor instantiate the cache key and the operator object.
    PyOpPillowResize(const nvcv::Size2D &maxSize, int maxBatchSize, nvcv::ImageFormat fmt)
        : m_key(maxSize, maxBatchSize, fmt)
        , m_op(maxSize, maxBatchSize, fmt)
    {
    }

    // The submit forwards its args to the OP's call operator.
    template<class... AA>
    void submit(AA &&...args)
    {
        m_op(std::forward<AA>(args)...);
    }

    // Required override to get the py object container.
    py::object container() const override
    {
        return *this;
    }

    // Required override to get the key as the base interface class.
    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    // The static fetch function can be used to specialize the fetch of a specific object from the cache.
    // It can be used to select the best object among a number of matched cache objects.
    // It can also be used to remove other objects that are not needed in the cache anymore.
    // Here, it fetches the biggest payload OP among cache items and remove all other OPs from the cache.
    // It is ok to remove them since the biggest payload OP can be used to accomodate all of them,
    // so they will never be reused and thus are no longer necessary.
    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());

        std::shared_ptr<nvcvpy::ICacheItem> retItem        = cache[0];
        size_t                              maxPayloadSize = 0;

        for (const auto &item : cache)
        {
            const Key &key            = static_cast<const Key &>(item.get()->key());
            size_t     keyPayloadSize = key.payloadSize();

            if (keyPayloadSize > maxPayloadSize)
            {
                maxPayloadSize = keyPayloadSize;
                retItem        = item;
            }
        }

        cache.clear();

        nvcvpy::Cache::removeAllNotInUseMatching(retItem.get()->key());

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
    auto in_access  = nvcv::TensorDataAccessStridedImagePlanar::Create(input.exportData());
    auto out_access = nvcv::TensorDataAccessStridedImagePlanar::Create(output.exportData());
    if (!in_access || !out_access)
    {
        throw std::runtime_error("Incompatible input/output tensor layout");
    }

    nvcv::Size2D maxSize{std::max(in_access->numCols(), out_access->numCols()),
                         std::max(in_access->numRows(), out_access->numRows())};

    int maxBatchSize = static_cast<int>(in_access->numSamples());

    // Use CreateOperatorEx to use the extended create operator function passing the specialized PyOperator above
    // as template type, instead of the regular cvcuda::OP class used in the CreateOperator function.
    auto pillowResize = CreateOperatorEx<PyOpPillowResize>(maxSize, maxBatchSize, format);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*pillowResize});

    pillowResize->submit(pstream->cudaHandle(), input, output, interp);

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

    nvcv::Size2D maxSrcSize = input.maxSize();
    nvcv::Size2D maxDstSize = output.maxSize();

    nvcv::Size2D maxSize{std::max(maxSrcSize.w, maxDstSize.w), std::max(maxSrcSize.h, maxDstSize.h)};

    int maxBatchSize = static_cast<int>(input.capacity());

    // The same PyOpPillowResize class and CreateOperatorEx function can be used regardless of Tensors or VarShape.
    auto pillowResize = CreateOperatorEx<PyOpPillowResize>(maxSize, maxBatchSize, input.uniqueFormat());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*pillowResize});

    pillowResize->submit(pstream->cudaHandle(), input, output, interpolation);

    return output;
}

ImageBatchVarShape VarShapePillowResize(ImageBatchVarShape &input, const std::vector<std::tuple<int, int>> &outSizes,
                                        NVCVInterpolationType interpolation, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    if (static_cast<int32_t>(outSizes.size()) != input.numImages())
    {
        throw std::runtime_error("Invalid outSizes passed");
    }

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        auto              size   = outSizes[i];
        auto              image  = Image::Create({std::get<0>(size), std::get<1>(size)}, format);
        output.pushBack(image);
    }

    return VarShapePillowResizeInto(output, input, interpolation, pstream);
}

} // namespace

void ExportOpPillowResize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("pillowresize", &PillowResize, "src"_a, "shape"_a, "format"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            shape (Shape): Shape of the output image.
            format (ImageFormat): Format of the input and output images.
            interp(Interp): Interpolation type used for transform.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("pillowresize_into", &PillowResizeInto, "dst"_a, "src"_a, "format"_a, "interp"_a = NVCV_INTERP_LINEAR,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            shape (Shape): Shape of the output image.
            format (ImageFormat): Format of the input and output images.
            interp(Interp): Interpolation type used for transform.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("pillowresize", &VarShapePillowResize, "src"_a, "sizes"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            sizes (Tuple vector): Shapes of output images.
            interp(Interp): Interpolation type used for transform.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("pillowresize_into", &VarShapePillowResizeInto, "dst"_a, "src"_a, "interp"_a = NVCV_INTERP_LINEAR,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            sizes (Tuple vector): Shapes of output images.
            interp(Interp): Interpolation type used for transform.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
