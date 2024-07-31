/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "WorkspaceCache.hpp"

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
        Key() {}

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
    PyOpPillowResize()
        : m_key()
        , m_op()
    {
    }

    inline void submit(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, nvcv::ImageFormat format,
                       NVCVInterpolationType interpolation)
    {
        int          batch_size = getBatchSize(in);
        nvcv::Size2D in_size    = imageSize(in);
        nvcv::Size2D out_size   = imageSize(out);

        auto req = m_op.getWorkspaceRequirements(batch_size, out_size, in_size, format);
        auto ws  = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, interpolation);
    }

    inline int getBatchSize(const nvcv::Tensor &tensor)
    {
        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensor.exportData());
        if (!access)
            throw std::runtime_error("Incompatible tensor layout");

        return access->numSamples();
    }

    static nvcv::Size2D imageSize(const nvcv::Tensor &tensor)
    {
        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensor.exportData());
        if (!access)
            throw std::runtime_error("Incompatible tensor layout");

        return access->size();
    }

    inline void submit(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                       const NVCVInterpolationType interpolation)
    {
        assert(in.numImages() == out.numImages());
        auto in_sizes  = imageSizes(in);
        auto out_sizes = imageSizes(out);
        int  N         = in_sizes.size();
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

    // Use CreateOperatorEx to use the extended create operator function passing the specialized PyOperator above
    // as template type, instead of the regular cvcuda::OP class used in the CreateOperator function.
    auto pillowResize = CreateOperatorEx<PyOpPillowResize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*pillowResize});

    pillowResize->submit(pstream->cudaHandle(), input, output, format, interp);

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

    py::options options;
    options.disable_function_signatures();

    m.def("pillowresize", &PillowResize, "src"_a, "shape"_a, "format"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

	cvcuda.pillowresize(src: nvcv.Tensor, shape:Shape, format:ImageFormat, interp: Interp = cvcuda.Interp.LINEAR, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more images.
            shape (tuple): Shape of the output image.
            format (nvcv.Format): Format of the input and output images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("pillowresize_into", &PillowResizeInto, "dst"_a, "src"_a, "format"_a, "interp"_a = NVCV_INTERP_LINEAR,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

	cvcuda.pillowresize_into(dst: nvcv.Tensor, src: nvcv.Tensor, shape: Tuple[int], format: nvcv.Format, interp: Interp = cvcuda.Interp.LINEAR, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor to store the result of the operation.
            src (nvcv.Tensor): Input tensor containing one or more images.
            shape (tuple): Shape of the output image.
            format (nvcv.Format): Format of the input and output images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("pillowresize", &VarShapePillowResize, "src"_a, "sizes"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

	cvcuda.pillowresize(src: nvcv.ImageBatchVarShape, shape: Tuple[int], format: nvcv.Format, interp: Interp = cvcuda.Interp.LINEAR, stream: Optional[nvcv.cuda.Stream] = None) ->ImageBatchVarShape

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            sizes (Tuple[int]): Shapes of output images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("pillowresize_into", &VarShapePillowResizeInto, "dst"_a, "src"_a, "interp"_a = NVCV_INTERP_LINEAR,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

	cvcuda.pillowresize(dst: nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, shape: Tuple[int], format: nvcv.Format, interp: cvcuda.Interp = cvcuda.Interp.LINEAR, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Pillow Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pillow Resize operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            dst (nvcv.ImageBatchVarShape): Output image batch containing the result of the operation.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
