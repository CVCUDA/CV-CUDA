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

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpMorphology.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor MorphologyInto(Tensor &output, Tensor &input, NVCVMorphologyType morph_type,
                      const std::tuple<int, int> &maskSize, const std::tuple<int, int> &anchor,
                      std::optional<Tensor> workspace, int32_t iteration, NVCVBorderType border,
                      std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto morphology = CreateOperator<cvcuda::Morphology>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*morphology});

    nvcv::Size2D maskSizeArg{std::get<0>(maskSize), std::get<1>(maskSize)};
    int2         anchorArg;
    anchorArg.x = std::get<0>(anchor);
    anchorArg.y = std::get<1>(anchor);

    if (workspace)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*workspace});
        guard.run(
            [&morphology, &pstream, &input, &output, &workspace, &morph_type, &maskSizeArg, &anchorArg, &iteration,
             &border]()
            {
                morphology->submit(pstream->cudaHandle(), input, output, nvcv::OptionalTensorConstRef{*workspace},
                                   morph_type, maskSizeArg, anchorArg, iteration, border);
            });
    }
    else
    {
        guard.run(
            [&morphology, &pstream, &input, &output, &morph_type, &maskSizeArg, &anchorArg, &iteration, &border]()
            {
                morphology->submit(pstream->cudaHandle(), input, output, nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                                   morph_type, maskSizeArg, anchorArg, iteration, border);
            });
    }

    return output;
}

Tensor Morphology(Tensor &input, NVCVMorphologyType morph_type, const std::tuple<int, int> &maskSize,
                  const std::tuple<int, int> &anchor, const std::optional<Tensor> &workspace, int32_t iteration,
                  NVCVBorderType border, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return MorphologyInto(output, input, morph_type, maskSize, anchor, workspace, iteration, border, pstream);
}

ImageBatchVarShape MorphologyVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                          NVCVMorphologyType morph_type, Tensor &masks, Tensor &anchors,
                                          std::optional<ImageBatchVarShape> workspace, const int32_t iteration,
                                          const NVCVBorderType borderMode, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto morphology = CreateOperator<cvcuda::Morphology>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, masks, anchors});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*morphology});

    if (workspace)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*workspace});
        guard.run(
            [&morphology, &pstream, &input, &output, &workspace, &morph_type, &masks, &anchors, &iteration,
             &borderMode]()
            {
                morphology->submit(pstream->cudaHandle(), input, output,
                                   nvcv::OptionalImageBatchVarShapeConstRef{*workspace}, morph_type, masks, anchors,
                                   iteration, borderMode);
            });
    }
    else
    {
        guard.run(
            [&morphology, &pstream, &input, &output, &morph_type, &masks, &anchors, &iteration, &borderMode]()
            {
                morphology->submit(pstream->cudaHandle(), input, output,
                                   nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt}, morph_type, masks, anchors,
                                   iteration, borderMode);
            });
    }

    return output;
}

ImageBatchVarShape MorphologyVarShape(ImageBatchVarShape &input, NVCVMorphologyType morph_type, Tensor &masks,
                                      Tensor &anchors, const std::optional<ImageBatchVarShape> &workspace,
                                      const int32_t iteration, const NVCVBorderType borderMode,
                                      std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return MorphologyVarShapeInto(output, input, morph_type, masks, anchors, workspace, iteration, borderMode, pstream);
}

} // namespace

void ExportOpMorphology(py::module &m)
{
    using namespace pybind11::literals;

    m.def("morphology", NvtxTrace("cvcuda.morphology", &Morphology), "src"_a, "morphologyType"_a, "maskSize"_a,
          "anchor"_a, py::kw_only(), "workspace"_a = nullptr, "iteration"_a = 1,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr, R"pbdoc(
        Executes the Morphology operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            morphologyType (cvcuda.MorphologyType): Type of operation to perform (e.g. cvcuda.MorphologyType.ERODE or
                cvcuda.MorphologyType.DILATE).
            maskSize (Tuple[int, int]): Mask width and height for morphology operation.
            anchor (Tuple[int, int]): X,Y offset of kernel, use -1,-1 for center.
            workspace (cvcuda.Tensor, optional): Workspace tensor for intermediate results, must be the same size as src. Can be omitted if operation is Dilate/Erode with iteration = 1.
            iteration (int, optional): Number of times to run the kernel.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("morphology_into", NvtxTrace("cvcuda.morphology_into", &MorphologyInto), "dst"_a, "src"_a, "morphologyType"_a,
          "maskSize"_a, "anchor"_a, py::kw_only(), "workspace"_a = nullptr, "iteration"_a = 1,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr,
          R"pbdoc(
        Executes the Morphology operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            morphologyType (cvcuda.MorphologyType): Type of operation to perform (e.g. cvcuda.MorphologyType.ERODE or
                cvcuda.MorphologyType.DILATE).
            maskSize (Tuple[int, int]): Mask width and height for morphology operation.
            anchor (Tuple[int, int]): X,Y offset of kernel, use -1,-1 for center.
            workspace (cvcuda.Tensor, optional): Workspace tensor for intermediate results, must be the same size as src. Can be omitted if operation is Dilate/Erode with iteration = 1.
            iteration (int, optional): Number of times to run the kernel.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("morphology", NvtxTrace("cvcuda.morphology", &MorphologyVarShape), "src"_a, "morphologyType"_a, "masks"_a,
          "anchors"_a, py::kw_only(), "workspace"_a = nullptr, "iteration"_a = 1,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr, R"pbdoc(
        Executes the Morphology operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            morphologyType (cvcuda.MorphologyType): Type of operation to perform (e.g. cvcuda.MorphologyType.ERODE or
                cvcuda.MorphologyType.DILATE).
            maskSize (cvcuda.Tensor): Mask width and height for morphology operation for every image.
            anchor (cvcuda.Tensor): X,Y offset of kernel for every image, use -1,-1 for center.
            workspace (cvcuda.ImageBatchVarShape, optional): Workspace tensor for intermediate results, must be the same size as src. Can be omitted if operation is Dilate/Erode with iteration = 1.
            iteration (int, optional): Number of times to run the kernel.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("morphology_into", NvtxTrace("cvcuda.morphology_into", &MorphologyVarShapeInto), "dst"_a, "src"_a,
          "morphologyType"_a, "masks"_a, "anchors"_a, py::kw_only(), "workspace"_a = nullptr, "iteration"_a = 1,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr,
          R"pbdoc(
        Executes the Morphology operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            morphologyType (cvcuda.MorphologyType): Type of operation to perform (e.g. cvcuda.MorphologyType.ERODE or
                cvcuda.MorphologyType.DILATE).
            maskSize (cvcuda.Tensor): Mask width and height for morphology operation for every image.
            anchor (cvcuda.Tensor): X,Y offset of kernel for every image, use -1,-1 for center.
            workspace (cvcuda.ImageBatchVarShape, optional): Workspace tensor for intermediate results, must be the same size as src. Can be omitted if operation is Dilate/Erode with iteration = 1.
            iteration (int, optional): Number of times to run the kernel.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}
} // namespace cvcudapy
