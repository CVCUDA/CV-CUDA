/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpNonMaximumSuppression.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <limits>

namespace cvcudapy {

namespace {

Tensor NonMaximumSuppressionInto(Tensor &dst, Tensor &src, Tensor &scores, float scoreThreshold, float iouThreshold,
                                 std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::NonMaximumSuppression>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {src, scores});
    guard.add(LockMode::LOCK_WRITE, {dst});
    guard.add(LockMode::LOCK_NONE, {*op});

    op->submit(pstream->cudaHandle(), src, dst, scores, scoreThreshold, iouThreshold);

    return std::move(dst);
}

Tensor NonMaximumSuppression(Tensor &src, Tensor &scores, float scoreThreshold, float iouThreshold,
                             std::optional<Stream> pstream)
{
    nvcvpy::Shape dstShape(2);

    dstShape[0] = src.shape()[0];
    dstShape[1] = src.shape()[1];

    Tensor dst = Tensor::Create(dstShape, nvcv::TYPE_U8);

    return NonMaximumSuppressionInto(dst, src, scores, scoreThreshold, iouThreshold, pstream);
}

} // namespace

void ExportOpNonMaximumSuppression(py::module &m)
{
    using namespace pybind11::literals;

    m.def("nms", &NonMaximumSuppression, "src"_a, "scores"_a,
          "score_threshold"_a = std::numeric_limits<float>::epsilon(), "iou_threshold"_a = 1.0, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes NMS.

        The Non-Maximum Suppression (NMS) operation reads a set of input bounding boxes (bboxes) proposals and
        their scores and writes an output boolean mask with suppressed bboxes as zeros and selected bboxes as ones.

        See also:
            Refer to the CV-CUDA C API reference for the Non-Maximum Suppression operator
            for more details and usage examples.

        Args:
            src (Tensor): src[i, j] is the set of input bounding box proposals
                for an image where i ranges from 0 to batch-1, j ranges from 0
                to number of bounding box proposals anchored at the top-left of
                the bounding box area
            scores (Tensor): scores[i, j] are the associated scores for each
                bounding box proposal in ``src`` considered during the reduce
                operation of NMS
            score_threshold (float): Minimum score of a bounding box proposals
            iou_threshold (float): Maximum overlap between bounding box proposals
                covering the same effective image region as calculated by
                Intersection-over-Union (IoU)
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor of selected bounding boxes.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
    m.def("nms_into", &NonMaximumSuppressionInto, "dst"_a, "src"_a, "scores"_a,
          "score_threshold"_a = std::numeric_limits<float>::epsilon(), "iou_threshold"_a = 1.0, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes NMS.

        The Non-Maximum Suppression (NMS) operation reads a set of input bounding boxes (bboxes) proposals and
        their scores and writes an output boolean mask with suppressed bboxes as zeros and selected bboxes as ones.

        See also:
            Refer to the CV-CUDA C API reference for the Non-Maximum Suppression operator
            for more details and usage examples.

        Args:
            dst (Tensor): dst[i, j] is the output boolean mask marking selected
                bounding boxes, where i ranges from 0 to batch-1, j ranges from 0
                to the number of bounding box proposals anchored at the
                top-left of the bounding box area
            src (Tensor): src[i, j] is the set of input bounding box proposals
                for an image where i ranges from 0 to batch-1, j ranges from 0
                to number of bounding box proposals anchored at the top-left of
                the bounding box area
            scores (Tensor): scores[i, j] are the associated scores for each
                bounding box proposal in ``src`` considered during the reduce
                operation of NMS
            score_threshold (float): Minimum score of a bounding box proposals
            iou_threshold (float): Maximum overlap between bounding box proposals
                covering the same effective image region as calculated by
                Intersection-over-Union (IoU)
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
