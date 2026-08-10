/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PY_UNARY_ELEMENTWISE_OP_HPP
#define CVCUDA_PY_UNARY_ELEMENTWISE_OP_HPP

// Shared Python-binding plumbing for unary element-wise operators (Invert, Solarize, Posterize, …).
// Every such operator's binding has the same body — create the operator, take a ResourceGuard over
// the input/output, and submit on the stream — differing only by the operator type and any extra
// scalar parameters (e.g. Solarize's threshold, Posterize's bits). make-op generates each binding
// from one template, so that body is centralized here and each Op<Name>.cpp keeps only the thin,
// Python-facing wrappers (with op-specific argument names) plus the m.def docstrings. The extra
// scalar parameters are forwarded through the variadic `Args...` to the operator's submit().

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <optional>

namespace cvcudapy {

// Run a unary element-wise operator into a caller-provided output (Tensor or ImageBatchVarShape).
// `Args...` are forwarded verbatim as the trailing submit() parameters (none for Invert, the
// threshold for Solarize, the bit count for Posterize, …).
template<class Op, class InOut, class... Args>
InOut UnaryElementwiseInto(InOut &output, InOut &input, std::optional<Stream> pstream, Args... args)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<Op>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run([&op, &pstream, &input, &output, &args...]()
              { op->submit(pstream->cudaHandle(), input, output, args...); });

    return output;
}

// Tensor convenience overload: allocate an output matching the input's shape/dtype/layout.
template<class Op, class... Args>
Tensor UnaryElementwiseTensor(Tensor &input, std::optional<Stream> pstream, Args... args)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return UnaryElementwiseInto<Op>(output, input, pstream, args...);
}

// ImageBatchVarShape convenience overload: allocate an output matching the input batch.
template<class Op, class... Args>
ImageBatchVarShape UnaryElementwiseVarShape(ImageBatchVarShape &input, std::optional<Stream> pstream, Args... args)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return UnaryElementwiseInto<Op>(output, input, pstream, args...);
}

} // namespace cvcudapy

#endif // CVCUDA_PY_UNARY_ELEMENTWISE_OP_HPP
