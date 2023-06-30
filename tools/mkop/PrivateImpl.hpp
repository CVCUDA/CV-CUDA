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

/**
 * @file Op__OPNAME__.hpp
 *
 * @brief Defines the private C++ Class for the __OPNAME__ operation.
 */

#ifndef CVCUDA_PRIV___OPNAMECAP___HPP
#define CVCUDA_PRIV___OPNAMECAP___HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class __OPNAME__ final : public IOperator
{
public:
    explicit __OPNAME__();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const;

private:
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV___OPNAMECAP___HPP
