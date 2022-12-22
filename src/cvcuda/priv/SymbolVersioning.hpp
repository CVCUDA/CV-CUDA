/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PRIV_SYMBOLVERSIONING_HPP
#define CVCUDA_PRIV_SYMBOLVERSIONING_HPP

#include <util/SymbolVersioning.hpp>

#define CVCUDA_DEFINE_API(...)     NVCV_PROJ_DEFINE_API(CVCUDA, __VA_ARGS__)
#define CVCUDA_DEFINE_OLD_API(...) NVCV_PROJ_DEFINE_OLD_API(CVCUDA, __VA_ARGS__)

#endif // CVCUDA_PRIV_SYMBOLVERSIONING_HPP
