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

#include "priv/AllocatorManager.hpp"
#include "priv/ImageBatchManager.hpp"
#include "priv/ImageManager.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"
#include "priv/TensorManager.hpp"

#include <nvcv/Config.h>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxImageCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::ImageManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxImageBatchCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::ImageBatchManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxTensorCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::TensorManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxAllocatorCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::AllocatorManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}
