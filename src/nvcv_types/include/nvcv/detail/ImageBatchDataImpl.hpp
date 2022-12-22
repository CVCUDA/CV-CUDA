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

#ifndef NVCV_IMAGEBATCHDATA_IMPL_HPP
#define NVCV_IMAGEBATCHDATA_IMPL_HPP

#ifndef NVCV_IMAGEBATCHDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// ImageBatchVarShapeDataStridedCuda implementation -----------------------
inline ImageBatchVarShapeDataStridedCuda::ImageBatchVarShapeDataStridedCuda(int32_t numImages, const Buffer &buffer)
{
    NVCVImageBatchData &data = this->cdata();

    data.numImages              = numImages;
    data.bufferType             = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA;
    data.buffer.varShapeStrided = buffer;
}

} // namespace nvcv

#endif // NVCV_IMAGEBATCHDATA_IMPL_HPP
