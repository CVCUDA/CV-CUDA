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

#ifndef NVCV_IIMAGEBATCH_HPP
#define NVCV_IIMAGEBATCH_HPP

#include "Image.hpp"
#include "ImageBatch.h"
#include "ImageBatchData.hpp"
#include "detail/Optional.hpp"

#include <iterator>

namespace nvcv {

class IImageBatch
{
public:
    using HandleType    = NVCVImageBatchHandle;
    using BaseInterface = IImageBatch;

    virtual ~IImageBatch() = default;

    NVCVImageBatchHandle handle() const;
    static IImageBatch  *cast(HandleType h);

    int32_t capacity() const;
    int32_t numImages() const;

    const IImageBatchData *exportData(CUstream stream) const;

    void  setUserPointer(void *ptr);
    void *userPointer() const;

private:
    virtual NVCVImageBatchHandle doGetHandle() const = 0;

    // Only one leaf, we can use an optional for now.
    mutable detail::Optional<ImageBatchVarShapeDataStridedCuda> m_cacheData;
};

class IImageBatchVarShape : public IImageBatch
{
public:
    template<class IT>
    void pushBack(IT itBeg, IT itend);
    void pushBack(const IImage &img);
    void popBack(int32_t imgCount = 1);

    // For any invocable functor with zero parameters
    template<class F, class = decltype(std::declval<F>()())>
    void pushBack(F &&cb);

    void clear();

    Size2D      maxSize() const;
    ImageFormat uniqueFormat() const;

    const IImageBatchVarShapeData *exportData(CUstream stream) const;

    IImage &operator[](ptrdiff_t n) const;

    class Iterator;

    using ConstIterator = Iterator;

    ConstIterator begin() const;
    ConstIterator end() const;

    ConstIterator cbegin() const;
    ConstIterator cend() const;
};

class IImageBatchVarShape::Iterator
{
public:
    using value_type        = IImage;
    using reference         = const value_type &;
    using pointer           = const value_type *;
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = ptrdiff_t;

    Iterator();
    Iterator(const Iterator &that);
    Iterator &operator=(const Iterator &that);

    reference operator*() const;
    Iterator  operator++(int);
    Iterator &operator++();
    pointer   operator->() const;

    bool operator==(const Iterator &that) const;
    bool operator!=(const Iterator &that) const;

private:
    const IImageBatchVarShape *m_batch;
    int                        m_curIndex;

    friend class IImageBatchVarShape;
    Iterator(const IImageBatchVarShape &batch, int32_t idxImage);
};

} // namespace nvcv

#include "detail/IImageBatchImpl.hpp"

#endif // NVCV_IIMAGEBATCH_HPP
