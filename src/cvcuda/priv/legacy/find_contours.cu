/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cooperative_groups.h>
#include <nvcv/alloc/Requirements.hpp>
#include <util/Math.hpp>

#include <type_traits>

namespace cg = cooperative_groups;

namespace nvcv::legacy::cuda_op {

template<typename T>
__forceinline__ __host__ __device__ std::enable_if_t<std::is_integral<T>::value, T> mod(const T &a, const T &b)
{
    T c = a % b;
    return (c < 0) ? (c + b) : c;
}

// Importing scope of the helpers namespace
using namespace nvcv::legacy::helpers;

using CountType = uint32_t;
using IndexType = int32_t;
using PixelType = uint8_t;
using LabelType = IndexType;
using PointType = int2;
using CoordType = uint3;
using MaskType  = uint32_t;

template<typename PixelType>
using DeviceImage   = cuda::FullTensorWrap<PixelType, 3>;
using BoundaryLabel = Ptr2dNHWC<LabelType>;
using Neighborhood  = Ptr2dNHWC<MaskType>;
using ConnectList   = Ptr2dNHWC<IndexType>;
using CountList     = Ptr2dNL<CountType>;
using NodeList      = cuda::FullTensorWrap<int, 3>;
using NodeCounts    = cuda::FullTensorWrap<int, 2>;

using KernelGrid  = cg::grid_group;
using KernelBlock = cg::thread_block;
using KernelWarp  = cg::thread_block_tile<32, KernelBlock>;
using ActiveWarp  = cg::coalesced_group;

template<typename ValueType>
class SharedHWWrapper
{
public:
    using type       = SharedHWWrapper<ValueType>;
    using value_type = ValueType;
    using size_type  = IndexType;

    __forceinline__ __device__ SharedHWWrapper()
        : m_height{0}
        , m_width{0}
        , m_data{nullptr}
    {
    }

    __forceinline__ __device__ SharedHWWrapper(const type &other)
        : m_height{other.m_height}
        , m_width{other.m_width}
        , m_data{other.m_data}
    {
    }

    __forceinline__ __device__ SharedHWWrapper(type &&other)
    {
        this->m_height = other.m_height;
        other.m_height = 0;
        this->m_width  = other.m_width;
        other.m_width  = 0;
        this->m_data   = other.m_data;
        other.m_data   = nullptr;
    }

    __forceinline__ __device__ SharedHWWrapper(size_type rows, size_type cols, ValueType *data)
        : m_height{rows}
        , m_width{cols}
        , m_data{data}
    {
    }

    __forceinline__ __device__ ~SharedHWWrapper()
    {
        m_height = 0;
        m_width  = 0;
        m_data   = nullptr;
    }

    __forceinline__ __device__ type &operator=(const type &other)
    {
        this->m_height = other.m_height;
        this->m_width  = other.m_width;
        this->m_data   = other.m_data;
        return *this;
    }

    __forceinline__ __device__ type &operator=(type &&other)
    {
        this->m_height = other.m_height;
        other.m_height = 0;
        this->m_width  = other.m_width;
        other.m_width  = 0;
        this->m_data   = other.m_data;
        other.m_data   = nullptr;
        return *this;
    }

    __forceinline__ __device__ value_type &operator[](IndexType index)
    {
        assert(0 <= index && index < static_cast<IndexType>(m_width * m_height));
        return this->m_data[index];
    }

    const __forceinline__ __device__ value_type &operator[](IndexType index) const
    {
        assert(0 <= index && index < static_cast<IndexType>(m_width * m_height));
        return this->m_data[index];
    }

    __forceinline__ __device__ value_type &operator[](PointType pos)
    {
        using AxisType = decltype(pos.x);
        assert(0 <= pos.x && pos.x < static_cast<AxisType>(m_width));
        assert(0 <= pos.y && pos.y < static_cast<AxisType>(m_height));
        return (*this)[this->pointToIndex(pos)];
    }

    const __forceinline__ __device__ value_type &operator[](PointType pos) const
    {
        using AxisType = decltype(pos.x);
        assert(0 <= pos.x && pos.x < static_cast<AxisType>(m_width));
        assert(0 <= pos.y && pos.y < static_cast<AxisType>(m_height));
        return (*this)[this->pointToIndex(pos)];
    }

    const __forceinline__ __device__ value_type *ptr(IndexType index) const
    {
        return &((*this)[index]);
    }

    const __forceinline__ __device__ value_type *ptr(PointType pos) const
    {
        return &((*this)[pos]);
    }

    __forceinline__ __device__ IndexType pointToIndex(PointType pos) const
    {
        return static_cast<IndexType>(pos.y * this->m_width + pos.x);
    }

    __forceinline__ __device__ PointType indexToPoint(IndexType index) const
    {
        return PointType{mod(index, m_width), index / m_width};
    }

    __forceinline__ __device__ size_type height() const
    {
        return m_height;
    }

    __forceinline__ __device__ size_type width() const
    {
        return m_width;
    }

    __forceinline__ __device__ size_type volume() const
    {
        return this->height() * this->width();
    }

private:
    size_type m_height{0};
    size_type m_width{0};

    value_type *m_data{nullptr};
};

// Representing the shared memory in int32 for to avoid bank conflicts
using SharedImage = SharedHWWrapper<int32_t>;
using SharedLabel = SharedHWWrapper<LabelType>;

template<typename ValueType, IndexType MAX_SIZE>
class LocalQueue
{
public:
    __device__ LocalQueue()
        : m_front(0)
        , m_back(0)
    {
    }

    // Push an item to the queue; returns false if the queue is full.
    __device__ bool push(const ValueType &value)
    {
        if (mod(m_back + 1, MAX_SIZE) == m_front)
            return false; // Queue is full

        m_data[m_back] = value;
        m_back         = mod(m_back + 1, MAX_SIZE);

        return true;
    }

    __device__ IndexType pushOrDelete(const ValueType &value)
    {
        IndexType removeAt = MAX_SIZE;

        // Check if value is already in the queue
        for (auto i = m_front; i != m_back; i = mod(i + 1, MAX_SIZE))
        {
            if (m_data[i] == value)
            {
                // Remove the value by shifting everything to the left
                removeAt = i;
                this->remove(i);
                i = m_back;
            }
        }

        // If we've reached here, value is not in the queue. Push it.
        if (removeAt == MAX_SIZE)
        {
            push(value);
        }

        return removeAt;
    }

    __device__ void remove(IndexType index)
    {
        // Remove the value by shifting everything to the left
        for (auto j = mod(index + 1, MAX_SIZE); j != m_back; j = mod(j + 1, MAX_SIZE))
        {
            m_data[index] = m_data[j];
            index         = j;
            j             = mod(j + 1, MAX_SIZE);
        }
        m_back = mod(m_back - 1, MAX_SIZE);
    }

    // Pop an item from the queue; returns false if the queue is empty.
    __device__ bool pop(ValueType &value)
    {
        if (m_front == m_back)
            return false; // Queue is empty

        value   = m_data[m_front];
        m_front = mod(m_front + 1, MAX_SIZE);

        return true;
    }

    // Check if the queue is empty.
    __device__ bool isEmpty() const
    {
        return m_front == m_back;
    }

    // Check if the queue is full.
    __device__ bool isFull() const
    {
        return mod(m_back + 1, MAX_SIZE) == m_front;
    }

private:
    ValueType m_data[MAX_SIZE]; // Array to store the queue's elements.
    IndexType m_front;          // Index of the front element.
    IndexType m_back;           // Index where the next element will be pushed.
};

// Creating a list of point offsets for 8-point stencil which are specified
// in clock-wise rotating order from top-left
__constant__ PointType OFFSET[8] = {
    {-1, -1}, // 0 ==> Left-Up
    { 0, -1}, // 1 ==> Up
    { 1, -1}, // 2 ==> Right-Up
    { 1,  0}, // 3 ==> Right
    { 1,  1}, // 4 ==> Right-Down
    { 0,  1}, // 5 ==> Down
    {-1,  1}, // 6 ==> Left-Down
    {-1,  0}  // 7 ==> Left
};

// Pre-declarations
/******************************************************************************/
template<typename GlobalValueType, typename SharedValueType>
__device__ SharedValueType globalAt(const DeviceImage<GlobalValueType> &global, PointType pos, IndexType batch,
                                    GlobalValueType defaultValue = 0);

template<int PAD_SIZE, typename GlobalValueType, typename SharedValueType>
__device__ void doCopyDirected(const DeviceImage<GlobalValueType> &global, PointType globalPos, PointType localPos,
                               IndexType batch, PointType direction, GlobalValueType defaultValue,
                               SharedHWWrapper<SharedValueType> &shared);

template<int PAD_SIZE, typename GlobalValueType, typename SharedValueType>
__device__ void copyGlobalToShared(const DeviceImage<GlobalValueType> &global, PointType globalPos, PointType localPos,
                                   IndexType batch, GlobalValueType defaultValue,
                                   SharedHWWrapper<SharedValueType> &shared);

__device__ MaskType getNeighborhoodMask(const SharedImage &sharedImage, PointType localPos);

__device__ bool isEdgePixel(const SharedImage &sharedImage, PointType localPos);

__device__ void setLabels(const SharedImage &sharedImage, PointType localPos, SharedLabel &sharedLabels,
                          PointType globalPos, IndexType width, IndexType height);

__device__ LabelType findRoot(const BoundaryLabel &labels, CoordType pos, LabelType badLabel);

__device__ LabelType minLabelInNeighborhood(const BoundaryLabel &labels, CoordType pos, LabelType badLabel,
                                            Neighborhood &neighbors);

__device__ void resolveRoots(BoundaryLabel &segments, CoordType pos, LabelType badLabel, Neighborhood &neighbors);

__device__ IndexType nextDirectionNot(const Neighborhood &neighbors, IndexType from, IndexType lastDir, CoordType pos,
                                      bool flipDir = true);

__device__ LabelType findHead(const BoundaryLabel &labels, const BoundaryLabel &segments, const Neighborhood &neighbors,
                              CoordType pos, IndexType from, IndexType &lastDir);

__device__ void traverseContour(const BoundaryLabel &labels, const BoundaryLabel &segments,
                                const BoundaryLabel &connectedComponents, Neighborhood &neighbors, CoordType pos,
                                ConnectList &connectList, CountList &nodeCount, CountType *contourCount,
                                LabelType badLabel);

template<typename PixelType>
__host__ void findContours_impl(DeviceImage<PixelType> &dImage, LabelType *dLabels, LabelType *dSegments,
                                LabelType *dConnectedComponents, MaskType *dNeighbors, IndexType *dConnectList,
                                CountType *dNodeCount, CountType *dContourCount, NodeList &dNodeList,
                                NodeCounts &dPointCount, IndexType height, IndexType width, IndexType batchSize,
                                cudaStream_t stream);

/******************************************************************************/

template<typename GlobalValueType, typename SharedValueType>
__forceinline__ __device__ SharedValueType globalAt(const DeviceImage<GlobalValueType> &global, PointType pos,
                                                    IndexType batch, GlobalValueType defaultValue)
{
    SharedValueType result = static_cast<SharedValueType>(defaultValue);

    // Batch size is 1
    if (0 <= pos.x && pos.x < global.shapes()[2] && 0 <= pos.y && pos.y < global.shapes()[1] && 0 <= batch
        && batch < global.shapes()[0])
    {
        result = static_cast<SharedValueType>(*global.ptr(batch, pos.y, pos.x) > 0);
    }

    return result;
}

template<int PAD_SIZE, typename GlobalValueType, typename SharedValueType>
__forceinline__ __device__ void doCopyDirected(const DeviceImage<GlobalValueType> &global, PointType globalPos,
                                               PointType localPos, IndexType batch, PointType direction,
                                               GlobalValueType defaultValue, SharedHWWrapper<SharedValueType> &shared)
{
    for (IndexType i = 1; i <= PAD_SIZE; ++i)
    {
        for (IndexType j = 1; j <= PAD_SIZE; ++j)
        {
            const auto offset = PointType{i * direction.x, j * direction.y};
            shared[localPos + offset]
                = globalAt<GlobalValueType, SharedValueType>(global, globalPos + offset, batch, defaultValue);
        }
    }
}

template<int PAD_SIZE, typename GlobalValueType, typename SharedValueType>
__device__ void copyGlobalToShared(const DeviceImage<GlobalValueType> &global, PointType globalPos, PointType localPos,
                                   IndexType batch, GlobalValueType defaultValue,
                                   SharedHWWrapper<SharedValueType> &shared)
{
    // Copying over all data within the boundary
    shared[localPos] = globalAt<GlobalValueType, SharedValueType>(global, globalPos, batch, defaultValue);

    if (localPos.x == PAD_SIZE)
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[7], defaultValue, shared);
    }
    if (localPos.x == (shared.width() - 1 - PAD_SIZE))
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[3], defaultValue, shared);
    }
    if (localPos.y == PAD_SIZE)
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[1], defaultValue, shared);
    }
    if (localPos.y == (shared.height() - 1 - PAD_SIZE))
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[5], defaultValue, shared);
    }
    if (localPos.x == PAD_SIZE && localPos.y == PAD_SIZE)
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[0], defaultValue, shared);
    }
    if (localPos.x == (shared.width() - 1 - PAD_SIZE) && localPos.y == PAD_SIZE)
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[2], defaultValue, shared);
    }
    if (localPos.x == PAD_SIZE && localPos.y == (shared.height() - 1 - PAD_SIZE))
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[6], defaultValue, shared);
    }
    if (localPos.x == (shared.width() - 1 - PAD_SIZE) && localPos.y == (shared.height() - 1 - PAD_SIZE))
    {
        doCopyDirected<PAD_SIZE>(global, globalPos, localPos, batch, OFFSET[4], defaultValue, shared);
    }
}

__device__ MaskType getNeighborhoodMask(const SharedImage &sharedImage, PointType localPos)
{
    MaskType neighborhoodMask = 0;

    for (auto dir = 0; dir < 8; ++dir)
    {
        const auto neighborPos = localPos + OFFSET[dir];

        neighborhoodMask |= ((sharedImage[neighborPos] > 0) ? (1 << dir) : 0);
    }

    return neighborhoodMask;
}

__device__ bool isEdgePixel(const SharedImage &sharedImage, PointType localPos)
{
    // NOTE: This condition might need further thought. An edge pixel is a pixel
    //   with at least 1 zero pixel neighbor and at least two set neighbors
    const auto neighborhood = getNeighborhoodMask(sharedImage, localPos);
    return sharedImage[localPos] > 0 && __popc(neighborhood & 0xaa) < 4;
}

__device__ void setLabels(const SharedImage &sharedImage, PointType localPos, SharedLabel &sharedLabels,
                          PointType globalPos, IndexType width, IndexType height)
{
    // Collecting boundary evaluation
    const auto isBoundary = isEdgePixel(sharedImage, localPos);
    const auto index      = (localPos.y - 2) * sharedLabels.width() + (localPos.x - 2);

    // Collect boundary determination of neighbors
    LabelType minIndex = isBoundary ? (globalPos.y * width + globalPos.x) : (width * height);

    for (auto i = 0; i < 4; ++i)
    {
        auto neighborPos    = localPos + OFFSET[mod(i + 7, 8)];
        auto neighborIndex  = (neighborPos.y - 2) * sharedLabels.width() + (neighborPos.x - 2);
        auto neighborIsEdge = isBoundary && isEdgePixel(sharedImage, neighborPos);

        neighborPos   = neighborPos - localPos + globalPos;
        neighborIndex = neighborIsEdge ? (neighborPos.y * width + neighborPos.x) : (width * height);

        minIndex = min(minIndex, neighborIndex);
    }

    sharedLabels[index] = minIndex;
}

__device__ LabelType findRoot(const BoundaryLabel &labels, CoordType pos, LabelType badLabel)
{
    auto next = pos.y * labels.cols + pos.x; // Linearize the pixel position.
    auto root = *labels.ptr(pos.z, pos.y, pos.x);

    // Keep finding the root until the root is a bad label or the next label is the root itself.
    while (root != badLabel && next != root)
    {
        next = root;                              // Move on to the next label.
        root = *(labels.ptr(pos.z, 0, 0) + root); // Fetch the next root label.
    }

    return root; // Return the found root label.
}

__device__ LabelType minLabelInNeighborhood(const BoundaryLabel &labels, CoordType pos, LabelType badLabel,
                                            Neighborhood &neighbors)
{
    auto      label    = *labels.ptr(pos.z, pos.y, pos.x);
    LabelType minLabel = badLabel;

    *neighbors.ptr(pos.z, pos.y, pos.x) = 0;

    // Loop through all 8 neighbors to find the smallest label.
    for (auto dir = 1; dir < 8; dir += 2)
    {
        const auto neighborLabel = *labels.ptr(pos.z, pos.y + OFFSET[dir].y, pos.x + OFFSET[dir].x);
        minLabel                 = min(neighborLabel, minLabel);

        // Update the edge neighbors in the flow structure based on valid neighbors.
        *neighbors.ptr(pos.z, pos.y, pos.x) |= (neighborLabel != badLabel && label != badLabel) ? (1 << dir) : 0;
    }

    return label == badLabel ? badLabel : minLabel; // Return the smallest label found.
}

__device__ void resolveRoots(BoundaryLabel &segments, CoordType pos, LabelType badLabel, Neighborhood &neighbors)
{
    auto label1 = *segments.ptr(pos.z, pos.y, pos.x);
    auto label2 = minLabelInNeighborhood(segments, pos, badLabel, neighbors);
    auto label3 = badLabel;

    // Resolve the root for the label1 until it remains unchanged.
    while (label1 != badLabel && label2 != badLabel && label1 != label3)
    {
        label3 = label1;
        label1 = *(segments.ptr(pos.z, 0, 0) + label1);
    }

    // Resolve the root for the label2 until it remains unchanged.
    while (label1 != badLabel && label2 != badLabel && label2 != label3)
    {
        label3 = label2;
        label2 = *(segments.ptr(pos.z, 0, 0) + label2);
    }

    // Merge label1 and label2 if they are different and not bad labels.
    while (label1 != badLabel && label2 != badLabel && label1 != label2)
    {
        label3 = atomicMin(segments.ptr(pos.z, 0, 0) + label1, label2);
        label1 = label1 == label3 ? label2 : label3;
        label2 = label3;
    }
}

__device__ IndexType nextDirectionNot(const Neighborhood &neighbors, IndexType from, IndexType lastDir, CoordType pos,
                                      bool flipDir)
{
    // Start from the direction opposite (180 degrees) to the last direction.
    // This is done by adding 4 (half of 8 directions) and taking modulo 8.
    // This ensures the result lies between 0 and 7 (inclusive).
    IndexType nextDirection = mod(lastDir + (flipDir ? 4 : 0), 8);

    // Loop to search for the next valid direction in a clockwise manner.
    // The loop starts from 1 and iterates 7 times, covering all directions.
    for (auto dir = 1; dir < 8; ++dir)
    {
        // Move in a clockwise manner by incrementing the direction
        // and taking modulo 8 to ensure it stays in the valid range.
        nextDirection = mod(nextDirection + dir, 8);

        // Check if the direction pointed by nextDirection is valid by inspecting
        // the neighbors bitmask. If valid, break out of the loop.
        if (((*(neighbors.ptr(pos.z, 0, 0) + from)) & (1 << nextDirection)) > 0)
        {
            break;
        }
    }

    // Return the determined valid direction.
    return nextDirection;
}

__device__ LabelType findHead(const BoundaryLabel &labels, const BoundaryLabel &segments, const Neighborhood &neighbors,
                              CoordType pos, IndexType from, IndexType &lastDir)
{
    // Begin at the starting position.
    IndexType next = from;

    // If segments at this new pixel is equal to the pixel, end
    while ((*(neighbors.ptr(pos.z, 0, 0) + next) & 0x87) != 0)
    {
        // Use nextDirectionNot to get the next direction to move in.
        lastDir = nextDirectionNot(neighbors, next, lastDir - (next != from ? 0 : 1), pos, next != from);

        // Update the current position by moving in the direction provided by nextDirectionNot.
        next += OFFSET[lastDir].y * labels.cols + OFFSET[lastDir].x;
    }

    // Return the position that matches the condition.
    return next;
}

__device__ void traverseContour(const BoundaryLabel &labels, const BoundaryLabel &segments,
                                const BoundaryLabel &connectedComponents, Neighborhood &neighbors, CoordType pos,
                                ConnectList &connectList, CountList &nodeCount, CountType *contourCount,
                                LabelType badLabel)
{
    // Obtain the root label for the connected component.
    // It represents the label assigned to this specific group of connected pixels.
    auto root = *connectedComponents.ptr(pos.z, pos.y, pos.x);

    // The head is a reference pixel on the contour; essentially, our starting point.
    auto head = *segments.ptr(pos.z, pos.y, pos.x);

    // The current pixel label we're working on.
    auto next = *labels.ptr(pos.z, pos.y, pos.x);
    auto curr = pos.y * labels.cols + pos.x;

    // Return early if the current pixel isn't the root pixel.
    // This ensures we are only processing root pixels.
    if (curr != root || root == badLabel)
        return;

    // Get the contour neighbor data for the current pixel.
    auto neighborhood = *neighbors.ptr(pos.z, pos.y, pos.x);

    // Calculate the first direction which has a neighbor on the edge.
    auto nextDir = __ffs(static_cast<int32_t>(neighborhood));
    for (auto dir = nextDir; dir < 7; ++dir)
    {
        nextDir = (((1 << dir) & neighborhood) > 0) ? dir : nextDir;
    }
    auto lastDir = nextDir;

    // Prepare local queues for storing pixel labels and directions.
    // These help manage which pixels/directions are processed next.
    constexpr int                   MAX_SIZE = 64;
    LocalQueue<LabelType, MAX_SIZE> labelQueue;
    LocalQueue<IndexType, MAX_SIZE> dirQueue;

    // Initialize the queues with starting values.
    labelQueue.push(root);
    dirQueue.push(nextDir);

    // Adjust the next label based on the initial direction.
    next += OFFSET[nextDir].y * labels.cols + OFFSET[nextDir].x;

    // Temporary variables for dequeuing operations.
    LabelType frusLabel;
    IndexType frusDir;

    // Keep processing pixels until there's nothing left in our queues.
    auto &counts = labels.batches > 1 ? contourCount[pos.z] : *contourCount;
    while (!labelQueue.isEmpty() && !dirQueue.isEmpty() && counts < FindContours::MAX_NUM_CONTOURS)
    {
        // Fetch the next label and direction from the front of our queues.
        labelQueue.pop(frusLabel);
        dirQueue.pop(frusDir);

        // Identify the contour's starting pixel for this segment.
        head         = findHead(labels, segments, neighbors, pos, frusLabel, frusDir);
        neighborhood = *(neighbors.ptr(pos.z, 0, 0) + head);
        if (neighborhood == 0)
        {
            continue;
        }

        // Update tracking variables to work on the head pixel.
        curr    = head;
        nextDir = mod(frusDir + (frusLabel == head ? 0 : 4), 8);
        lastDir = mod(nextDirectionNot(neighbors, head, nextDir, pos) + 4, 8);
        next    = curr + OFFSET[nextDir].y * labels.cols + OFFSET[nextDir].x;

        // Increment the total count of contours.
        IndexType contourIndex = atomicInc(&counts, FindContours::MAX_NUM_CONTOURS);
        if (contourIndex == FindContours::MAX_NUM_CONTOURS)
        {
            atomicExch(&counts, FindContours::MAX_NUM_CONTOURS);
            break;
        }
        *nodeCount.ptr(pos.z, contourIndex) = 0;

        // Traverse the contour until it loops back to the head pixel.
        while (next != head && *nodeCount.ptr(pos.z, contourIndex) != FindContours::MAX_CONTOUR_POINTS)
        {
            // Register the current pixel to the contour.
            IndexType pointIndex                              = (*nodeCount.ptr(pos.z, contourIndex))++;
            *connectList.ptr(pos.z, contourIndex, pointIndex) = curr;

            // Update the next direction based on the neighbors.
            for (auto dir = 1; dir < 8; ++dir)
            {
                nextDir = mod(lastDir + 4 - dir, 8);
                next    = curr + OFFSET[nextDir].y * labels.cols + OFFSET[nextDir].x;

                auto ccNext = *(connectedComponents.ptr(pos.z, 0, 0) + next);
                if (ccNext == root)
                {
                    break;
                }
            }

            // Move to the next pixel in the chosen direction.
            curr    = next;
            lastDir = nextDir;
        }
    }
}

template<typename PixelType>
__global__ void labelEdges(DeviceImage<PixelType> image, IndexType height, IndexType width, IndexType batchSize,
                           LabelType *dLabels)
{
    // NOTE: Potential for improvement to reduce thread divergences.

    // Shared memory buffer allocation.
    extern __shared__ int32_t sharedBuffer[];

    // Setting up the labels data structure.
    BoundaryLabel labels{batchSize, height, width, 1, dLabels};

    // Initializing cooperative groups for thread management.
    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();

    // Deriving grid and block properties.
    auto gridBlocks  = grid.group_dim().x * grid.group_dim().y * grid.group_dim().z;
    auto gridShape   = grid.group_dim() * block.group_dim();
    auto blockHeight = block.group_dim().y;
    auto blockWidth  = block.group_dim().x;
    auto blockRank   = block.group_index().z * grid.group_dim().x * grid.group_dim().y
                   + block.group_index().y * grid.group_dim().x + block.group_index().x;

    // Get pointers to shared memory for image and labels.
    auto        sharedOffset = 0;
    SharedImage sharedImage{static_cast<IndexType>(blockHeight + 4), static_cast<IndexType>(blockWidth + 4),
                            reinterpret_cast<typename SharedImage::value_type *>(&sharedBuffer[sharedOffset])};
    sharedOffset += sharedImage.volume() * sizeof(typename SharedImage::value_type) / sizeof(int32_t);
    SharedLabel sharedLabels{static_cast<IndexType>(blockHeight), static_cast<IndexType>(blockWidth),
                             reinterpret_cast<typename SharedLabel::value_type *>(&sharedBuffer[sharedOffset])};

    // Computing block dimensions in terms of tiles.
    auto blocksTileWidth  = util::DivUp(width, blockWidth);
    auto blocksTileHeight = util::DivUp(height, blockHeight);
    auto numSteps         = util::DivUp(blocksTileWidth * blocksTileHeight * batchSize, gridBlocks);

    // Thread positions within a block.
    PointType threadBlockPos{static_cast<int>(block.thread_rank() % blockWidth),
                             static_cast<int>(block.thread_rank() / blockWidth)};

    // Iterate through the steps to cover the entire image.
    for (auto step = 0; step < numSteps; ++step)
    {
        // Compute block index.
        auto      blockIndex = blockRank + step * gridBlocks;
        CoordType blockGridPos{blockIndex % blocksTileWidth, (blockIndex / blocksTileWidth) % blocksTileHeight,
                               blockIndex / (blocksTileWidth * blocksTileHeight)};

        // Compute local and global positions.
        PointType localPos{threadBlockPos.x + 2, threadBlockPos.y + 2};
        PointType globalPos{static_cast<int>(threadBlockPos.x + blockWidth * blockGridPos.x),
                            static_cast<int>(threadBlockPos.y + blockHeight * blockGridPos.y)};
        IndexType batchIndex = static_cast<int>(blockGridPos.z);

        // Populate shared memory with image data.
        copyGlobalToShared<2, PixelType, int32_t>(image, globalPos, localPos, batchIndex, 0, sharedImage);
        block.sync();

        // Assign labels to the edges.
        if (batchIndex < batchSize)
        {
            setLabels(sharedImage, localPos, sharedLabels, globalPos, width, height);
        }
        block.sync();

        // Copy labels from shared memory back to global memory.
        const auto index = (localPos.y - 2) * sharedLabels.width() + (localPos.x - 2);
        if (globalPos.x < width && globalPos.y < height && batchIndex < batchSize)
        {
            *labels.ptr(batchIndex, globalPos.y, globalPos.x) = sharedLabels[index];
        }
    }
}

__global__ void labelConnectedComponents(LabelType *dLabels, IndexType height, IndexType width, IndexType batchSize,
                                         LabelType *dSegments, LabelType *dConnectedComponents, MaskType *dNeighbors)
{
    // Set up data structures to provide structure and ease of access to labels, segments,
    // connected components, and neighbors.
    BoundaryLabel labels{batchSize, height, width, 1, dLabels};
    BoundaryLabel segments{batchSize, height, width, 1, dSegments};
    BoundaryLabel connectedComponents{batchSize, height, width, 1, dConnectedComponents};
    Neighborhood  neighbors{batchSize, height, width, 1, dNeighbors};

    // Initialize cooperative groups, which provide synchronization primitives for CUDA threads.
    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();

    // Calculate properties for the grid and blocks.
    auto gridBlocks  = grid.group_dim().x * grid.group_dim().y * grid.group_dim().z;
    auto blockHeight = block.group_dim().y;
    auto blockWidth  = block.group_dim().x;
    auto blockRank   = block.group_index().z * grid.group_dim().x * grid.group_dim().y
                   + block.group_index().y * grid.group_dim().x + block.group_index().x;

    // Calculate the width and height of blocks in tiles and the total number of steps required.
    auto blocksTileWidth  = util::DivUp(width, blockWidth);
    auto blocksTileHeight = util::DivUp(height, blockHeight);
    auto numSteps         = util::DivUp(blocksTileWidth * blocksTileHeight * batchSize, gridBlocks);

    // Determine block dimensions and thread's position within the block.
    CoordType blockDims{blockWidth, blockHeight, 1};
    CoordType threadBlockPos{block.thread_rank() % blockWidth, block.thread_rank() / blockWidth, 0};

    const auto badLabel = height * width;

    // Lambda function to encapsulate the repeated logic. It operates on the thread's position
    // and performs the given action if the position is within the image boundaries.
    auto performOperationOnThreadPos = [&](auto operation)
    {
        for (auto step = 0; step < numSteps; ++step)
        {
            // Calculate the block's position in the grid.
            auto      blockIndex = blockRank + step * gridBlocks;
            CoordType blockGridPos{blockIndex % blocksTileWidth, (blockIndex / blocksTileWidth) % blocksTileHeight,
                                   blockIndex / (blocksTileWidth * blocksTileHeight)};

            // Calculate the global position of the thread.
            auto threadPos = blockGridPos * blockDims + threadBlockPos;

            // Check if the thread position is within the boundaries of the image.
            bool inLabels = (threadPos.x < width && threadPos.y < height && threadPos.z < batchSize);

            // If valid, perform the given operation.
            if (inLabels)
            {
                operation(threadPos);
            }
        }
        grid.sync(); // Synchronize threads in the grid to ensure they are all done.
    };

    // 1. Extract edge segments of contiguous edges.
    performOperationOnThreadPos(
        [&](const CoordType &threadPos)
        { *segments.ptr(threadPos.z, threadPos.y, threadPos.x) = findRoot(labels, threadPos, badLabel); });

    // 2. Resolve roots in the segments to connect neighboring components.
    performOperationOnThreadPos([&](const CoordType &threadPos)
                                { resolveRoots(segments, threadPos, badLabel, neighbors); });

    // 3. Label the connected components.
    performOperationOnThreadPos(
        [&](const CoordType &threadPos)
        { *connectedComponents.ptr(threadPos.z, threadPos.y, threadPos.x) = findRoot(segments, threadPos, badLabel); });
}

__global__ void resolveContours(LabelType *dLabels, LabelType *dSegments, LabelType *dConnectedComponents,
                                MaskType *dNeighbors, IndexType height, IndexType width, IndexType batchSize,
                                IndexType *dConnectList, CountType *dNodeCount, CountType *contourCount)
{
    // Organize input/output data into structured objects for easier access.
    BoundaryLabel labels{batchSize, height, width, 1, dLabels};
    BoundaryLabel segments{batchSize, height, width, 1, dSegments};
    BoundaryLabel connectedComponents{batchSize, height, width, 1, dConnectedComponents};
    Neighborhood  neighbors{batchSize, height, width, 1, dNeighbors};
    ConnectList   connectList{batchSize, FindContours::MAX_NUM_CONTOURS, FindContours::MAX_CONTOUR_POINTS, 1,
                            dConnectList};
    CountList     nodeCount{batchSize, FindContours::MAX_NUM_CONTOURS, dNodeCount};

    // Initialize cooperative groups for thread synchronization.
    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();

    // Compute properties of the grid and blocks.
    auto gridBlocks  = grid.group_dim().x * grid.group_dim().y * grid.group_dim().z;
    auto blockHeight = block.group_dim().y;
    auto blockWidth  = block.group_dim().x;
    auto blockRank   = block.group_index().z * grid.group_dim().x * grid.group_dim().y
                   + block.group_index().y * grid.group_dim().x + block.group_index().x;

    // Calculate block tile dimensions and total number of iterations needed.
    auto blocksTileWidth  = util::DivUp(width, blockWidth);
    auto blocksTileHeight = util::DivUp(height, blockHeight);
    auto numSteps         = util::DivUp(blocksTileWidth * blocksTileHeight * batchSize, gridBlocks);

    // Calculate the thread's block dimensions and its position within the block.
    CoordType blockDims{blockWidth, blockHeight, 1};
    CoordType threadBlockPos{block.thread_rank() % blockWidth, block.thread_rank() / blockWidth, 0};

    const auto badLabel = height * width;

    // Traverse and label contours for each step.
    for (auto step = 0; step < numSteps; ++step)
    {
        // Calculate block's position within the grid.
        auto      blockIndex = blockRank + step * gridBlocks;
        CoordType blockGridPos{blockIndex % blocksTileWidth, (blockIndex / blocksTileWidth) % blocksTileHeight,
                               blockIndex / (blocksTileWidth * blocksTileHeight)};

        // Calculate the thread's global position.
        auto threadPos = blockGridPos * blockDims + threadBlockPos;

        // Check if thread's position is within image boundaries.
        bool inLabels = (threadPos.x < width && threadPos.y < height && threadPos.z < batchSize);

        // If within boundaries, traverse and label the contour for the current position.
        if (inLabels)
        {
            traverseContour(labels, segments, connectedComponents, neighbors, threadPos, connectList, nodeCount,
                            contourCount, badLabel);
        }
    }
}

__global__ void flattenContours(IndexType *dConnectList, CountType *dNodeCount, CountType *contourCount,
                                IndexType width, IndexType batchSize, NodeList nodeList, NodeCounts pointCount)
{
    // Structuring the input/output data
    ConnectList connectList{batchSize, FindContours::MAX_NUM_CONTOURS, FindContours::MAX_CONTOUR_POINTS, 1,
                            dConnectList};
    CountList   nodeCount{batchSize, FindContours::MAX_NUM_CONTOURS, dNodeCount};

    // Initialize cooperative groups for thread synchronization.
    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Compute properties of the grid and blocks.
    auto gridBlocks = grid.group_dim().x * grid.group_dim().y * grid.group_dim().z;
    auto blockRank  = block.group_index().z * grid.group_dim().x * grid.group_dim().y
                   + block.group_index().y * grid.group_dim().x + block.group_index().x;

    // Calculate block tile dimensions and total number of iterations needed.
    auto contourTile       = util::DivUp(FindContours::MAX_NUM_CONTOURS, warp.meta_group_size());
    auto neededThreads     = warp.size() * FindContours::MAX_NUM_CONTOURS * batchSize;
    auto neededBlocks      = (neededThreads + block.size() - 1) / block.size();
    auto numStepsBatchSize = ((batchSize * contourTile - blockRank) + gridBlocks - 1) / gridBlocks;
    auto numSteps          = max((neededBlocks + gridBlocks - 1) / gridBlocks, numStepsBatchSize);

    // Calculate the thread's block dimensions and its position within the block.
    CoordType blockDims{warp.size(), warp.meta_group_size(), 1};
    CoordType threadBlockPos{warp.thread_rank(), warp.meta_group_rank(), 0};

    // Traverse and label contours for each step.
    for (auto step = 0; step < numSteps; ++step)
    {
        // Calculate block's position within the grid.
        auto      blockIndex = blockRank + step * gridBlocks;
        CoordType blockGridPos{0, blockIndex % contourTile, blockIndex / contourTile};

        // Calculate the thread's global position.
        auto pos          = blockGridPos * blockDims + threadBlockPos;
        auto contourIndex = pos.y;

        // Make sure we're within the boundaries of our contour count
        if (pos.z < batchSize && contourIndex < contourCount[pos.z])
        {
            auto indexOffset = 0;
            for (auto i = 0; i < contourIndex; ++i)
            {
                indexOffset += *nodeCount.ptr(pos.z, i);
            }

            if ((indexOffset + *nodeCount.ptr(pos.z, contourIndex)) > FindContours::MAX_TOTAL_POINTS)
            {
                return;
            }

            for (auto i = pos.x; i < *nodeCount.ptr(pos.z, contourIndex); i += blockDims.x)
            {
                auto      point = *connectList.ptr(pos.z, contourIndex, i);
                PointType node{mod(point, width), point / width};
                *nodeList.ptr(static_cast<int>(pos.z), static_cast<int>(indexOffset + i), 0) = node.x;
                *nodeList.ptr(static_cast<int>(pos.z), static_cast<int>(indexOffset + i), 1) = node.y;
            }
            if (pos.x == 0)
            {
                *pointCount.ptr(static_cast<int>(pos.z), static_cast<int>(contourIndex))
                    = *nodeCount.ptr(pos.z, contourIndex);
            }
        }
    }
}

namespace detail {
template<typename Lambda, typename... Args>
void forwardArgs(Lambda &&f, Args &&...args)
{
    // Create a lambda to capture each forwarded arg, then use pack expansion
    // to expand and call the lambda for each arg.
    auto forwarder = [&f](auto &&...a)
    {
        (f(&a), ...);
    };
    forwarder(std::forward<Args>(args)...);
}

template<typename KernelFunction, typename... KernelParameters>
inline void cooperativeLaunch(const KernelFunction &func, cudaStream_t stream, dim3 grid, dim3 block, size_t sharedMem,
                              KernelParameters... params)
{
    void *args[sizeof...(params)];
    int   argIndex = 0;

    // Capture args by address into the args array
    forwardArgs([&](auto p) { args[argIndex++] = p; }, params...);

    cudaLaunchCooperativeKernel<KernelFunction>(&func, grid, block, args, sharedMem, stream);
}
} // namespace detail

template<typename PixelType>
__host__ void findContours_impl(DeviceImage<PixelType> &dImage, LabelType *dLabels, LabelType *dSegments,
                                LabelType *dConnectedComponents, MaskType *dNeighbors, IndexType *dConnectList,
                                CountType *dNodeCount, CountType *dContourCount, NodeList &dNodeList,
                                NodeCounts &dPointCount, IndexType height, IndexType width, IndexType batchSize,
                                cudaStream_t stream)
{
    // Determine shared memory size needed for labelEdges kernel, considering halo cells and storage.
    auto labelEdgesSharedMem = [&](int blockSize)
    {
        int dimX = 32;
        int dimY = static_cast<int>((blockSize + dimX - 1) / dimX);
        return (dimX + 4) * (dimY + 4) * sizeof(typename SharedImage::value_type)
             + dimX * dimY * sizeof(typename SharedLabel::value_type);
    };

    // Parameters for kernel launches
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    int  maxGridSize  = 1;
    int  maxBlockSize = 32;

    // 1. Labeling Image Edges:
    // Query for optimal block size for the labelEdges kernel.
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&maxGridSize, &maxBlockSize, labelEdges<PixelType>,
                                                                   labelEdgesSharedMem, 1024));
    block                 = dim3(32, (maxBlockSize + 31) / 32);
    auto blocksTileWidth  = util::DivUp(width, block.x);
    auto blocksTileHeight = util::DivUp(height, block.y);
    grid.x                = std::min(blocksTileWidth * blocksTileHeight * batchSize, maxGridSize);
    detail::cooperativeLaunch(labelEdges<PixelType>, stream, grid, block, labelEdgesSharedMem(block.x * block.y),
                              dImage, height, width, batchSize, dLabels);

    // 2. Labeling Connected Components:
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&maxGridSize, &maxBlockSize, labelConnectedComponents, 0, 1024));
    block            = dim3(32, (maxBlockSize + 31) / 32);
    blocksTileWidth  = util::DivUp(width, block.x);
    blocksTileHeight = util::DivUp(height, block.y);
    grid.x           = std::min(blocksTileWidth * blocksTileHeight * batchSize, maxGridSize);
    detail::cooperativeLaunch(labelConnectedComponents, stream, grid, block, 0, dLabels, height, width, batchSize,
                              dSegments, dConnectedComponents, dNeighbors);

    // 3. Resolving Contours:
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&maxGridSize, &maxBlockSize, resolveContours, 0, 1024));
    block            = dim3(32, (maxBlockSize + 31) / 32);
    blocksTileWidth  = util::DivUp(width, block.x);
    blocksTileHeight = util::DivUp(height, block.y);
    grid.x           = std::min(blocksTileWidth * blocksTileHeight * batchSize, maxGridSize);
    detail::cooperativeLaunch(resolveContours, stream, grid, block, 0, dLabels, dSegments, dConnectedComponents,
                              dNeighbors, height, width, batchSize, dConnectList, dNodeCount, dContourCount);

    // 4. Flattening Contours:
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&maxGridSize, &maxBlockSize, flattenContours, 0, 1024));
    auto idealThreads   = 32 * FindContours::MAX_NUM_CONTOURS * batchSize;
    auto bestBlockCount = util::DivUp(idealThreads, maxBlockSize);
    block               = dim3(32, (maxBlockSize + 31) / 32, 1);
    grid.x              = 1;
    grid.y              = std::min(bestBlockCount, maxGridSize);
    grid.z              = 1;
    detail::cooperativeLaunch(flattenContours, stream, grid, block, 0, dConnectList, dNodeCount, dContourCount, width,
                              batchSize, dNodeList, dPointCount);
}

// =============================================================================
// FindContours Class Definition
// =============================================================================

FindContours::FindContours(DataShape max_input_shape, DataShape max_output_shape)
    : CudaBaseOp(max_input_shape, max_output_shape)
{
    // Calculating the size of the workspace buffers
    auto gpuBufferSize = this->calBufferSize(max_input_shape, max_output_shape, kCV_8U);

    // Allocating GPU memory
    NVCV_CHECK_LOG(cudaMalloc(&gpu_workspace, gpuBufferSize));
}

FindContours::~FindContours()
{
    NVCV_CHECK_LOG(cudaFree(gpu_workspace));
}

size_t FindContours::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    // Number of images in the batch times...
    return max_input_shape.N
         * (
               // Size of labels buffer
               max_input_shape.H * max_input_shape.W * sizeof(LabelType) +

               // Size of segments buffer
               max_input_shape.H * max_input_shape.W * sizeof(LabelType) +

               // Size of connected components buffer
               max_input_shape.H * max_input_shape.W * sizeof(LabelType) +

               // Size of neighborhood flag buffer
               max_input_shape.H * max_input_shape.W * sizeof(MaskType) +

               // Size of maximum contours heads found
               FindContours::MAX_TOTAL_POINTS * sizeof(IndexType) +

               FindContours::MAX_NUM_CONTOURS * sizeof(CountType) +

               // Size of contour counter
               sizeof(CountType)

               // done...
         );
}

ErrorCode FindContours::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &pointCoords,
                              const TensorDataStridedCuda &numPoints, cudaStream_t stream)
{
    // Testing inData for valid structure
    auto format = GetLegacyDataFormat(inData.layout());
    if (format != kNHWC && format != kHWC)
    {
        LOG_ERROR("Invalid DataFormat for input image: " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto data_type = GetLegacyDataType(inData.dtype());
    if (!(data_type == kCV_8U /*|| data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F */))
    {
        LOG_ERROR("Invalid DataType for input image: " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    // Creating a access overlay for the input data
    auto inAccess = TensorDataAccessStridedImage::Create(inData);
    NVCV_ASSERT(inAccess);
    auto pointsAccess = TensorDataAccessStrided::Create(pointCoords);
    NVCV_ASSERT(pointsAccess);
    auto countAccess = TensorDataAccessStrided::Create(numPoints);
    NVCV_ASSERT(countAccess);

    // Extracting input shape information
    auto input_shape  = GetLegacyDataShape(inAccess->infoShape());
    auto points_shape = pointsAccess->infoShape();
    auto counts_shape = countAccess->infoShape();

    const auto nImage   = input_shape.N;
    const auto width    = input_shape.W;
    const auto height   = input_shape.H;
    const auto channels = input_shape.C;

    if (channels != 1)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (nImage != points_shape.shape()[0] || nImage != counts_shape.shape()[0])
    {
        LOG_ERROR("Invalid INVALID_PARAMETER: batch size must be equal for all parameters");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (points_shape.shape()[1] > FindContours::MAX_TOTAL_POINTS)
    {
        LOG_ERROR("Invalid INVALID_PARAMETER: points cannot be larger than the max total number of points");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (points_shape.shape()[2] != 2)
    {
        LOG_ERROR("Invalid INVALID_PARAMETER: points shape can only hold xy coordinates");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (counts_shape.shape()[1] > FindContours::MAX_NUM_CONTOURS)
    {
        LOG_ERROR("Invalid INVALID_PARAMETER: points cannot be larger than the max number of contours");
        return ErrorCode::INVALID_PARAMETER;
    }

    DeviceImage<uint8_t> dImage{
        reinterpret_cast<uint8_t *>(inAccess->sampleData(0)),
        {static_cast<int>(inAccess->sampleStride()), static_cast<int>(inAccess->rowStride()),
                                              static_cast<int>(inAccess->colStride())                                                                     },
        {                  static_cast<int>(nImage),                static_cast<int>(height), static_cast<int>(width)}
    };
    NodeList   dNodeList{pointCoords};
    NodeCounts dPointCount{numPoints};

    // Creating some temporaries
    char *bufferBoundaryStart = (char *)gpu_workspace;

    // Initialize buffer for the GPU image.

    // Initialize buffer for storing neighborhood indices.
    LabelType *dLabels = reinterpret_cast<LabelType *>(bufferBoundaryStart);
    bufferBoundaryStart += sizeof(LabelType) * input_shape.N * input_shape.H * input_shape.W;

    // Initialize buffer for storing segment boundaries.
    LabelType *dSegments = reinterpret_cast<LabelType *>(bufferBoundaryStart);
    bufferBoundaryStart += sizeof(LabelType) * input_shape.N * input_shape.H * input_shape.W;

    // Initialize buffer for storing connected component data.
    LabelType *dConnectedComponents = reinterpret_cast<LabelType *>(bufferBoundaryStart);
    bufferBoundaryStart += sizeof(LabelType) * input_shape.N * input_shape.H * input_shape.W;

    // Initialize buffer for storing neighbor mask data.
    MaskType *dNeighbors = reinterpret_cast<MaskType *>(bufferBoundaryStart);
    bufferBoundaryStart += sizeof(MaskType) * input_shape.N * input_shape.H * input_shape.W;

    // Initialize buffer to keep track of contours.
    IndexType *dConnectList = reinterpret_cast<IndexType *>(bufferBoundaryStart);
    bufferBoundaryStart += sizeof(IndexType) * input_shape.N * FindContours::MAX_TOTAL_POINTS;

    // Initialize buffer to keep track of contours.
    CountType *dNodeCount = reinterpret_cast<CountType *>(bufferBoundaryStart);
    bufferBoundaryStart += sizeof(CountType) * input_shape.N * FindContours::MAX_NUM_CONTOURS;

    // Initialize buffer for counting contours.
    CountType *dContourCount = reinterpret_cast<CountType *>(bufferBoundaryStart);
    bufferBoundaryStart += sizeof(CountType) * input_shape.N;

    // Clear GPU buffers to prepare for computation.
    checkCudaErrors(cudaMemsetAsync(reinterpret_cast<void *>(dLabels), height * width,
                                    nImage * sizeof(LabelType) * input_shape.H * input_shape.W, stream));
    checkCudaErrors(cudaMemsetAsync(reinterpret_cast<void *>(dSegments), height * width,
                                    nImage * sizeof(LabelType) * input_shape.H * input_shape.W, stream));
    checkCudaErrors(cudaMemsetAsync(reinterpret_cast<void *>(dConnectedComponents), height * width,
                                    nImage * sizeof(LabelType) * input_shape.H * input_shape.W, stream));
    checkCudaErrors(cudaMemsetAsync(reinterpret_cast<void *>(dNeighbors), 0,
                                    nImage * sizeof(MaskType) * input_shape.H * input_shape.W, stream));
    checkCudaErrors(cudaMemsetAsync(reinterpret_cast<void *>(dContourCount), 0, nImage * sizeof(CountType), stream));

    // get boundaries of the binary image, which is called contour.
    findContours_impl(dImage, dLabels, dSegments, dConnectedComponents, dNeighbors, dConnectList, dNodeCount,
                      dContourCount, dNodeList, dPointCount, height, width, nImage, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
