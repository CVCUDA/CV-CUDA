/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Assert.h"
#include "OpPairwiseMatcher.hpp"

#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cub/cub.cuh>

#include <sstream>

namespace {

// Utilities definitions -------------------------------------------------------

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

constexpr int kIntMax = cuda::TypeTraits<int>::max; // maximum number represented in an int

constexpr int kNumThreads = 64; // number of threads per block

// Key value pair type used in CUB (CUDA Unbound) sort and reduce-min operations
// The idea is to sort or get the minimum by distance (dist) and then by index (idx)
struct KeyValueT
{
    float dist;
    int   idx;
};

// Point class primary template is not intended to be used directly, instead only its partial specializations
// (below) are used, where: <T> is the point type, i.e. the tensor type of the set storing the points; <NB> is the
// maximum number of bytes hold by the point class as a cache from global memory (GMEM)
template<class T, int NB>
class PointT;

// Point class partial specialization for type T with NB = 0, a fall-through Point class meaning that n-dimensional
// points in a set are loaded directly from global memory (GMEM) without caching
template<class T>
class PointT<T, 0>
{
public:
    static constexpr int kMaxSize = 0; // maximum size in bytes of a single point stored by this class

    __device__ PointT() = default;

    inline __device__ void load(cuda::Tensor3DWrap<T> set, int sampleIdx, int setIdx, int numDim)
    {
        data = set.ptr(sampleIdx, setIdx);
    }

    inline __device__ T &operator[](int i) const
    {
        return data[i];
    }

private:
    T *data;
};

using RT = uint32_t; // resource type used in a cache inside the PointT class below

// Point class partial specialization for type T with any NB, meaning that n-dimensional points in a set are loaded
// from global memory (GMEM) and stored in a cache (can be registers or local memory or shared memory), thus this
// class can only be used by points with up to NB size in bytes, i.e. n * sizeof(T) <= NB
template<class T, int NB>
class PointT
{
    static_assert(NB > 0, "Maximum number of bytes capacity in PointT class must be positive");

public:
    static constexpr int kMaxSize = NB;              // maximum size in bytes of a single point stored by this class
    static constexpr int kNumElem = NB / sizeof(RT); // number of elements in array serving as a cache
    static constexpr int kMaxDims = NB / sizeof(T);  // maximum number of dimensions a single point may have

    __device__ PointT() = default;

    inline __device__ void load(cuda::Tensor3DWrap<T> set, int sampleIdx, int setIdx, int numDim)
    {
#pragma unroll
        for (int i = 0; i < kNumElem && i < util::DivUp(numDim * (int)sizeof(T), (int)sizeof(RT)); ++i)
        {
            data[i] = *reinterpret_cast<const RT *>(set.ptr(sampleIdx, setIdx, i * (int)(sizeof(RT) / sizeof(T))));
        }
    }

    inline __device__ T &operator[](int i) const
    {
        return reinterpret_cast<T *>(&data[0])[i];
    }

private:
    RT data[kNumElem];
};

// Is compatible checks if a {numDim}-dimensional point fits in the corresponding Point T class (above)
template<typename T, int NB>
inline __host__ bool isCompatible(int numDim)
{
    return (numDim * (int)sizeof(T)) <= NB;
}

// Get minimum stride is used to check if a {numDim}-dimensional Point of type T smaller than RT (the cache
// resource type in PointT class) can be read in steps of RT, allowing overflow after the last T element
template<typename T>
inline __host__ int getMinStride(int numDim)
{
    return util::DivUp(numDim * (int)sizeof(T), (int)sizeof(RT)) * (int)sizeof(RT);
}

// CUDA functions --------------------------------------------------------------

// Reduce-min by key a key-value pair for CUB (CUDA Unbound) to do block-wide reduction to minimum in the first thread
inline __device__ KeyValueT minkey(const KeyValueT &a, const KeyValueT &b)
{
    return (a.dist < b.dist || (a.dist == b.dist && a.idx < b.idx)) ? a : b;
}

// Absolute difference | a - b | for floating-point values
template<typename T, class = cuda::Require<std::is_floating_point_v<T>>>
inline __device__ T absdiff(T a, T b)
{
    return cuda::abs(a - b);
}

// Absolute difference for integral values, computing difference in unsigned types may lead to wrap around
template<typename T, class = cuda::Require<std::is_integral_v<T>>>
inline __device__ std::make_unsigned_t<T> absdiff(T a, T b)
{
    return a < b ? b - a : a - b; // wrapping around is fine!
}

// Compute {distance} between elements {e1} and {e2} from n-dimensional points p1 and p2
template<NVCVNormType NORM, typename T>
inline __device__ void ComputeDistance(float &distance, const T &e1, const T &e2)
{
    if constexpr (NORM == NVCV_NORM_HAMMING)
    {
        distance += __popc(e1 ^ e2);
    }
    else if constexpr (NORM == NVCV_NORM_L1)
    {
        distance += absdiff(e1, e2);
    }
    else
    {
        static_assert(NORM == NVCV_NORM_L2, "ComputeDistance accepts only HAMMING, L1 or L2 norms");

        float d = absdiff(e1, e2);

        distance = fma(d, d, distance); // square-root is postponed as not needed to find best matches
    }
}

// Sort pairs of (distance, index) one per thread from a fixed point p1 to all points p2 in set2 with numDim
// dimensions, each point is an array with numDim elements of source type ST, each set is an array of points, and
// the tensor is an array of sets where the sampleIdx selects the current set within it with set2Size points
template<NVCVNormType NORM, class Point, class SetWrapper>
inline __device__ void SortKeyValue(float &sortedDist, int &sortedIdx, const Point &p1, const SetWrapper &set2,
                                    int numDim, int matchesPerPoint, int sampleIdx, int set2Size)
{
    sortedDist = cuda::TypeTraits<float>::max;
    sortedIdx  = -1;

    float curDist;
    Point p2;

    for (int set2Idx = threadIdx.x; set2Idx < set2Size; set2Idx += kNumThreads)
    {
        p2.load(set2, sampleIdx, set2Idx, numDim);

        curDist = 0.f;

        if constexpr (Point::kMaxSize > 0)
        {
#pragma unroll
            for (int i = 0; i < Point::kMaxDims && i < numDim; ++i)
            {
                ComputeDistance<NORM>(curDist, p1[i], p2[i]);
            }
        }
        else
        {
            for (int i = 0; i < numDim; ++i)
            {
                ComputeDistance<NORM>(curDist, p1[i], p2[i]);
            }
        }

        if (curDist < sortedDist)
        {
            sortedDist = curDist;
            sortedIdx  = set2Idx;
        }
    }

    __syncthreads(); // wait for all the threads to complete their local sorted (distance, index) pair

    if (matchesPerPoint == 1) // fast path for top-1 sort is reduce minimum
    {
        using BlockReduce = cub::BlockReduce<KeyValueT, kNumThreads>;

        __shared__ typename BlockReduce::TempStorage cubTempStorage;

        KeyValueT keyValue{sortedDist, sortedIdx};

        KeyValueT minKeyValue = BlockReduce(cubTempStorage).Reduce(keyValue, minkey);

        if (threadIdx.x == 0)
        {
            sortedDist = minKeyValue.dist;
            sortedIdx  = minKeyValue.idx;
        }
    }
    else // normal path to get top-N where N > 1 requires block sort
    {
        using BlockSort = cub::BlockRadixSort<float, kNumThreads, 1, int>;

        __shared__ typename BlockSort::TempStorage cubTempStorage;

        float keys[1]   = {sortedDist};
        int   values[1] = {sortedIdx};

        BlockSort(cubTempStorage).Sort(keys, values);

        if (threadIdx.x < matchesPerPoint)
        {
            sortedDist = keys[0];
            sortedIdx  = values[0];
        }
    }
}

// Write a match of (set1Idx, set2Idx) with (distance) found at matchIdx inside output matches and distances
template<NVCVNormType NORM>
inline __device__ void WriteMatch(int matchIdx, int set1Idx, int set2Idx, int sampleIdx, float &distance,
                                  cuda::Tensor3DWrap<int> matches, cuda::Tensor2DWrap<float> distances)
{
    *matches.ptr(sampleIdx, matchIdx, 0) = set1Idx;
    *matches.ptr(sampleIdx, matchIdx, 1) = set2Idx;

    if (distances.ptr(0) != nullptr)
    {
        if constexpr (NORM == NVCV_NORM_L2)
        {
            distance = cuda::sqrt(distance); // square-root was postpone for writing time, which is now
        }

        *distances.ptr(sampleIdx, matchIdx) = distance;
    }
}

// Brute-force matcher finds closest pairs of n-dimensional points in set1 and set2, comparing all against all, it
// is instantiated by: <NB> an upper limit of each point size in bytes; <NORM> type; and <ST> source type
template<int NB, NVCVNormType NORM, typename ST>
__global__ void BruteForceMatcher(cuda::Tensor3DWrap<ST> set1, cuda::Tensor3DWrap<ST> set2,
                                  cuda::Tensor1DWrap<const int> numSet1, cuda::Tensor1DWrap<const int> numSet2,
                                  cuda::Tensor3DWrap<int> matches, cuda::Tensor1DWrap<int> numMatches,
                                  cuda::Tensor2DWrap<float> distances, int set1Capacity, int set2Capacity,
                                  int outCapacity, int numDim, bool crossCheck, int matchesPerPoint)
{
    int sampleIdx = blockIdx.x;
    int set1Idx   = blockIdx.y;
    int set1Size  = set1Capacity;

    if (numSet1.ptr(0) != nullptr)
    {
        set1Size = numSet1[sampleIdx];
        set1Size = set1Size > set1Capacity ? set1Capacity : set1Size;
    }

    if (set1Idx >= set1Size)
    {
        return;
    }

    int set2Size = set2Capacity;

    if (numSet2.ptr(0) != nullptr)
    {
        set2Size = numSet2[sampleIdx];
        set2Size = set2Size > set2Capacity ? set2Capacity : set2Size;
    }

    PointT<ST, NB> p;

    p.load(set1, sampleIdx, set1Idx, numDim);

    float dist;
    int   set2Idx;

    SortKeyValue<NORM>(dist, set2Idx, p, set2, numDim, matchesPerPoint, sampleIdx, set2Size);

    if (crossCheck)
    {
        __shared__ int set2Idx2;

        if (threadIdx.x == 0)
        {
            set2Idx2 = set2Idx;
        }

        __syncthreads(); // wait the first thread to communicate the best match in set2 index

        p.load(set2, sampleIdx, set2Idx2, numDim);

        float dist2;
        int   set1Idx2;

        SortKeyValue<NORM>(dist2, set1Idx2, p, set1, numDim, matchesPerPoint, sampleIdx, set1Size);

        if (threadIdx.x == 0 && set1Idx2 == set1Idx)
        {
            int matchIdx = atomicAdd(numMatches.ptr(sampleIdx), 1);

            if (matchIdx < outCapacity)
            {
                WriteMatch<NORM>(matchIdx, set1Idx, set2Idx, sampleIdx, dist, matches, distances);
            }
        }
    }
    else
    {
        if (threadIdx.x < matchesPerPoint)
        {
            int matchIdx = set1Idx * matchesPerPoint + threadIdx.x;

            if (matchIdx < outCapacity)
            {
                WriteMatch<NORM>(matchIdx, set1Idx, set2Idx, sampleIdx, dist, matches, distances);
            }
        }
    }
}

// Write number of matches in the case without cross check this number is set1 size times matches per point
__global__ void WriteNumMatches(cuda::Tensor1DWrap<const int> numSet1, cuda::Tensor1DWrap<int> numMatches,
                                int set1Capacity, int matchesPerPoint)
{
    int sampleIdx = blockIdx.x;
    int set1Size  = (numSet1.ptr(0) == nullptr) ? set1Capacity : numSet1[sampleIdx];

    numMatches[sampleIdx] = set1Size * matchesPerPoint;
}

// Run functions ---------------------------------------------------------------

// Run brute-force matcher, using NORM type for distance calculations and SrcT is the input source data type
template<NVCVNormType NORM, typename SrcT>
inline void RunBruteForceMatcherForNorm(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                                        const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2,
                                        const nvcv::Tensor &matches, const nvcv::Tensor &numMatches,
                                        const nvcv::Tensor &distances, bool crossCheck, int matchesPerPoint)
{
    cuda::Tensor3DWrap<const SrcT>    w_set1, w_set2; // tensor wraps of set1 and set2 and other tensors
    cuda::Tensor1DWrap<const int32_t> w_numSet1, w_numSet2;
    cuda::Tensor3DWrap<int32_t>       w_matches;
    cuda::Tensor1DWrap<int32_t>       w_numMatches;
    cuda::Tensor2DWrap<float>         w_distances;

#define CVCUDA_BFM_WRAP(TENSOR)                                                                                     \
    if (TENSOR)                                                                                                     \
    {                                                                                                               \
        auto data = TENSOR.exportData<nvcv::TensorDataStridedCuda>();                                               \
        if (!data)                                                                                                  \
        {                                                                                                           \
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, #TENSOR " tensor must be cuda-accessible"); \
        }                                                                                                           \
        w_##TENSOR = decltype(w_##TENSOR)(*data);                                                                   \
    }

    CVCUDA_BFM_WRAP(set1);
    CVCUDA_BFM_WRAP(set2);

    CVCUDA_BFM_WRAP(numSet1);
    CVCUDA_BFM_WRAP(numSet2);

    CVCUDA_BFM_WRAP(matches);
    CVCUDA_BFM_WRAP(numMatches);

    CVCUDA_BFM_WRAP(distances);

#undef CVCUDA_BFM_WRAP

    int numSamples   = set1.shape()[0];            // number of samples, where each sample is a set of points
    int set1Capacity = set1.shape()[1];            // set capacity is the maximum allowed number of points in set1
    int set2Capacity = set2.shape()[1];            // set capacity is the maximum allowed number of points in set2
    int numDim       = set1.shape()[2];            // number of dimensions of each n-dimensional point in set1 and set2
    int outCapacity  = matches.shape()[1];         // output capacity to store matches and distances
    int minStride    = getMinStride<SrcT>(numDim); // minimum stride in sets to allow the usage of PointT class

    dim3 threads(kNumThreads, 1, 1);
    dim3 blocks1(numSamples, 1, 1);
    dim3 blocks2(numSamples, set1Capacity, 1);

    if (crossCheck)
    {
        // Cross check returns a varying number of matches, as a match is only valid if it is the best (closest)
        // match from set1 to set2 and back from set2 to set1, the numMatches output starts at zero and is
        // atomically incremented in the BruteForceMatcher kernel

        NVCV_CHECK_THROW(cudaMemsetAsync(w_numMatches.ptr(0), 0, sizeof(int32_t) * numSamples, stream));
    }
    else
    {
        // Without cross check has a fixed number of matches equal to the set1 size, meaning for every point in
        // set1 there is (are) one (or more) matche(s) (up to matchesPerPoint) in set2

        if (numMatches)
        {
            WriteNumMatches<<<blocks1, threads, 0, stream>>>(w_numSet1, w_numMatches, set1Capacity, matchesPerPoint);
        }
    }

    // Cache-based kernel specialization: numDim and SrcT must fit a cache in PointT class; it works for 32B and
    // 128B descriptors, such as ORB and SIFT.  Even though it has 256 bytes spill loads/stores for NB = 128, it
    // still gives almost 2x performance benefit.

    // TODO: The caveat of below kernel specializations is that it takes time to compile (~30sec) and it does not
    //       cover points bigger than 128B in size, incurring in low performance for big points.  It may be better
    //       to use shared memory for those big points, given a certain maximum point dimension, and use threads to
    //       compute per element results instead of per point.

#define CVCUDA_BFM_RUN(NB)                                                                                      \
    BruteForceMatcher<NB, NORM><<<blocks2, threads, 0, stream>>>(                                               \
        w_set1, w_set2, w_numSet1, w_numSet2, w_matches, w_numMatches, w_distances, set1Capacity, set2Capacity, \
        outCapacity, numDim, crossCheck, matchesPerPoint);                                                      \
    return

    if (w_set1.strides()[1] >= minStride && w_set2.strides()[1] >= minStride)
    {
        if (isCompatible<SrcT, 32>(numDim))
        {
            CVCUDA_BFM_RUN(32);
        }
        else if (isCompatible<SrcT, 128>(numDim))
        {
            CVCUDA_BFM_RUN(128);
        }
    }

    CVCUDA_BFM_RUN(0);

#undef CVCUDA_BFM_RUN
}

template<typename SrcT>
inline void RunBruteForceMatcherForType(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                                        const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2,
                                        const nvcv::Tensor &matches, const nvcv::Tensor &numMatches,
                                        const nvcv::Tensor &distances, bool crossCheck, int matchesPerPoint,
                                        NVCVNormType normType)
{
    switch (normType)
    {
    case NVCV_NORM_HAMMING:
        if constexpr (std::is_floating_point_v<SrcT>)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid norm Hamming with float input type");
        }
        else
        {
            RunBruteForceMatcherForNorm<NVCV_NORM_HAMMING, SrcT>(stream, set1, set2, numSet1, numSet2, matches,
                                                                 numMatches, distances, crossCheck, matchesPerPoint);
        }
        break;

#define CVCUDA_BFM_CASE(NORM)                                                                                         \
    case NORM:                                                                                                        \
        RunBruteForceMatcherForNorm<NORM, SrcT>(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, \
                                                crossCheck, matchesPerPoint);                                         \
        break

        CVCUDA_BFM_CASE(NVCV_NORM_L1);
        CVCUDA_BFM_CASE(NVCV_NORM_L2);

#undef CVCUDA_BFM_CASE

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid norm type");
    }
}

inline void RunBruteForceMatcher(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                                 const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2, const nvcv::Tensor &matches,
                                 const nvcv::Tensor &numMatches, const nvcv::Tensor &distances, bool crossCheck,
                                 int matchesPerPoint, NVCVNormType normType)
{
    switch (set1.dtype())
    {
#define CVCUDA_BFM_CASE(DT, T)                                                                               \
    case nvcv::TYPE_##DT:                                                                                    \
        RunBruteForceMatcherForType<T>(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, \
                                       crossCheck, matchesPerPoint, normType);                               \
        break

        CVCUDA_BFM_CASE(U8, uint8_t);
        CVCUDA_BFM_CASE(U32, uint32_t);
        CVCUDA_BFM_CASE(F32, float);

#undef CVCUDA_BFM_CASE

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

PairwiseMatcher::PairwiseMatcher(NVCVPairwiseMatcherType algoChoice)
    : m_algoChoice(algoChoice)
{
    // Support additional algorithms here (only brute force for now), they may require payload
    if (algoChoice != NVCV_BRUTE_FORCE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid algorithm choice");
    }
}

// Tensor operator -------------------------------------------------------------

void PairwiseMatcher::operator()(cudaStream_t stream, const nvcv::Tensor &set1, const nvcv::Tensor &set2,
                                 const nvcv::Tensor &numSet1, const nvcv::Tensor &numSet2, const nvcv::Tensor &matches,
                                 const nvcv::Tensor &numMatches, const nvcv::Tensor &distances, bool crossCheck,
                                 int matchesPerPoint, NVCVNormType normType)
{
    // Check each input and output tensor and their properties are conforming to what is expected

    if (!set1 || !set2 || !matches)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Required tensors: set1 set2 matches");
    }
    if (set1.rank() != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input set1 must be a rank-3 tensor");
    }
    if (set2.rank() != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input set2 must be a rank-3 tensor");
    }
    if (set1.dtype() != set2.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input sets must have the same data type");
    }

    int64_t numSamples = set1.shape()[0];
    int64_t numDim     = set1.shape()[2];

    if (set2.shape()[0] != numSamples || set2.shape()[2] != numDim)
    {
        std::ostringstream oss;
        oss << (set2 ? set2.shape() : nvcv::TensorShape());
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid set2 shape %s is not [NMD]: N=%ld D=%ld",
                              oss.str().c_str(), numSamples, numDim);
    }

    if (numSamples > kIntMax || numDim > kIntMax || set1.shape()[1] > kIntMax || set2.shape()[1] > kIntMax)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Too big input tensors, shape > %d", kIntMax);
    }

    if (numSet1
        && ((numSet1.rank() != 1 && numSet1.rank() != 2) || numSet1.shape()[0] != numSamples
            || (numSet1.rank() == 2 && numSet1.shape()[1] != 1) || numSet1.dtype() != nvcv::TYPE_S32))
    {
        std::ostringstream oss;
        oss << numSet1.shape();
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid numSet1 shape %s dtype %s are not [N] or [NC]: N=%ld C=1 dtype=S32",
                              oss.str().c_str(), nvcvDataTypeGetName(numSet1.dtype()), numSamples);
    }

    if (numSet2
        && ((numSet2.rank() != 1 && numSet2.rank() != 2) || numSet2.shape()[0] != numSamples
            || (numSet2.rank() == 2 && numSet2.shape()[1] != 1) || numSet2.dtype() != nvcv::TYPE_S32))
    {
        std::ostringstream oss;
        oss << numSet2.shape();
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid numSet2 shape %s dtype %s are not [N] or [NC]: N=%ld C=1 dtype=S32",
                              oss.str().c_str(), nvcvDataTypeGetName(numSet2.dtype()), numSamples);
    }

    if (matches.rank() != 3 || matches.shape()[0] != numSamples || matches.shape()[1] >= kIntMax
        || matches.shape()[2] != 2 || matches.dtype() != nvcv::TYPE_S32)
    {
        std::ostringstream oss;
        oss << matches.shape();
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid matches shape %s dtype %s are not [NMA]: N=%ld M<%d A=2 dtype=S32",
                              oss.str().c_str(), nvcvDataTypeGetName(matches.dtype()), numSamples, kIntMax);
    }

    if (numMatches
        && ((numMatches.rank() != 1 && numMatches.rank() != 2) || numMatches.shape()[0] != numSamples
            || (numMatches.rank() == 2 && numMatches.shape()[1] != 1) || numMatches.dtype() != nvcv::TYPE_S32))
    {
        std::ostringstream oss;
        oss << numMatches.shape();
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid numMatches shape %s dtype %s are not [N] or [NC]: N=%ld C=1 dtype=S32",
                              oss.str().c_str(), nvcvDataTypeGetName(numMatches.dtype()), numSamples);
    }

    int64_t outCapacity = matches.shape()[1];

    if (distances
        && ((distances.rank() != 2 && distances.rank() != 3) || distances.shape()[0] != numSamples
            || distances.shape()[1] != outCapacity || (distances.rank() == 3 && distances.shape()[2] != 1)
            || distances.dtype() != nvcv::TYPE_F32))
    {
        std::ostringstream oss;
        oss << distances.shape();
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid distances shape %s dtype %s are not [NM] or [NMC]: N=%ld M=%ld C=1 dtype=S32",
                              oss.str().c_str(), nvcvDataTypeGetName(distances.dtype()), numSamples, outCapacity);
    }

    if (matchesPerPoint <= 0 || matchesPerPoint > kNumThreads)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid matchesPerPoint %d is not in [1, %d]",
                              matchesPerPoint, kNumThreads);
    }
    if (crossCheck && matchesPerPoint != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid matchesPerPoint %d for crossCheck=true is not 1", matchesPerPoint);
    }
    if (crossCheck && !numMatches)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid numMatches=NULL for crossCheck=true");
    }

    if (m_algoChoice == NVCV_BRUTE_FORCE)
    {
        RunBruteForceMatcher(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, crossCheck,
                             matchesPerPoint, normType);
    }
}

} // namespace cvcuda::priv
