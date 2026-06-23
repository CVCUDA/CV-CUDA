/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
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

// ROCm/HIP compatibility layer for CV-CUDA. Force-included on every HIP
// translation unit (CMAKE_HIP_FLAGS -include) so the aliases below are in
// scope before any CV-CUDA or CUDA-toolkit header is parsed. The NVIDIA build
// never sees this file: the cmake/hip shim dir and this header are only on the
// HIP include path. Only the actual CUDA-runtime/library symbols CV-CUDA uses
// are aliased here; CV-CUDA's own cuda<Op>Submit/cuda<Op>Create public API
// names are deliberately left untouched.

#ifndef CVCUDA_HIP_COMPAT_H
#define CVCUDA_HIP_COMPAT_H

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)

#if defined(__cplusplus)
// libc host declarations must win over HIP's device overloads of memcpy/memset:
// pull them in before the HIP runtime so host TUs keep the standard host
// prototypes.
#include <cstdlib>
#include <cstring>

#if defined(__HIPCC__)
// .cu translation units (compiled by hipcc) need the full device runtime.
#include <hip/hip_runtime.h>
#else
// Plain g++ host TUs only need the runtime API (types + host-callable entry
// points); the device runtime header is heavier and pulls device builtins.
#include <hip/hip_runtime_api.h>
#endif
#endif

// CUDA_VERSION: several headers (Compat.hpp, Metaprogramming.hpp, StreamId.cpp)
// branch on it. CV-CUDA wants the pre-13.0 compound-type aliases, so report a
// version below 13000. Do NOT define __CUDA_ARCH__ on HIP: the SaturateCast PTX
// table and the NVCV SIMD-video-intrinsic paths are gated on __CUDA_ARCH__ and
// must stay inert, falling through to their portable C++/per-element bodies.
#ifndef CUDA_VERSION
#define CUDA_VERSION 12020
#endif

// ---- runtime: error/status -------------------------------------------------
#define cudaError_t                hipError_t
#define cudaError                  hipError_t
#define cudaSuccess                hipSuccess
#define cudaErrorNotReady          hipErrorNotReady
#define cudaErrorInvalidValue      hipErrorInvalidValue
#define cudaErrorMemoryAllocation  hipErrorOutOfMemory
#define cudaErrorCudartUnloading   hipErrorDeinitialized
#define cudaErrorTextureFetchFailed hipErrorInvalidTexture
#define cudaGetLastError           hipGetLastError
#define cudaPeekAtLastError        hipPeekAtLastError
#define cudaGetErrorString         hipGetErrorString
#define cudaGetErrorName           hipGetErrorName
#define cudaGetVersion             hipRuntimeGetVersion

// ---- runtime: device -------------------------------------------------------
#define cudaGetDevice              hipGetDevice
#define cudaSetDevice              hipSetDevice
#define cudaGetDeviceCount         hipGetDeviceCount
#define cudaDeviceSynchronize      hipDeviceSynchronize
#define cudaDeviceProp             hipDeviceProp_t
#define cudaGetDeviceProperties    hipGetDeviceProperties
#define cudaDevAttrTextureAlignment      hipDeviceAttributeTextureAlignment
#define cudaDevAttrTexturePitchAlignment hipDeviceAttributeTexturePitchAlignment

// NVCV derives a tensor/image row-pitch alignment from the texture *pitch*
// alignment device attribute. On NVIDIA that attribute is 32 bytes, so a tightly
// packed image (e.g. a 640-byte uchar row) keeps a 640-byte row stride. AMD
// reports 256 there, which would pad that row to 768 and silently change the
// in-memory layout every NVCV consumer (and the whole-buffer test comparisons)
// assumes. No CV-CUDA tensor is bound to a HW texture object, so the larger
// HW pitch is unnecessary here; clamp the queried pitch alignment to the NVIDIA
// value to keep the byte layout identical to the CUDA build. Other attribute
// queries pass through unchanged.
#if defined(__cplusplus)
__host__ inline hipError_t cvcuda_hipDeviceGetAttribute(int *value, hipDeviceAttribute_t attr, int device)
{
    hipError_t err = hipDeviceGetAttribute(value, attr, device);
    if (err == hipSuccess && attr == hipDeviceAttributeTexturePitchAlignment && value && *value > 32)
    {
        *value = 32;
    }
    return err;
}
#define cudaDeviceGetAttribute cvcuda_hipDeviceGetAttribute
#else
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif

// ---- runtime: memory -------------------------------------------------------
#define cudaMalloc                 hipMalloc
#define cudaMallocManaged          hipMallocManaged
#define cudaFree                   hipFree
#define cudaMallocHost             hipHostMalloc
#define cudaHostAlloc              hipHostMalloc
#define cudaFreeHost               hipHostFree
#define cudaHostFree               hipHostFree
#define cudaHostAllocMapped        hipHostMallocMapped
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#define cudaMemset                 hipMemset
#define cudaMemsetAsync            hipMemsetAsync
#define cudaMemset2D               hipMemset2D
#define cudaMemset2DAsync          hipMemset2DAsync
#define cudaMemcpy                 hipMemcpy
#define cudaMemcpyAsync            hipMemcpyAsync
#define cudaMemcpy2D               hipMemcpy2D
#define cudaMemcpy2DAsync          hipMemcpy2DAsync
#define cudaMemcpyKind             hipMemcpyKind
#define cudaMemcpyHostToDevice     hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice   hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost       hipMemcpyHostToHost
#define cudaMemcpyDefault          hipMemcpyDefault
#define cudaPointerAttributes      hipPointerAttribute_t
#define cudaPointerGetAttributes   hipPointerGetAttributes
#define cudaMemoryTypeHost         hipMemoryTypeHost
#define cudaMemoryTypeDevice       hipMemoryTypeDevice
#define cudaMemoryTypeManaged      hipMemoryTypeManaged
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered

// ---- runtime: stream / event ----------------------------------------------
#define cudaStream_t                       hipStream_t
#define cudaStreamDefault                  hipStreamDefault
#define cudaStreamPerThread                hipStreamPerThread
#define cudaStreamNonBlocking              hipStreamNonBlocking
#define cudaStreamSynchronize              hipStreamSynchronize
#define cudaStreamWaitEvent                hipStreamWaitEvent
#define cudaStreamDestroy                  hipStreamDestroy
#define cudaStreamGetId                    hipStreamGetId
#define cudaStreamCreate                   hipStreamCreate
#define cudaStreamCreateWithFlags          hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority       hipStreamCreateWithPriority
#define cudaDeviceGetStreamPriorityRange   hipDeviceGetStreamPriorityRange
#define cudaEvent_t                hipEvent_t
#define cudaEventDefault           hipEventDefault
#define cudaEventDisableTiming     hipEventDisableTiming
#define cudaEventCreate            hipEventCreate
#define cudaEventCreateWithFlags   hipEventCreateWithFlags
#define cudaEventRecord            hipEventRecord
#define cudaEventQuery             hipEventQuery
#define cudaEventSynchronize       hipEventSynchronize
#define cudaEventElapsedTime       hipEventElapsedTime
#define cudaEventDestroy           hipEventDestroy

// ---- full-wavefront mask ---------------------------------------------------
// __shfl*_sync on ROCm static_asserts a 64-bit mask regardless of wave width.
// The width argument (kept explicit at every call site that needs it) controls
// the subgroup; the mask just marks participants.
#define NVCV_WARP_FULL_MASK 0xffffffffffffffffULL

// ---- built-in index types --------------------------------------------------
// On CUDA blockIdx/blockDim/threadIdx are uint3/dim3, so the kernels' common
// idiom `blockIdx * blockDim + threadIdx` resolves through the cuda:: compound
// operators. On HIP these are distinct __hip_builtin_*_t structs (with only a
// dim3 conversion), so neither HIP's nor CV-CUDA's vector operators deduce them.
// Provide the exact whole-vector forms the kernels use, lowering to uint3. These
// take only the builtin index types, which carry no NVCV TypeTraits, so they do
// not compete with the cuda:: compound operators.
// Defined only under hipcc (where the __hip_builtin_*_t types exist and where
// __global__ bodies are parsed); plain g++ host TUs that include only the HIP
// runtime API do not see these builtin types, so the operators must not appear
// there. hipcc parses these in both its host and device passes.
#if defined(__cplusplus) && defined(__HIPCC__)
__host__ __device__ __forceinline__ uint3 operator*(const __hip_builtin_blockIdx_t &a,
                                                    const __hip_builtin_blockDim_t &b)
{
    return uint3{a.x * b.x, a.y * b.y, a.z * b.z};
}
__host__ __device__ __forceinline__ uint3 operator*(const __hip_builtin_blockDim_t &a,
                                                    const __hip_builtin_blockIdx_t &b)
{
    return uint3{a.x * b.x, a.y * b.y, a.z * b.z};
}
__host__ __device__ __forceinline__ uint3 operator+(const uint3 &a, const __hip_builtin_threadIdx_t &b)
{
    return uint3{a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ __forceinline__ uint3 operator+(const __hip_builtin_threadIdx_t &a, const uint3 &b)
{
    return uint3{a.x + b.x, a.y + b.y, a.z + b.z};
}
#endif

// std::declval is __host__-only (libstdc++); clang rejects it in the unevaluated
// decltype of a __device__-only function (where nvcc is lenient). Provide a
// __host__ __device__ equivalent for those few device-side decltype sites.
#if defined(__cplusplus)
namespace nvcv { namespace cuda { namespace compat {
template<typename T>
__host__ __device__ T &&declval() noexcept;
}}} // namespace nvcv::cuda::compat
#endif

#endif // __HIP_PLATFORM_AMD__ || USE_HIP

#endif // CVCUDA_HIP_COMPAT_H
