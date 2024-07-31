/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file Allocator.h
 *
 * @brief Defines the public C interface to NVCV resource allocators.
 *
 * Allocators objects allow users to override the resource
 * allocation strategy used by several NVCV entities, such as images,
 * operators, etc.
 *
 * NVCV currently support three resource types:
 * - host memory: memory directly accessible by CPU.
 * - cuda memory : memory directly accessible by cuda-enabled GPU.
 * - host pinned memory: memory directly accessible by both CPU and cuda-enabled GPU.
 *
 * By default, the following functions are used to allocate
 * and deallocate resourcees for these types.
 *
 * @anchor default_res_allocators
 *
 * | Resource type      | Malloc        | Free         |
 * |--------------------|---------------|--------------|
 * | host memory        | malloc        | free         |
 * | cuda memory        | cudaMalloc    | cudaFree     |
 * | host pinned memory | cudaHostAlloc | cudaHostFree |
 *
 * By using defining custom resource allocators, user can override the allocation
 * and deallocation functions used for each resource type. When overriding, they can pass
 * a pointer to some user-defined context. It'll be passed unchanged to the
 * corresponding malloc and free function. This allows passing, for instance, a
 * pointer to an object whose methods will be called from inside the overriden
 * functions.
 */

#ifndef NVCV_ALLOCATOR_H
#define NVCV_ALLOCATOR_H

#include "../Export.h"
#include "../Status.h"
#include "Fwd.h"

#include <stdalign.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Function type for memory resource allocation.
 *
 * @param [in] ctx        Pointer to user context.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        It's guaranteed that >= 0 and it's an integral
 *                        multiple of alignBytes.
 * @param [in] alignBytes Address alignment in bytes.
 *                        It's guaranteed to be a power of two.
 *                        The returned address will be multiple of this value.
 *
 * @returns Pointer to allocated memory buffer.
 *          Must return NULL if buffer cannot be allocated.
 */
typedef void *(*NVCVMemAllocFunc)(void *ctx, int64_t sizeBytes, int32_t alignBytes);

/** Function type for memory deallocation.
 *
 * @param [in] ctx        Pointer to user context.
 * @param [in] ptr        Pointer to memory buffer to be deallocated.
 *                        If NULL, the operation must do nothing, successfully.
 * @param [in] sizeBytes, alignBytes Parameters passed during buffer allocation.
 */
typedef void (*NVCVMemFreeFunc)(void *ctx, void *ptr, int64_t sizeBytes, int32_t alignBytes);

/** Memory types handled by the memory resource allocator. */
typedef enum
{
    NVCV_RESOURCE_MEM_HOST,       /**< Memory accessible by host (CPU). */
    NVCV_RESOURCE_MEM_CUDA,       /**< Memory accessible by cuda (GPU). */
    NVCV_RESOURCE_MEM_HOST_PINNED /**< Memory accessible by both host and cuda. */
} NVCVResourceType;

#define NVCV_NUM_RESOURCE_TYPES (3)

typedef struct NVCVCustomMemAllocatorRec
{
    /** Pointer to function that performs memory allocation.
     *  + Function must return memory buffer with type specified by memType.
     *  + Cannot be NULL.
     */
    NVCVMemAllocFunc fnAlloc;

    /** Pointer to function that performs memory deallocation.
     *  + Function must deallocate memory allocated by @ref fnMemAlloc.
     *  + Cannot be NULL.
     */
    NVCVMemFreeFunc fnFree;
} NVCVCustomMemAllocator;

typedef union NVCVCustomResourceAllocatorRec
{
    NVCVCustomMemAllocator mem;
} NVCVCustomResourceAllocator;

typedef struct NVCVResourceAllocatorRec NVCVResourceAllocator;

/** Custom allocator cleanup function type */
typedef void (*NVCVResourceAllocatorCleanupFunc)(void *ctx, NVCVResourceAllocator *data);

struct NVCVResourceAllocatorRec
{
    /** Pointer to user context.
     *  It's passed unchanged to memory allocation/deallocation functions.
     *  It can be NULL, in this case no context is passed in.
     */
    void *ctx;

    /** Type of memory being handled by fnMemAlloc and fnMemFree. */
    NVCVResourceType resType;

    NVCVCustomResourceAllocator res;

    NVCVResourceAllocatorCleanupFunc cleanup;
};

typedef struct NVCVAllocator *NVCVAllocatorHandle;

/** Constructs a custom allocator instance in the given storage.
 *
 * The constructed allocator is configured to use the default resource
 * allocator functions specified @ref default_mem_allocators "here".
 *
 * When not needed anymore, the allocator instance must be destroyed by
 * @ref nvcvAllocatorDecRef function.
 *
 * @param [in] customAllocators    Array of custom resource allocators.
 *                                 + There must be at most one custom allocator for each memory type.
 *                                 + Restrictions on the custom allocator members apply,
 *                                   see \ref NVCVResourceAllocator.
 *
 * @param [in] numCustomAllocators Number of custom allocators in the array.
 *
 * @param [out] halloc Where new instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the allocator.
 * @retval #NVCV_SUCCESS                Allocator created successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorConstructCustom(const NVCVResourceAllocator *customAllocators,
                                                    int32_t numCustomAllocators, NVCVAllocatorHandle *handle);

/** Decrements the reference count of an existing allocator instance.
 *
 * The allocator is destroyed when its reference count reaches zero.
 *
 * @note All objects that depend on the allocator instance must already be destroyed,
 *       if not undefined behavior will ensue, possibly segfaults.
 *
 * @param [in] handle       Allocator to be destroyed.
 *                          If NULL, no operation is performed, successfully.
 *                          + The handle must have been created with any of the nvcvAllocatorConstruct functions.
 *
 * @param [out] newRefCount The decremented reference count. If the return value is 0, the object was destroyed.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorDecRef(NVCVAllocatorHandle handle, int *newRefCount);

/** Increments the reference count of an allocator.
 *
 * @param [in] handle       Allocator to be retained.
 *
 * @param [out] newRefCount The incremented reference count.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorIncRef(NVCVAllocatorHandle handle, int *newRefCount);

/** Returns the current reference count of an allocator
 *
 * @param [in] handle       The handle whose reference count is to be obtained.
 *
 * @param [out] newRefCount The reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorRefCount(NVCVAllocatorHandle handle, int *newRefCount);

/** Gets a custom allocator descriptor for given resource type.
 *
 * Retrieves a resource descriptor
 *
 * @param handle        The allocator handle
 * @param resType       The resource type for which to get the descriptor
 * @param [out] result  The underlying custom allocator
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid or there's no allocator that corresponds to resType.
 * @retval #NVCV_SUCCESS                Allocator created successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorGet(NVCVAllocatorHandle handle, NVCVResourceType resType,
                                        NVCVResourceAllocator *result);

/** Associates a user pointer to the allocator handle.
 *
 * This pointer can be used to associate any kind of data with the allocator object.
 *
 * @param [in] handle Allocator to be associated with the user pointer.
 *
 * @param [in] userPtr User pointer.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorSetUserPointer(NVCVAllocatorHandle handle, void *userPtr);

/** Returns the user pointer associated with the allocator handle.
 *
 * If no user pointer was associated, it'll return a pointer to NULL.
 *
 * @param [in] handle Allocator to be queried.
 *
 * @param [in] outUserPtr Pointer to where the user pointer will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorGetUserPointer(NVCVAllocatorHandle handle, void **outUserPtr);

/** Allocates a memory buffer of a host-accessible memory.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the resource allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [out] ptr       Holds a pointer to the allocated buffer.
 *                        + Cannot be NULL.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        + Must be >= 0.
 *                        + Must be an integral multiple of @p alignBytes.
 * @param [in] alignBytes Address alignment in bytes.
 *                        The returned address will be multiple of this value.
 *                        + Must a power of 2.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough free memory.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorAllocHostMemory(NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes,
                                                    int32_t alignBytes);

/** Frees a host-accessible memory buffer.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the memory allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [in] memType    Type of memory to be freed.
 * @param [in] ptr        Pointer to the memory buffer to be freed.
 *                        It can be NULL. In this case, no operation is performed.
 *                        + Must have been allocated by @ref nvcvAllocatorAllocHostMemory.
 * @param [in] sizeBytes,alignBytes Parameters passed during buffer allocation.
 *                                  + Not passing the exact same parameters
 *                                    passed during allocation will lead to undefined behavior.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorFreeHostMemory(NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes,
                                                   int32_t alignBytes);

/** Allocates a memory buffer of both host- and cuda-accessible memory.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the resource allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [out] ptr       Holds a pointer to the allocated buffer.
 *                        + Cannot be NULL.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        + Must be >= 0.
 *                        + Must be an integral multiple of @p alignBytes.
 * @param [in] alignBytes Address alignment in bytes.
 *                        The returned address will be multiple of this value.
 *                        + Must a power of 2.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough free memory.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorAllocHostPinnedMemory(NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes,
                                                          int32_t alignBytes);

/** Frees a both host- and cuda-accessible memory buffer.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the memory allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [in] memType    Type of memory to be freed.
 * @param [in] ptr        Pointer to the memory buffer to be freed.
 *                        It can be NULL. In this case, no operation is performed.
 *                        + Must have been allocated by @ref nvcvAllocatorAllocHostPinnedMemory.
 * @param [in] sizeBytes,alignBytes Parameters passed during buffer allocation.
 *                                  + Not passing the exact same parameters
 *                                    passed during allocation will lead to undefined behavior.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorFreeHostPinnedMemory(NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes,
                                                         int32_t alignBytes);

/** Allocates a memory buffer of cuda-accessible memory.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the resource allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [out] ptr       Holds a pointer to the allocated buffer.
 *                        + Cannot be NULL.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        + Must be >= 0.
 *                        + Must be an integral multiple of @p alignBytes.
 * @param [in] alignBytes Address alignment in bytes.
 *                        The returned address will be multiple of this value.
 *                        + Must a power of 2.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough free memory.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorAllocCudaMemory(NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes,
                                                    int32_t alignBytes);

/** Frees a cuda-accessible memory buffer.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the memory allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [in] memType    Type of memory to be freed.
 * @param [in] ptr        Pointer to the memory buffer to be freed.
 *                        It can be NULL. In this case, no operation is performed.
 *                        + Must have been allocated by @ref nvcvAllocatorAllocCudaMemory.
 * @param [in] sizeBytes,alignBytes Parameters passed during buffer allocation.
 *                                  + Not passing the exact same parameters
 *                                    passed during allocation will lead to undefined behavior.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorFreeCudaMemory(NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes,
                                                   int32_t alignBytes);

/** Returns a string representation of the resource type.
 *
 * @param[in] resource Resource type whose name is to be returned.
 *
 * @returns The string representation of the resource type.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvResourceTypeGetName(NVCVResourceType resource);

#ifdef __cplusplus
}
#endif

#endif // NVCV_ALLOCATOR_H
