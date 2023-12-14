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

#ifndef CVCUDAERATORS_WORKSPACE_H
#define CVCUDAERATORS_WORKSPACE_H

#include <cuda_runtime_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Defines requirements for workspace memory
 */
typedef struct NVCVWorkspaceMemRequirementsRec
{
    /** Size, in bytes, of the required memory */
    size_t size;
    /** Alignment, in bytes, of the required memory */
    size_t alignment;
} NVCVWorkspaceMemRequirements;

/** Aggregates requirements for all resource kinds in a workspace
 */
typedef struct NVCVWorkspaceRequirementsRec
{
    /** Requirements for plain host memory */
    NVCVWorkspaceMemRequirements hostMem;
    /** Requirements for GPU-accessible host memory (e.g. allocated with cudaHostAlloc) */
    NVCVWorkspaceMemRequirements pinnedMem;
    /** Requirements for GPU memory */
    NVCVWorkspaceMemRequirements cudaMem;
} NVCVWorkspaceRequirements;

/** Memory block for use in a workspace object.
 *
 * A workspace memory structure contains the requriements (these can be useful when obtaining memory from the workspace)
 * a pointer to the memory object and an optional CUDA event object which notifies that the memory is ready to use.
 *
 */
typedef struct NVCVWorkspaceMemRec
{
    /** The requirements that the memory pointed to by `data` must satisfy */
    NVCVWorkspaceMemRequirements req;

    /** The pointer to the workspace memory.
     *
     * @remark The accessibility of the memory may be restricted to the host or a specific device.
     */
    void *data;

    /** The event which notifies that the memory is ready to use.
     *
     * The event object is used in two ways - the user (e.g. an operator) of the workspace memory should wait for the
     * event in the context in which it will use the memory as well as record the event after it has scheduled all work
     * that uses the memory object.
     */
    cudaEvent_t ready;
} NVCVWorkspaceMem;

/** Aggregates multiple resources into a single workspace objects */
typedef struct NVCVWorkspaceRec
{
    /** Plain host memory. This should not be used in any GPU code.
     *
     * On systems with a discrete GPU, this kind of memory doesn't need a CUDA event. On systems with integrated GPU
     * or HMM systems, there's no difference between plain and pinned host memory with respect to synchronization.
     */
    NVCVWorkspaceMem hostMem;

    /** Pinned host memory.
     *
     * cudaXxxAsync operations on this kind of memory are performed truly asynchronously, which calls for
     * synchronization.
     * When used as a staging buffer for passing data to a CUDA kernel, a typical synchronization scheme would be to
     * wait for the `ready` event on host (cudaEventSynchronize), issue H2D copy and record the `ready` event.
     */
    NVCVWorkspaceMem pinnedMem;

    /** GPU memory */
    NVCVWorkspaceMem cudaMem;
} NVCVWorkspace;

#ifdef __cplusplus
}
#endif

#endif /* CVCUDAERATORS_WORKSPACE_H */
