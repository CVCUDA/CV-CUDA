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

#include "StreamId.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>

using cuStreamGetId_t = CUresult(CUstream, unsigned long long *);

#if CUDA_VERSION >= 12000

namespace {

cuStreamGetId_t *_cuStreamGetId = cuStreamGetId;

bool _hasPreciseHint()
{
    return true;
}

} // namespace

#else

#    include <dlfcn.h>
#    include <sys/syscall.h>
#    include <unistd.h>

namespace {

inline int getTID()
{
    return syscall(SYS_gettid);
}

constexpr uint64_t MakeLegacyStreamId(int dev, int tid)
{
    return (uint64_t)dev << 32 | tid;
}

CUresult cuStreamGetIdFallback(CUstream stream, unsigned long long *id)
{
    // If the stream handle is a pseudohandle, use some special treatment....
    if (stream == 0 || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)
    {
        int dev = -1;
        if (cudaGetDevice(&dev) != cudaSuccess)
            return CUDA_ERROR_INVALID_CONTEXT;
        // If we use a per-thread stream, get TID; otherwise use -1 as a pseudo-tid
        *id = MakeLegacyStreamId(dev, stream == CU_STREAM_PER_THREAD ? getTID() : -1);
        return CUDA_SUCCESS;
    }
    else
    {
        // Otherwise just use the handle - it's not perfactly safe, but should do.
        *id = (uint64_t)stream;
        return CUDA_SUCCESS;
    }
}

cuStreamGetId_t *getRealStreamIdFunc()
{
    static cuStreamGetId_t *fn = []()
    {
        void *sym = nullptr;
        // If it fails, we'll just return nullptr.
        (void)cuGetProcAddress("cuStreamGetId", &sym, 12000, CU_GET_PROC_ADDRESS_DEFAULT);
        return (cuStreamGetId_t *)sym;
    }();
    return fn;
}

bool _hasPreciseHint()
{
    static bool ret = getRealStreamIdFunc() != nullptr;
    return ret;
}

CUresult cuStreamGetIdBootstrap(CUstream stream, unsigned long long *id);

cuStreamGetId_t *_cuStreamGetId = cuStreamGetIdBootstrap;

CUresult cuStreamGetIdBootstrap(CUstream stream, unsigned long long *id)
{
    cuStreamGetId_t *realFunc = getRealStreamIdFunc();
    if (realFunc)
        _cuStreamGetId = realFunc;
    else
        _cuStreamGetId = cuStreamGetIdFallback;

    return _cuStreamGetId(stream, id);
}

} // namespace

#endif

namespace nvcv::util {

bool IsCudaStreamIdHintUnambiguous()
{
    return _hasPreciseHint();
}

uint64_t GetCudaStreamIdHint(CUstream stream)
{
    static auto initResult = cuInit(0);
    (void)initResult;
    unsigned long long id;
    CUresult           err = _cuStreamGetId(stream, &id);
    if (err != CUDA_SUCCESS)
    {
        switch (err)
        {
        case CUDA_ERROR_DEINITIALIZED:
            // This is most likely to happen during process teardown, so likely in a destructor
            // - we don't want to throw there and the stream equality is immaterial anyway at this point.
            return -1;
        case CUDA_ERROR_INVALID_VALUE:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid stream handle");
        default:
        {
            const char *msg  = "";
            const char *name = "Unknown error";
            (void)cuGetErrorString(err, &msg);
            (void)cuGetErrorName(err, &name);
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CUDA error %s %i %s", name, err, msg);
        }
        }
    }
    return id;
}

} // namespace nvcv::util
