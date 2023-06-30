/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "GaussianNoiseUtils.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK 512

__global__ void setup_states(curandState *state, unsigned long long seed, int batch)
{
    auto id = threadIdx.x + batch * blockDim.x;
    curand_init(seed, id, 0, &state[threadIdx.x]);
}

__global__ void rand_kernel(curandState *state, float *rand, int size, int per_channel)
{
    int         offset     = threadIdx.x;
    curandState localState = state[threadIdx.x];
    while (offset < size)
    {
        if (per_channel)
        {
            for (int i = 0; i < 3; i++) rand[offset * 3 + i] = curand_normal(&localState);
        }
        else
            rand[offset] = curand_normal(&localState);
        offset += blockDim.x;
    }
}

void get_random(float *rand_h, bool per_channel, int batch, int mem_size)
{
    curandState *states;
    cudaMalloc((void **)&states, sizeof(curandState) * BLOCK);
    setup_states<<<1, BLOCK>>>(states, 12345, batch);

    float *rand_d;
    cudaMalloc((void **)&rand_d, sizeof(float) * mem_size);
    int img_size = mem_size;
    if (per_channel)
        img_size /= 3;
    rand_kernel<<<1, BLOCK>>>(states, rand_d, img_size, per_channel);
    cudaMemcpy(rand_h, rand_d, mem_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(states);
    cudaFree(rand_d);
}
