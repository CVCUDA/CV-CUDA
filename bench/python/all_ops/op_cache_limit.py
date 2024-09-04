# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvcv
import torch

# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401
from bench_utils import AbstractOpBase

# For the following setup depicted in the table, we have to repeatedly call the functions: cudaMalloc and/or
# cudaFree.
#
# ---------------------------------------------------------------------
# | shape\cache limit  |   small                         large        |
# |-------------------------------------------------------------------|
# | non-random         |   cudaMalloc + cudaFree       - (best-case)  |
# | random             |   cudaMalloc + cudaFree       cudaMalloc     |
# ---------------------------------------------------------------------
#
# Due to the this table, we benchmark three scenarios: {non-random, small}, {non-random, large},
# {random, large}


# Base class for cache limit benchmarks, to ensure all three classes have the same overhead, leading to
# consistent numbers.
class BaseOpCacheLimit(AbstractOpBase):
    def setup(self, input, new_cache_limit, low, high):
        super().setup(input)

        # make this benchmark compatible with older cvcuda/nvncv versions
        if hasattr(nvcv, "set_cache_limit_inbytes"):
            nvcv.set_cache_limit_inbytes(new_cache_limit)

        # We don't have access to the outer benchmark iterations (default=10), so we have to create our own
        # counter.
        self.max_iter_outer = 10
        self.iter_outer = 0

        # Number of "random" tensors created per benchmarked run
        self.n_tensors = 20
        self.hw = torch.randint(
            low=low, high=high, size=(self.max_iter_outer, 2, self.n_tensors)
        )

    def run(self, input):
        # If we exceed the outer bench iterations, we return.
        # If we didn't return, we might re-use the cache, which we specifically don't want for
        # "OpCacheLimitLargeAndRandom".
        # For the other classes (OpCacheLimitZero, OpCacheLimitLarge), we could continue running the
        # benchmarks, but then we would not get comparable numbers between all three classes
        if self.iter_outer >= self.max_iter_outer:
            return

        for ii in range(self.n_tensors):
            shape = (
                self.hw[self.iter_outer, 0, ii].item(),
                self.hw[self.iter_outer, 1, ii].item(),
                3,
            )
            _ = nvcv.Tensor(shape, nvcv.Type.F32, nvcv.TensorLayout.HWC)

        self.iter_outer += 1
        return


# This is the {non-random, small} case. The smallest we can choose is 0, so we set the cache limit to 0 and
# effectively disable the cache
class OpCacheLimitZero(BaseOpCacheLimit):
    def setup(self, input):
        # Set the cache limit to 0 for this benchmark
        # low=1000, high=1001 results in always creating tensor's of shape (1000,1000,3)
        super().setup(input, 0, low=1000, high=1001)

    def run(self, input):
        super().run(input)


# This is the {non-random, large} case. This is the best case scenario, always re-using the cache
class OpCacheLimitLarge(BaseOpCacheLimit):
    def setup(self, input):
        # Set the cache limit to the total gpu memory for this benchmark
        # low=1000, high=1001 results in always creating tensor's of shape (1000,1000,3)
        total = torch.cuda.mem_get_info()[1]
        super().setup(input, total, low=1000, high=1001)

    def run(self, input):
        super().run(input)


# This is the {random, large} case. This is the worst case scenario, never re-using the cache
class OpCacheLimitLargeAndRandom(BaseOpCacheLimit):
    def setup(self, input):
        # Set the cache limit to the total gpu memory for this benchmark
        # low=1000, high=2000 results in always creating tensor's of random shape
        # between [(1000,1000,3), (1999,1999,3)]
        total = torch.cuda.mem_get_info()[1]
        super().setup(input, total, low=1000, high=2000)

    def run(self, input):
        super().run(input)
