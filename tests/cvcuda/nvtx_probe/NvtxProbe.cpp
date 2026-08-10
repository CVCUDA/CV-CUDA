/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Test-only NVTX injection library. NVTX exposes no API to read ranges back, so to observe the
// ranges CV-CUDA emits this library is loaded via NVTX_INJECTION64_PATH: NVTX dlopen()s it on its
// first call and invokes InitializeInjectionNvtx2, which swaps in the recorders below. The test
// process reads the recorded names through the CvcudaNvtxProbe_* accessors.

#include <nvtx3/nvToolsExt.h>

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Cap on recorded names. NVTX reads the injection path once at process start, so the probe stays
// installed for the lifetime of whatever process loads it. A long-lived host (e.g. a full pytest
// session driving every operator, plus cupy's own NVTX ranges) would otherwise grow pushedNames
// without bound and can exhaust host memory on tighter runners. The marker test resets before each
// operator and reads immediately, so it never approaches this cap; recording simply stops here.
constexpr std::size_t kMaxRecordedNames = 4096;

struct ProbeState
{
    std::mutex               mutex;
    std::vector<std::string> pushedNames;
};

// A function-local static holds the recorder state: it gives the C callbacks and accessors a
// single shared, thread-safe-initialized record while avoiding non-const namespace-scope globals.
ProbeState &state()
{
    static ProbeState s;
    return s;
}

// Replacement for nvtxRangePushA: record the range name. The return value is the (1-based) range
// nesting level; NVTX callers don't depend on the exact value here.
int NVTX_API ProbeRangePushA(const char *message)
{
    std::scoped_lock lock(state().mutex);
    if (state().pushedNames.size() < kMaxRecordedNames)
    {
        state().pushedNames.emplace_back(message ? message : "");
    }
    return static_cast<int>(state().pushedNames.size());
}

// Replacement for nvtxRangePop: nothing to record, pushes alone identify the ranges we assert on.
int NVTX_API ProbeRangePop(void)
{
    return 0;
}

} // namespace

// Forward declarations keep these exported entry points free of -Wmissing-declarations
// (CI builds with -DWARNINGS_AS_ERRORS=1) while retaining external linkage.
extern "C" int          InitializeInjectionNvtx2(NvtxGetExportTableFunc_t getExportTable);
extern "C" void         CvcudaNvtxProbe_Reset(void);
extern "C" unsigned int CvcudaNvtxProbe_Count(void);
extern "C" const char  *CvcudaNvtxProbe_Name(unsigned int index);

// NVTX injection entry point. Returning non-zero tells NVTX the injection succeeded and that the
// function table it handed us should be used.
extern "C" int InitializeInjectionNvtx2(NvtxGetExportTableFunc_t getExportTable)
{
    if (getExportTable == nullptr)
    {
        return 0;
    }

    const auto *callbacks = static_cast<const NvtxExportTableCallbacks *>(getExportTable(NVTX_ETID_CALLBACKS));
    if (callbacks == nullptr || callbacks->struct_size < sizeof(NvtxExportTableCallbacks)
        || callbacks->GetModuleFunctionTable == nullptr)
    {
        return 0;
    }

    // table is an array of pointers to the core module's function-pointer slots; size is the
    // highest valid index. Each entry is the *address* of a slot, so we install a replacement by
    // writing through it.
    NvtxFunctionTable table = nullptr;
    unsigned int      size  = 0;
    if (callbacks->GetModuleFunctionTable(NVTX_CB_MODULE_CORE, &table, &size) == 0 || table == nullptr)
    {
        return 0;
    }

    if (size < static_cast<unsigned int>(NVTX_CBID_CORE_RangePop))
    {
        return 0;
    }

    if (table[NVTX_CBID_CORE_RangePushA] != nullptr)
    {
        *table[NVTX_CBID_CORE_RangePushA] = reinterpret_cast<NvtxFunctionPointer>(&ProbeRangePushA);
    }
    if (table[NVTX_CBID_CORE_RangePop] != nullptr)
    {
        *table[NVTX_CBID_CORE_RangePop] = reinterpret_cast<NvtxFunctionPointer>(&ProbeRangePop);
    }

    return 1;
}

// --- Accessors read by the test process (same process; this library is dlopen()ed into it). ---

extern "C" void CvcudaNvtxProbe_Reset(void)
{
    std::scoped_lock lock(state().mutex);
    state().pushedNames.clear();
}

extern "C" unsigned int CvcudaNvtxProbe_Count(void)
{
    std::scoped_lock lock(state().mutex);
    return static_cast<unsigned int>(state().pushedNames.size());
}

// Returns the i-th recorded range name, or "" if out of range. The pointer is valid until the next
// CvcudaNvtxProbe_Reset; the test reads names after the operator call completes (no concurrent
// pushes), so the backing storage is stable at read time.
extern "C" const char *CvcudaNvtxProbe_Name(unsigned int index)
{
    std::scoped_lock lock(state().mutex);
    if (index >= state().pushedNames.size())
    {
        return "";
    }
    return state().pushedNames[index].c_str();
}
