/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#if NVCV_UNIT_TESTS
#    include <private/core/Status.hpp>
#else
#    include <nvcv/Status.h>
#endif

namespace t = ::testing;

namespace {

class EventListener : public t::EmptyTestEventListener
{
public:
    virtual void OnTestStart(const t::TestInfo &tinfo) override
    {
        // Swallow any existing error so that test isn't affected by it.
        // Actual error must have been already trapped in the previous test.
#if NVCV_UNIT_TESTS
        nvcv::priv::SetThreadStatus(std::exception_ptr{});
#else
        nvcvGetLastError();
#endif

        g_HasSanitizerError = false;
    }

    virtual void OnTestEnd(const t::TestInfo &tinfo) override
    {
        int devCount = 0;
        cudaGetDeviceCount(&devCount);

        if (devCount)
        {
            EXPECT_EQ(cudaSuccess, cudaGetLastError()) << "Some test leaked a cuda error";

            // Make sure all activities on the GPU are stopped.
            EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
            cudaGetLastError(); // swallow the error
        }

        EXPECT_FALSE(g_HasSanitizerError);
    }

    static void OnSanitizerError()
    {
        g_HasSanitizerError = true;
    }

private:
    static bool g_HasSanitizerError;
};

bool EventListener::g_HasSanitizerError;

} // namespace

#if NVCV_SANITIZED
#    define ASAN_ATTRIBUTES                                                      \
        __attribute__((no_sanitize_address)) __attribute__((no_sanitize_thread)) \
        __attribute__((visibility("default"))) __attribute__((used))

// called when asan flags a bug
extern "C" ASAN_ATTRIBUTES void __asan_on_error()
{
    EventListener::OnSanitizerError();
}

extern "C" ASAN_ATTRIBUTES const char *vpi_additional_asan_default_options();

extern "C" ASAN_ATTRIBUTES const char *vpi_additional_asan_default_options()
{
    // Always halt on error since ASAN errors usually mean memory corruption
    // that could affect several tests. We're not interested in seeing the same
    // error over and over. This can be overriden by setting envvar
    // ASAN_OPTIONS=halt_on_error=0
    return "halt_on_error=1";
}

extern "C" ASAN_ATTRIBUTES const char *vpi_additional_lsan_default_suppressions();

extern "C" ASAN_ATTRIBUTES const char *vpi_additional_lsan_default_suppressions()
{
    // nothing so far
    return "";
}

#endif

int main(int argc, char **argv)
{
    t::InitGoogleTest(&argc, argv);

    t::UnitTest *unitTest = t::UnitTest::GetInstance();

    t::TestEventListeners &listeners = unitTest->listeners();

    listeners.Append(new EventListener);

    return RUN_ALL_TESTS();
}
