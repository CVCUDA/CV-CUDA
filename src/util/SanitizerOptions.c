/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Compiler.hpp"

#include <stdio.h>  // fprintf
#include <stdlib.h> // abort
#include <string.h>

#if NVCV_SANITIZED

// Here we define a lot of functions that will be used by ASAN,
// we don't need to declare them anywhere. This here will shut
// up gcc that complains we're not doing it.
#    pragma GCC diagnostic ignored "-Wmissing-declarations"

#    define ASAN_ATTRIBUTES                                                      \
        __attribute__((no_sanitize_address)) __attribute__((no_sanitize_thread)) \
        __attribute__((visibility("default"))) __attribute__((used))

// HACK needed by gcc-7.3/7.4 or else it'll complain that
// it can't inline fprintf in a function with attributes
// like the ones below.
static void printError(const char *str)
{
    fprintf(stderr, "%s", str);
}

// Used by our test suite to pass additional default options to asan
__attribute__((weak)) const char *nvcv_additional_asan_default_options();

// default runtime options for AddressSanitizer
ASAN_ATTRIBUTES const char *__asan_default_options()
{
    static char options[256] =
        // Needed so that CUDA initialization doesn't fail
        // ref: https://github.com/google/sanitizers/issues/629
        "protect_shadow_gap=0"
        // needed for displaying complete call stacks
        ":fast_unwind_on_malloc=0";

    if (nvcv_additional_asan_default_options)
    {
        const char *more = nvcv_additional_asan_default_options();

        int moresize = strlen(more);

        // +1 -> ':', +1 = '\0'
        if (strlen(options) + moresize + 1 + 1 >= sizeof(options))
        {
            printError("ASAN default options too long\n");
            abort();
        }

        int cur = strlen(options);

        options[cur++] = ':';

        // can't use strcpy or else asan will be triggered,
        // and it'll fail saying it's not initialized yet
        for (int i = 0; i < moresize + 1; ++i) // +1 -> copy \0 too
            options[cur++] = more[i];
    }

    return options;
}

// default options for AddressSanitizer
ASAN_ATTRIBUTES const char *__asan_default_suppressions()
{
    return "";
}

// default options for LeakSanitizer
ASAN_ATTRIBUTES const char *__lsan_default_options()
{
    return "";
}

__attribute__((weak)) const char *nvcv_additional_lsan_default_suppressions();

// default suppressions for LeakSanitizer
ASAN_ATTRIBUTES const char *__lsan_default_suppressions()
{
    static char supp[256] =
        // Known leak when enabling leak detector and undefined sanitizer:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80578
        //"leak:__cxa_demangle\n"
        // When CUDA's JIT for our cuda kernels kicks in, there are some leaks. It's
        // not a big deal as the number of cuda kernels are limited, and JIT is only
        // done in the first run.
        //"leak:nvPTXCompilerCompile\n"
        // seen with cuda-11-4-local_11.4.20210914-1
        //"leak:cudaGetDevice\n"
        "";

    if (nvcv_additional_lsan_default_suppressions)
    {
        const char *more = nvcv_additional_lsan_default_suppressions();

        int moresize = strlen(more);

        // +1 -> ':', +1 = '\0'
        if (strlen(supp) + moresize + 1 + 1 >= sizeof(supp))
        {
            printError("LSAN default suppressions too long\n");
            abort();
        }

        int cur = strlen(supp);

        supp[cur++] = '\n';

        // can't use strcpy or else asan will be triggered,
        // and it'll fail saying it's not initialized yet
        for (int i = 0; i < moresize + 1; ++i) // +1 -> copy \0 too
            supp[cur++] = more[i];
    }

    return supp;
}

// default options for UndefinedBehaviorSanitizer
ASAN_ATTRIBUTES const char *__ubsan_default_options()
{
    return "print_stacktrace=1"; // by default stack traces aren't printed
}

// default suppressions for UndefinedBehaviorSanitizer
ASAN_ATTRIBUTES const char *__ubsan_default_suppressions()
{
    return "";
}

#endif // VPI_ADDRESS_SANITIZER
