# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE on)

# nvcv_util_sanitizer ---------------------------------
add_library(nvcv_util_sanitizer
    SanitizerOptions.c
)

if(ENABLE_SANITIZER)
    # needed to force inclusion of ASAN default option functions in
    # modules that link to this library, see SanitizerOptions.c
    target_link_libraries(nvcv_util_sanitizer
        PUBLIC
        -Wl,-u__asan_default_options
        -Wl,-u__asan_default_suppressions
        -Wl,-u__lsan_default_options
        -Wl,-u__lsan_default_suppressions
        -Wl,-u__ubsan_default_options
        -Wl,-u__ubsan_default_suppressions)
endif()

# nvcv_util_compat ---------------------------------

if(ENABLE_COMPAT_OLD_GLIBC)
    set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS compat_symbols.txt)

    # Load up the symbols that must be replaced
    file(STRINGS compat_symbols.txt symbols)
    set(srccompat "")
    set(linkcompat "")
    foreach(sym ${symbols})
        # Skip lines that start with #
        if(NOT "${sym}" MATCHES "^#")
            string(REGEX REPLACE "([^@]+).*" "\\1" fnname ${sym})

            # tell linker to resolve "fnname" to a wrapper symbol, __wrap_fname
            list(APPEND linkcompat "-Wl,--wrap=${fnname}")

            # Define the wrapper that will jump into the old symbol
            set(srccompat " \
    ${srccompat}
    __attribute__((used,naked))
    void __wrap_${fnname}()
    {
        asm(\".symver __my_${fnname},${sym}\");
        asm(\"jmp __my_${fnname}@PLT\" ::);
    }
    ")
        endif()
    endforeach()

    macro(path_is_relative PATH IS_REL)
        string(FIND ${PATH} "/" idx)
        if(idx EQUAL "0")
            set(${IS_REL} OFF)
        else()
            set(${IS_REL} ON)
        endif()
    endmacro()

    # Same as configure_file, but create backups instead
    # of overwriten existing destination file.
    macro(safe_configure_file TEMPLATE DEST)
        set(dest ${DEST})
        path_is_relative(${dest} isrel)
        if(isrel)
            set(dest ${CMAKE_CURRENT_BINARY_DIR}/${dest})
        endif()

        configure_file(${TEMPLATE} ${dest}.new ${ARGN})

        if(EXISTS ${dest})
            file(READ ${dest} orig_data)
            file(READ ${dest}.new new_data)
            if(NOT orig_data STREQUAL new_data)
                set(idx 0)
                while(EXISTS ${dest}.bak${idx})
                    math(EXPR idx "${idx}+1")
                endwhile()

                file(COPY ${dest} DESTINATION ${dest}.bak${idx})
                file(RENAME ${dest}.new ${dest})
            else()
                file(REMOVE ${dest}.new)
            endif()
        else()
            file(RENAME ${dest}.new ${dest})
        endif()
    endmacro()

    # Replace the wrappers into the template source/header
    set(TEMPLATE "${srccompat}")
    safe_configure_file(Compat.c.in Compat.c @ONLY)

    add_library(nvcv_util_compat OBJECT
        Compat.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Compat.c
    )

    # Compat.c needs Compat.h
    get_source_file_property(COMPAT_INCDIRS ${CMAKE_CURRENT_BINARY_DIR}/Compat.c INCLUDE_DIRECTORIES)
    if(COMPAT_INCDIRS)
        list(APPEND COMPAT_INCDIRS ${CMAKE_CURRENT_SOURCE_DIR}/..)
    else()
        set(COMPAT_INCDIRS ${CMAKE_CURRENT_SOURCE_DIR}/..)
    endif()
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/Compat.c PROPERTIES INCLUDE_DIRECTORIES "${COMPAT_INCDIRS}")

    target_include_directories(nvcv_util_compat
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    # glibc-2.17 came split into different libraries.
    # We have to link against them instead of only against libc.so.
    # In later glibc versions, these libraries
    # are still present, but their empty. The symbols are all in libc.

    # We'll link to the corresponding stub libraries so that glibc<2.30's
    # dl-reloc can find the symbols in the correct sonames. Later glibc
    # can resolve symbols in different recorded sonames.

    # UPDATE: We're disabling this feature for further tests.
    # There's a change of infinite recursion due to gold linker
    # behavior when using symver + wrapping.
    # ref: https://marc.info/?l=binutils&m=1498883554126845&w=2
    # Moreover using the normal linker on Ubuntu 18.04 results in
    # linking errors.

    target_link_options(nvcv_util_compat
        PUBLIC
            -static-libstdc++
            -static-libgcc
            -Wl,--wrap=__libc_start_main
            -Wl,-u__cxa_thread_atexit_impl
            ${linkcompat}
            -Wl,--push-state,--no-as-needed
            ${CMAKE_CURRENT_SOURCE_DIR}/stubs/libdl-2.17_stub.so
            ${CMAKE_CURRENT_SOURCE_DIR}/stubs/librt-2.17_stub.so
            ${CMAKE_CURRENT_SOURCE_DIR}/stubs/libpthread-2.17_stub.so
            -Wl,--pop-state
    )
else()
    # dummy target
    add_library(nvcv_util_compat INTERFACE)
endif()

# nvcv_util_symver ---------------------------------
add_library(nvcv_util_symver INTERFACE)
if(LTO_ENABLED AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
    # symver attribute only works with gcc>=10.0. If using anything below, we
    # use the symver pragma, which doesn't work correcly when LTO is enabled.
    # See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=48200
    # Our solution is to disable LTO *only* in targets that depend on
    # nvcv_util_symver, which will be the ones that defines the public
    # functions, not the private implementation. This minimizes the perf hit
    # of disabling LTO to a small subset of the code.
    target_compile_options(nvcv_util_symver INTERFACE -fno-lto)
endif()
target_include_directories(nvcv_util_symver
    INTERFACE ..
)

# nvcv_util ---------------------------------
add_library(nvcv_util
    Assert.cpp
    CheckError.cpp
    String.cpp
    Version.cpp
    TensorDataUtils.cpp
)

target_include_directories(nvcv_util
    INTERFACE
        ..
    PUBLIC
        ../core/include
)

target_link_libraries(nvcv_util
    PUBLIC
        nvcv_util_sanitizer
        nvcv_types_headers
        CUDA::cudart_static
        nvcv_util_compat
        -lrt
)

target_compile_definitions(nvcv_util
    PUBLIC
        # Must always be enabled so that we can check assertion
        # failures in release mode.
        -DNVCV_DEBUG=1
)

if(EXPOSE_CODE)
    target_compile_definitions(nvcv_util
        PUBLIC
            -DNVCV_EXPOSE_CODE=1
    )
endif()
