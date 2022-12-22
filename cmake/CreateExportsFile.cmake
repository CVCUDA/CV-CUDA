# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(NOT OUTPUT)
    message(FATAL_ERROR "No output exports file specified")
endif()

if(NOT SOURCES)
    message(FATAL_ERROR "No source files specified")
endif()

if(NOT VERPREFIX)
    message(FATAL_ERROR "No version prefix specified")
endif()

string(REPLACE " " ";" SOURCES ${SOURCES})

# Create an empty file
file(WRITE ${OUTPUT} "")

set(all_versions "")

foreach(src ${SOURCES})
    file(STRINGS ${src} funcdef_list REGEX "_DEFINE_API.*")

    foreach(func_def ${funcdef_list})
        if(func_def MATCHES "^[A-Z_]+_DEFINE_API\\(+([^,]+),([^,]+),[^,]+,([^,]+).*$")
            string(STRIP "${CMAKE_MATCH_1}" ver_major)
            string(STRIP "${CMAKE_MATCH_2}" ver_minor)
            string(STRIP "${CMAKE_MATCH_3}" func)
            list(APPEND all_versions ${ver_major}.${ver_minor})
            list(APPEND funcs_${ver_major}_${ver_minor} ${func})
        else()
            message(FATAL_ERROR "I don't understand ${func_def}")
        endif()
    endforeach()
endforeach()

list(SORT all_versions COMPARE NATURAL)
list(REMOVE_DUPLICATES all_versions)

if(all_versions)
    set(prev_version "")
    foreach(ver ${all_versions})
        if(ver MATCHES "([0-9]+)\\.([0-9]+)")
            set(ver_major ${CMAKE_MATCH_1})
            set(ver_minor ${CMAKE_MATCH_2})

            file(APPEND ${OUTPUT} "${VERPREFIX}_${ver} {\nglobal:\n")

            if(NOT funcs_${ver_major}_${ver_minor})
                message(FATAL_ERROR "funcs_${ver_major}_${ver_minor} must not be empty")
            endif()

            list(SORT funcs_${ver_major}_${ver_minor})

            foreach(func ${funcs_${ver_major}_${ver_minor}})
                file(APPEND ${OUTPUT} "    ${func};\n")
            endforeach()

            if(prev_version)
                file(APPEND ${OUTPUT} "} ${VERPREFIX}_${prev_version};\n\n")
            else()
                file(APPEND ${OUTPUT} "local: *;\n};\n\n")
            endif()

            set(prev_version ${ver})
        else()
            message(FATAL_ERROR "I don't version ${ver}")
        endif()
    endforeach()
else()
    file(APPEND ${OUTPUT} "${VERPREFIX} {\nlocal: *;\n};\n")
endif()
