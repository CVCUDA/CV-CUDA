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

# We must run the following at "include" time, not at function call time,
# to find the path to this module rather than the path to a calling list file
get_filename_component(config_version_script_path ${CMAKE_CURRENT_LIST_FILE} PATH)

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC REPO_COMMIT)

set(PROJECT_VERSION "${PROJECT_VERSION}${PROJECT_VERSION_SUFFIX}")

function(configure_version target LIBPREFIX incpath VERSION_FULL)
    string(TOUPPER "${target}" TARGET)

    string(REGEX MATCH "-(.*)$" version_suffix "${VERSION_FULL}")
    set(VERSION_SUFFIX ${CMAKE_MATCH_1})

    string(REGEX MATCHALL "[0-9]+" version_list "${VERSION_FULL}")
    list(GET version_list 0 VERSION_MAJOR)
    list(GET version_list 1 VERSION_MINOR)
    list(GET version_list 2 VERSION_PATCH)

    list(LENGTH version_list num_version_components)

    if(num_version_components EQUAL 3)
        set(VERSION_TWEAK 0)
    elseif(num_version_components EQUAL 4)
        list(GET version_list 3 VERSION_TWEAK)
    else()
        message(FATAL_ERROR "Version must have either 3 or 4 components")
    endif()

    math(EXPR VERSION_API_CODE "${VERSION_MAJOR}*100 + ${VERSION_MINOR}")

    string(REPLACE "-" "_" tmp ${VERSION_FULL})
    set(VERSION_BUILD "${tmp}-${CVCUDA_BUILD_SUFFIX}")

    configure_file(${config_version_script_path}/VersionDef.h.in include/${incpath}/VersionDef.h @ONLY ESCAPE_QUOTES)
    configure_file(${config_version_script_path}/VersionUtils.h.in include/${incpath}/detail/VersionUtils.h @ONLY ESCAPE_QUOTES)

    set(${LIBPREFIX}_VERSION_FULL ${VERSION_FULL} CACHE INTERNAL "${TARGET} full version")
    set(${LIBPREFIX}_VERSION_MAJOR ${VERSION_MAJOR} CACHE INTERNAL "${TARGET} major version")
    set(${LIBPREFIX}_VERSION_MINOR ${VERSION_MINOR} CACHE INTERNAL "${TARGET} minor version")
    set(${LIBPREFIX}_VERSION_PATCH ${VERSION_PATCH} CACHE INTERNAL "${TARGET} patch version")
    set(${LIBPREFIX}_VERSION_TWEAK ${VERSION_TWEAK} CACHE INTERNAL "${TARGET} tweak version")
    set(${LIBPREFIX}_VERSION_SUFFIX ${VERSION_SUFFIX} CACHE INTERNAL "${TARGET} version suffix")
    set(${LIBPREFIX}_VERSION_API ${VERSION_MAJOR}.${VERSION_MINOR} CACHE INTERNAL "${TARGET} API version")
    set(${LIBPREFIX}_VERSION_API_CODE ${VERSION_API_CODE} CACHE INTERNAL "${TARGET} API code")
    set(${LIBPREFIX}_VERSION_BUILD ${VERSION_BUILD} CACHE INTERNAL "${TARGET} build version")

    # So that the generated headers are found
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/VersionDef.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${incpath}
            COMPONENT dev)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/detail/VersionUtils.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${incpath}/detail
            COMPONENT dev)
endfunction()

function(configure_symbol_versioning dso_target VERPREFIX input_targets)
    # Create exports file for symbol versioning ---------------------------------
    set(EXPORTS_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/exports.ldscript")
    target_link_libraries(${dso_target}
        PRIVATE
        -Wl,--version-script ${EXPORTS_OUTPUT}
    )
    set(ALL_SOURCES "")
    foreach(tgt ${input_targets})
        get_target_property(tgt_sources ${tgt} SOURCES)
        get_target_property(tgt_srcdir ${tgt} SOURCE_DIR)

        foreach(src ${tgt_sources})
            if(${src} MATCHES "^/") # absolute paths?
                list(APPEND ALL_SOURCES ${src})
            else()
                list(APPEND ALL_SOURCES ${tgt_srcdir}/${src})
            endif()
        endforeach()
    endforeach()

    set(GEN_EXPORTS_SCRIPT "${config_version_script_path}/CreateExportsFile.cmake")

    add_custom_command(OUTPUT ${EXPORTS_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -DSOURCES="${ALL_SOURCES}"
                                 -DVERPREFIX=${VERPREFIX}
                                 -DOUTPUT=${EXPORTS_OUTPUT}
                                 -P "${GEN_EXPORTS_SCRIPT}"
        DEPENDS ${GEN_EXPORTS_SCRIPT} ${ALL_SOURCES})

    add_custom_target(create_${dso_target}_exports_file DEPENDS ${EXPORTS_OUTPUT})
    add_dependencies(${dso_target} create_${dso_target}_exports_file)
endfunction()
