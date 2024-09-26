#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This script searches for files that contain a specific SPDX license identifier in their comments, including
# multi-line comments, and deletes them.  It prepares the main branch to become a release branch without files
# and folders that should not be released. It also cleans up any references to the removed files and folders.
# To run:
# $ tools/prepare_release_branch.sh
# This script is intended to be used as in two passes:
#
# (1) $ tools/prepare_release_branch.sh
# (2) $ DEL_CI=true tools/prepare_release_branch.sh
#
# The process is as follows:
#  * Create a release branch.
#  * Run the script with step (1) to remove proprietary files and reference--except for the contents in the
#    ci/ and tools/ folders (to avoid removing this script).
#  * Push the cleaned branch and test if the CI passes without proprietary files.
#  * Run the script again prepending DEL_CI=true for step (2) to clear ci/ and tools/ folders (including this
#    script since it has a proprietary licence).
#
# Prepending "DRY_RUN=true " will do a dry run and thus avoid actual deletions while testing:
# $ DRY_RUN=true tools/prepare_release_branch.sh.
#
# To build a proprietary branch, set KEEP_PROP=true to keep the proprietary files that would otherwise be removed:
# $ KEEP_PROP=true tools/prepare_release_branch.sh

DRY_RUN="${DRY_RUN:-false}"
DEL_CI="${DEL_CI:-false}"
KEEP_PROP="${KEEP_PROP:-false}"

DRY_RUN_STR="I"
EXCLUDE_DIRS=(.git)
DEL_FOLDERS=(ci/jenkins lint)
DEL_FILES=(ci/*.yml ci/check_cicd.sh ci/download_assets.sh ci/gitlab_utils.sh ci/upload_assets.sh ci/install_deps*.sh .pre-commit-config.yaml)

CLEAN_BAK_DIR="/tmp/clean"

if $DRY_RUN; then
    DRY_RUN_STR="W (DRY RUN)"
fi

if ! $DEL_CI; then
    EXCLUDE_DIRS=(.git ci tools)
    DEL_FOLDERS=()
    DEL_FILES=()
fi

# Define the license string to search for
LICENSE_STRING="SPDX-License-Identifier: LicenseRef-NvidiaProprietary"

# C/C++/CUDA/Groovy scripts and headers use /* */ comments
SLASH_PATTERN="/\*([^*]|\*+[^*/])*\*+([^*]|\*+[^*/])*${LICENSE_STRING}([^*]|\*+[^*/])*\*+/"

# Python uses """ """ or ''' ''' comments
QUOTE_PATTERN="(\"\"\"|''')[^\"']*${LICENSE_STRING}[^\"']*(\"\"\"|''')"

# Python, shell and YAML use # comments
HASH_PATTERN="#.*${LICENSE_STRING}"

# Markdown uses [//]: # "Comment"
BRACKETS_PATTERN="\[//\]: # \".*${LICENSE_STRING}\""

# Tags to search for to remove references to proprietary files in shell
# (Separate full tag into pieces so script doesn't remove these lines)
REMOVE_TAG="NVIDIA PROPRIETARY"
REMOVE_BEG="### BEGIN ${REMOVE_TAG} ###"
REMOVE_END="### END ${REMOVE_TAG} ###"

# Function to delete files containing the license string
delete_files()
{
    local pattern=$1
    shift
    local extensions=("$@")

    local exclude_params=""

    for dir in "${EXCLUDE_DIRS[@]}"; do
        exclude_params=${exclude_params}"--exclude-dir=${dir} "
    done

    for extension in "${extensions[@]}"; do
        # List files with the specified extension and comment style
        for file in $(grep -lrzP ${exclude_params} --include="*.${extension}" "${pattern}"); do
            # For compilable files, we need to remove some contents referring to these files
            file_path=${file%/*}
            file_name=${file##*/}
            file_ext=${file_name##*.}
            file_noext=${file_name%.*}
            if [[ $file_ext == "c" || $file_ext == "cpp" || $file_ext == "cc" || $file_ext == "cu" ]]; then
                if ! $DRY_RUN; then
                    sed -i -e "/$file_name/d" $file_path/CMakeLists.txt
                fi
                echo "$DRY_RUN_STR Removed content $file_name from: $file_path/CMakeLists.txt"
                if [[ $file_path == "python/mod_cvcuda" ]]; then
                    if ! $DRY_RUN; then
                        sed -i -e "/$file_noext/d" python/mod_cvcuda/Operators.hpp python/mod_cvcuda/Main.cpp
                    fi
                    echo "$DRY_RUN_STR Removed content $file_noext from Python-based files:"
                    echo "$DRY_RUN_STR - python/mod_cvcuda/Operators.hpp"
                    echo "$DRY_RUN_STR - python/mod_cvcuda/Main.cpp"
                fi
            fi

            if ! $DRY_RUN; then
                rm -f $file
            fi
            echo "$DRY_RUN_STR Deleted file: $file"
        done
    done
}

clean_files()
{
    local marker_beg=$1
    shift
    local marker_end=$1
    shift
    local extensions=("$@")

    local exclude_params=""

    for dir in "${EXCLUDE_DIRS[@]}"; do
        exclude_params=${exclude_params}"--exclude-dir=${dir} "
    done

    rm -rf "$CLEAN_BAK_DIR"
    mkdir "$CLEAN_BAK_DIR"

    for extension in "${extensions[@]}"; do
        # List files with the specified extension and comment style
        for file in $(grep -lrz ${exclude_params} --include="*.${extension}" "${marker_beg}"); do
            file_path=${file%/*}
            file_name=${file##*/}
            sed -i.bak -e "/${marker_beg}/,/${marker_end}/d" $file
            mv "${file}.bak" "${CLEAN_BAK_DIR}/$file_name"
            echo "$DRY_RUN_STR Cleaned the following lines from $file:"
            echo "--------------------"
            # Temporarily disable exit on error since diff exits with error code 1 when files are different
            set +e
            diff "$file" "${CLEAN_BAK_DIR}/$file_name"
            set -e
            echo "--------------------"
            if ! $DRY_RUN; then
                rm -f "${CLEAN_BAK_DIR}/$file_name"
            else
                mv -f "${CLEAN_BAK_DIR}/$file_name" "$file_path"
            fi
        done
    done
    rm -rf "$CLEAN_BAK_DIR"
}

if [[ ${KEEP_PROP} == false ]]; then
    # Remove lines in scripts referring to proprietary files
    echo
    echo "******** Cleaning proprietary references from files ********"
    clean_files "$REMOVE_BEG" "$REMOVE_END" "sh" "py"

    # Delete proprietary files and their references
    echo
    echo "******** Removing proprietary files and references ********"
    delete_files "$SLASH_PATTERN" "c" "h" "cpp" "hpp" "cc" "cxx" "cu" "cuh" "groovy"
    delete_files "$QUOTE_PATTERN" "py"
    delete_files "$HASH_PATTERN" "py" "sh" "yaml" "rst"
    delete_files "$BRACKETS_PATTERN" "md"

    if ! ${DRY_RUN}; then
        sed -i -e "/list(APPEND CPPSAMPLES/d" samples/CMakeLists.txt
        sed -i -e "/list(APPEND PYSAMPLES/d" samples/CMakeLists.txt
    fi
    echo "$DRY_RUN_STR Removed content list append from: samples/CMakeLists.txt"
else
    echo "******** Keeping proprietary files and references ********"
fi

if [[ $DEL_CI == true ]]; then
    echo
    echo "******** Removing CI files and folders ********"
    for file in "${DEL_FILES[@]}"; do
        if ! $DRY_RUN; then
            rm -f $file
        fi
        echo "$DRY_RUN_STR Deleted file: $file"
    done

    for folder in "${DEL_FOLDERS[@]}"; do
        if ! $DRY_RUN; then
            rm -rf $folder
        fi
        echo "$DRY_RUN_STR Deleted folder: $folder"
    done
fi

# Remove reference to gitlab-master from docker config:
if ! $DRY_RUN; then
    sed -i -e "s/IMAGE_URL_BASE='gitlab-master.nvidia.com.*'/IMAGE_URL_BASE=''/g" docker/config
fi
echo "$DRY_RUN_STR Removed gitlab content in docker/config"
