#!/bin/bash -eE

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

# Check if input files have valid copyright message
# Ref: https://confluence.nvidia.com/display/RP/Standardizing+on+SPDX+Identifiers

valid_license='Apache-2.0'

# Detects that the line is a comment.
rgx_comment='^[[:space:]]*[[:graph:]]\+[[:space:]]\+'

function get_tag()
{
    local tag=$1
    shift

    local rgx="s@^\($rgx_comment\)\?$tag:[[:space:]]*\(.*\)@\2@p"

    sed -n "$rgx" "$file"
}

function error()
{
    local file=$1
    local msg=$2
    shift 2

    echo -e "In $file:\n\t$msg" && false
}

function check_license()
{
    local file=$1
    shift

    # Get tag value
    local tag='SPDX-License-Identifier'
    local license
    license=$(get_tag "$tag")
    if [ -z "$license" ]; then
        error "$file" "No well-formed $tag tag found."
    fi

    # Check if it is valid
    if [[ "$license" != "$valid_license" ]]; then
        error "$file" "License '$license' not valid. Must be '$valid_license'." && false
    fi
}

function get_copyright_year_range()
{
    local file=$1
    shift

    local tag='SPDX-FileCopyrightText'
    copyright=$(get_tag "$tag")
    if [ "$copyright" ]; then
        local rgx_copyright_year_range='Copyright[[:space:]]*([Cc])[[:space:]]\+\([[:digit:]]\+-\?[[:digit:]]\+\),\?[[:space:]]*NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.'

        # If copyright text is limited to fit 80 characters,
        if [ "$copyright" = 'NVIDIA CORPORATION & AFFILIATES' ]; then
            # Look for the non-tagged copyright message
            copyright=$(sed -n 's@^\('"$rgx_comment"'\)\?\('"$rgx_copyright_year_range"'\)@\2@p' "$file")
        fi
    fi

    echo "$copyright" | sed -n "s@$rgx_copyright_year_range@\1@p"
}

function check_copyright_message()
{
    local file=$1
    shift

    # Get tag value
    local tag='SPDX-FileCopyrightText'
    local copyright
    copyright=$(get_tag "$tag")
    if [ -z "$copyright" ]; then
        error "$file" "No well-formed $tag tag found." && false
    fi

    # Check if year range is valid
    local year_range
    year_range=$(get_copyright_year_range "$file")
    if [[ -z "$year_range" ]]; then
        error "$file" "Malformed copyright message '$copyright'. Must be 'Copyright (c) beg_year[-end_year] NVIDIA CORPORATION & AFFILIATES. All rights reserved.'" && false
    fi
}

function check_copyright_year()
{
    local file=$1
    shift

    local year_range
    year_range=$(get_copyright_year_range "$file")

    # Get copyright year range
    local rgx_year_range='\([[:digit:]]\{4\}\)\(-[[:digit:]]\{4\}\)\?'
    local beg_year end_year
    beg_year=$(echo "$year_range" | sed -n "s@$rgx_year_range@\1@p")
    end_year=$(echo "$year_range" | sed -n "s@$rgx_year_range@\2@p")
    end_year=${end_year:1} # remove '-' at beginning
    if [[ -z "$beg_year" ]]; then
        error "$file" "Malformed copyright year range '$year_range'. Must be beg_year[-end_year]." && false
    fi

    # Check if range is valid

    # Get the year when file was last modified.
    local cur_year

    # If file is staged,
    local is_staged
    is_staged=$(git diff --name-only --cached "$file")
    if [ "$is_staged" ]; then
        # it was modified now
        cur_year=$(date +'%Y') # YYYY
    else
        # get last modification time
        cur_year=$(git log -1 --pretty="format:%cs" "$file") # YYYY-MM-DD
        cur_year=${cur_year%%-*} # YYYY
    fi

    # Only start year?
    if [ -z "$end_year" ]; then
        if [[ $beg_year != "$cur_year" ]]; then
            error "$file" "Invalid year '$beg_year' in copyright message. Must be '$cur_year'." && false
        fi
    # Range doesn't include current year?
    elif [[ $beg_year -ge $cur_year || $end_year -lt $cur_year ]]; then
        error "$file" "Invalid year range '$year_range' in copyright message. '$cur_year' must be in range ($beg_year;$end_year]." && false
    fi
}

for file in "$@"; do
    check_license "$file"
    check_copyright_message "$file"
    check_copyright_year "$file"
done
