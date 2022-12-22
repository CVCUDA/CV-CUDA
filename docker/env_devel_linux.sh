#!/bin/bash -e

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

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

# shellcheck source=docker/config
. "$SDIR/config"

extra_args=""
if [ -d $HOME/.vim ]; then
    extra_args="$extra_args -v $HOME/.vim:$HOME/.vim"
elif [ -f $HOME/.vimrc ]; then
    extra_args="$extra_args -v $HOME/.vimrc:$HOME/.vimrc"
fi

if [ -f $HOME/.gdbinit ]; then
    extra_args="$extra_args -v $HOME/.gdbinit:$HOME/.gdbinit"
fi

if [ -f /etc/sudoers ]; then
    extra_args="$extra_args -v /etc/sudoers:/etc/sudoers"
fi
if [ -d /etc/sudoers.d ]; then
    extra_args="$extra_args -v /etc/sudoers.d:/etc/sudoers.d"
fi

extra_cmds="true"

# Set up git user inside docker
git_user_name=$(git config --global user.name || true)
git_user_email=$(git config --global user.email || true)
if [[ "$git_user_name" && "$git_user_email" ]]; then
    extra_cmds="$extra_cmds && git config --global user.name '$git_user_name'"
    extra_cmds="$extra_cmds && git config --global user.email '$git_user_email'"
    echo "Setting up cv-cuda dev environment for: $git_user_name <$git_user_email>"
else
    echo "Git user.name and user.email not set up"
    echo "Please run:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'your_nvlogin@nvidia.com'"
    exit 1
fi

# Run docker
# Notes:
#   - first and second cache mappings are for ccache and pre-commit respectively.
#   - pre-commit needs $HOME/.npm
docker run --gpus=all --pull always -ti \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v $HOME/.cache:/cache \
    -v $HOME/.cache:$HOME/.cache \
    -v $HOME/.npm:$HOME/.npm \
    -v /var/tmp:/var/tmp \
    -v $SDIR/..:$HOME/cvcuda \
    $extra_args \
    $IMAGE_URL_BASE/devel-linux:$TAG_IMAGE \
    /usr/bin/bash -c "mkdir -p $HOME && chown $USER:$USER $HOME && su - $USER -c \"$extra_cmds\" && su - $USER"
