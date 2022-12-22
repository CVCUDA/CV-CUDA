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

list(APPEND CPACK_COMPONENTS_ALL samples)

set(CPACK_COMPONENT_SAMPLES_DISABLED true)
set(CPACK_COMPONENT_SAMPLES_DISPLAY_NAME "Samples")
set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "NVIDIA CV-CUDA Samples")

if(UNIX)
    set(CVCUDA_SAMPLES_FILE_NAME "cvcuda-samples-${CVCUDA_VERSION_BUILD}")
    set(CPACK_DEBIAN_SAMPLES_FILE_NAME "${CVCUDA_SAMPLES_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_SAMPLES_FILE_NAME "${CVCUDA_SAMPLES_FILE_NAME}")

    set(CPACK_DEBIAN_SAMPLES_PACKAGE_NAME "cvcuda${PROJECT_VERSION_MAJOR}-samples")

    set(CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS "${CPACK_DEBIAN_DEV_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")
else()
    set(CPACK_COMPONENT_SAMPLES_DEPENDS dev)
endif()
