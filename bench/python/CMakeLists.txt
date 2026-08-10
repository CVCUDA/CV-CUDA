# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(cvcuda_bench_python)

# Copy Python benchmark scripts to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python_bench_utils.py
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_bench_utils.py
               COPYONLY)

# List of all Python benchmark scripts
set(python_bench_scripts
    ops/bench_jpegcompressiondistortion.py
    ops/bench_adjusthue.py
    ops/bench_adjustsaturation.py
    ops/bench_adjustsharpness.py
    ops/bench_adjustcontrast.py
    ops/bench_invert.py
    ops/bench_solarize.py
    ops/bench_posterize.py
    ops/bench_autocontrast.py
    ops/bench_adaptivethreshold.py
    ops/bench_advcvtcolor.py
    ops/bench_averageblur.py
    ops/bench_bilateralfilter.py
    ops/bench_bndbox.py
    ops/bench_boxblur.py
    ops/bench_brightnesscontrast.py
    ops/bench_centercrop.py
    ops/bench_channelreorder.py
    ops/bench_clahe.py
    ops/bench_colortwist.py
    ops/bench_composite.py
    ops/bench_conv2d.py
    ops/bench_convertto.py
    ops/bench_copymakeborder.py
    ops/bench_cropflipnormalizereformat.py
    ops/bench_customcrop.py
    ops/bench_cvtcolor.py
    ops/bench_erase.py
    ops/bench_findhomography.py
    ops/bench_flip.py
    ops/bench_gammacontrast.py
    ops/bench_gaussian.py
    ops/bench_gaussiannoise.py
    ops/bench_histogram.py
    ops/bench_histogrameq.py
    ops/bench_hqresize.py
    ops/bench_inpaint.py
    ops/bench_jointbilateralfilter.py
    ops/bench_label.py
    ops/bench_laplacian.py
    ops/bench_medianblur.py
    ops/bench_minarearect.py
    ops/bench_minmaxloc.py
    ops/bench_morphology.py
    ops/bench_nonmaximumsuppression.py
    ops/bench_normalize.py
    ops/bench_osd.py
    ops/bench_padandstack.py
    ops/bench_pairwisematcher.py
    ops/bench_pillowresize.py
    ops/bench_randomresizedcrop.py
    ops/bench_reformat.py
    ops/bench_remap.py
    ops/bench_resize.py
    ops/bench_resizecropconvertreformat.py
    ops/bench_rotate.py
    ops/bench_sift.py
    ops/bench_stack.py
    ops/bench_threshold.py
    ops/bench_warpaffine.py
    ops/bench_warpperspective.py
)

# Copy benchmark scripts to build directory (flattened - no ops subdirectory)
foreach(script IN LISTS python_bench_scripts)
    get_filename_component(script_name "${script}" NAME)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${script}
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${script_name}
                   COPYONLY)
endforeach()

# Install Python benchmark scripts
install(PROGRAMS
        ${CMAKE_CURRENT_SOURCE_DIR}/python_bench_utils.py
        ${CMAKE_CURRENT_SOURCE_DIR}/install_bench_dependencies.sh
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT bench)

# Install requirements files for dependency management
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.bench.common.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.bench.cu12.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.bench.cu13.txt
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT bench)

# Install benchmark scripts (flattened - no ops subdirectory)
foreach(script IN LISTS python_bench_scripts)
    install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/${script}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT bench)
endforeach()
