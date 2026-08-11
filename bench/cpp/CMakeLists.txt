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

project(cvcuda_bench)

# Standalone-bench mode: when invoked with BUILD_LIB=OFF (top-level), the
# cvcuda + nvcv_types targets aren't defined in this tree. Pull them in via
# the CMake config files installed by cvcuda-dev. This lets a CI test pod
# compile bench against a pre-built cvcuda from a manylinux artifact, without
# having to rebuild the lib itself. When BUILD_LIB=ON, the in-tree targets
# already exist and these find_package calls are skipped.
if(NOT TARGET cvcuda)
    find_package(nvcv_types CONFIG REQUIRED)
    find_package(cvcuda CONFIG REQUIRED)
endif()

set(bench_sources
    ops/BenchJpegCompressionDistortion.cpp
    ops/BenchAdjustHue.cpp
    ops/BenchAdjustSaturation.cpp
    ops/BenchAdjustSharpness.cpp
    ops/BenchAdjustContrast.cpp
    ops/BenchInvert.cpp
    ops/BenchSolarize.cpp
    ops/BenchPosterize.cpp
    ops/BenchAutoContrast.cpp
    ops/BenchMinMaxLoc.cpp
    ops/BenchNonMaximumSuppression.cpp
    ops/BenchRemap.cpp
    ops/BenchGaussian.cpp
    ops/BenchLaplacian.cpp
    ops/BenchAverageBlur.cpp
    ops/BenchMedianBlur.cpp
    ops/BenchBilateralFilter.cpp
    ops/BenchSIFT.cpp
    ops/BenchReformat.cpp
    ops/BenchResize.cpp
    ops/BenchFlip.cpp
    ops/BenchRotate.cpp
    ops/BenchPillowResize.cpp
    ops/BenchHQResize.cpp
    ops/BenchCenterCrop.cpp
    ops/BenchWarpPerspective.cpp
    ops/BenchWarpAffine.cpp
    ops/BenchThreshold.cpp
    ops/BenchRandomResizedCrop.cpp
    ops/BenchNormalize.cpp
    ops/BenchMorphology.cpp
    ops/BenchAdaptiveThreshold.cpp
    ops/BenchCvtColor.cpp
    ops/BenchAdvCvtColor.cpp
    ops/BenchBndBox.cpp
    ops/BenchBoxBlur.cpp
    ops/BenchBrightnessContrast.cpp
    ops/BenchChannelReorder.cpp
    ops/BenchColorTwist.cpp
    ops/BenchComposite.cpp
    ops/BenchConv2D.cpp
    ops/BenchConvertTo.cpp
    ops/BenchCopyMakeBorder.cpp
    ops/BenchCropFlipNormalizeReformat.cpp
    ops/BenchResizeCropConvertReformat.cpp
    ops/BenchCustomCrop.cpp
    ops/BenchErase.cpp
    ops/BenchGammaContrast.cpp
    ops/BenchGaussianNoise.cpp
    ops/BenchHistogramEq.cpp
    ops/BenchHistogram.cpp
    ops/BenchInpaint.cpp
    ops/BenchJointBilateralFilter.cpp
    ops/BenchMinAreaRect.cpp
    ops/BenchPadAndStack.cpp
    ops/BenchOSD.cpp
    ops/BenchLabel.cpp
    ops/BenchPairwiseMatcher.cpp
    ops/BenchStack.cpp
    ops/BenchFindHomography.cpp
    ops/BenchCLAHE.cpp
)

# Generate benchmark configuration headers from per-operator JSON
set(CONFIG_OPERATOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../config/operators")
set(CONFIG_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/GenerateBenchConfig.cmake")
set(AXIS_ORDER_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/../config/axis_order.py")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/ops/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

# Create a list to store all generated headers
set(ALL_GENERATED_HEADERS "")

# Create custom commands for each benchmark configuration header
# These will regenerate automatically when the matching operator config changes
foreach(bench_source IN LISTS bench_sources)
    # Extract operator name from filename: ops/BenchResize.cpp -> Resize
    get_filename_component(bench_file_name "${bench_source}" NAME_WLE)
    string(REPLACE "Bench" "" OPERATOR_CAMEL "${bench_file_name}")

    # Convert to lowercase for JSON lookup: Resize -> resize
    string(TOLOWER "${OPERATOR_CAMEL}" OPERATOR_LOWER)

    set(CONFIG_JSON_FILE "${CONFIG_OPERATOR_DIR}/${OPERATOR_LOWER}.json")
    set(OUTPUT_HEADER "${GENERATED_DIR}/Bench${OPERATOR_CAMEL}Config.hpp")

    # Add custom command to generate header at build time
    add_custom_command(
        OUTPUT "${OUTPUT_HEADER}"
        COMMAND ${CMAKE_COMMAND}
            -DJSON_FILE=${CONFIG_JSON_FILE}
            -DOPERATOR=${OPERATOR_LOWER}
            -DOPERATOR_CAMEL=${OPERATOR_CAMEL}
            -DOUTPUT_FILE=${OUTPUT_HEADER}
            -DAXIS_ORDER_HELPER=${AXIS_ORDER_HELPER}
            -P "${CONFIG_GENERATOR}"
        DEPENDS
            "${CONFIG_JSON_FILE}"
            "${CONFIG_GENERATOR}"
            "${AXIS_ORDER_HELPER}"
        COMMENT "Generating config header for ${OPERATOR_CAMEL} from operators/${OPERATOR_LOWER}.json"
        VERBATIM
    )

    list(APPEND ALL_GENERATED_HEADERS "${OUTPUT_HEADER}")
endforeach()

list(LENGTH bench_sources NUM_BENCHMARKS)
message(STATUS "Configured ${NUM_BENCHMARKS} benchmark config header generation rules")

# Metatarget for all benchmarks
add_custom_target(bench_all)

# Shared CUDA fill helpers — typed device-side random / checkerboard fills.
# Built once and linked into every bench executable; replaces the host-side
# uniform_int_distribution + cudaMemcpy round-trip that dominated big-tensor
# setup time on multi-hundred-MB src tensors (1-6 s host fill -> ms GPU fill).
add_library(cvcuda_bench_fill_kernels STATIC BenchFillKernels.cu)
set_target_properties(cvcuda_bench_fill_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF)
target_include_directories(cvcuda_bench_fill_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link cudart so the CUDA toolkit include dir is exposed as -isystem (matching
# every other .cu in the tree). Without this the nvcc host pass treats CUDA's
# own headers as user code and the global -Wredundant-decls / -Werror in
# ConfigCompiler.cmake fires on duplicate decls inside CUDA itself.
target_link_libraries(cvcuda_bench_fill_kernels PUBLIC CUDA::cudart_static)

foreach(bench_source IN LISTS bench_sources)
  get_filename_component(bench_file_name "${bench_source}" NAME_WLE)
  string(REPLACE "Bench" "bench_" algo_name ${bench_file_name})
  string(TOLOWER ${algo_name} bench_name)

  # Extract operator name to find corresponding generated header
  string(REPLACE "Bench" "" OPERATOR_CAMEL "${bench_file_name}")
  set(GENERATED_HEADER "${GENERATED_DIR}/Bench${OPERATOR_CAMEL}Config.hpp")

  # Add executable with generated header as a source dependency
  add_executable(${bench_name} "${bench_source}" "${GENERATED_HEADER}")
  target_link_libraries(${bench_name} PRIVATE cvcuda::nvbench::main cvcuda cvcuda_bench_fill_kernels)
  set_target_properties(${bench_name} PROPERTIES COMPILE_FEATURES cuda_std_17)

  # Add generated config headers directory to include path
  target_include_directories(${bench_name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

  # Ensure benchmark config JSON is up-to-date before building
  add_dependencies(${bench_name} copy_bench_config)

  add_dependencies(bench_all ${bench_name})

  # Install benchmark executable
  install(TARGETS ${bench_name}
          DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT bench)
endforeach()

# Install benchmark runner scripts
install(PROGRAMS
        ${CMAKE_CURRENT_SOURCE_DIR}/../run_bench.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../compare_wheels.py
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT bench)

set(BENCH_INTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../_internal")
set(BENCH_INTERNAL_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/_internal")
set(BENCH_INTERNAL_MODULES
    __init__.py
    axes.py
    baselines.py
    quality.py
    warmup.py)
set(BENCH_INTERNAL_SOURCES "")
file(MAKE_DIRECTORY "${BENCH_INTERNAL_OUTPUT_DIR}")
foreach(module IN LISTS BENCH_INTERNAL_MODULES)
    list(APPEND BENCH_INTERNAL_SOURCES "${BENCH_INTERNAL_SOURCE_DIR}/${module}")
    configure_file("${BENCH_INTERNAL_SOURCE_DIR}/${module}"
                   "${BENCH_INTERNAL_OUTPUT_DIR}/${module}"
                   COPYONLY)
endforeach()
install(FILES ${BENCH_INTERNAL_SOURCES}
        DESTINATION ${CMAKE_INSTALL_BINDIR}/_internal
        COMPONENT bench)

# Copy script to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../run_bench.py
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run_bench.py
               COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../compare_wheels.py
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/compare_wheels.py
               COPYONLY)
