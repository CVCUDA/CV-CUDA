#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


SCRIPT_PATH = Path(__file__).resolve()
POLICY_MODULE_CANDIDATES = (
    SCRIPT_PATH.parents[2] / "cmake" / "CUDAArchitecturePolicy.cmake",
    SCRIPT_PATH.parents[1]
    / "share"
    / "cvcuda"
    / "tests"
    / "cmake"
    / "CUDAArchitecturePolicy.cmake",
)
POLICY_MODULE = next(
    (path for path in POLICY_MODULE_CANDIDATES if path.is_file()),
    POLICY_MODULE_CANDIDATES[0],
)


class CUDAArchitecturePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = Path(self.temp_dir.name) / "source"
        self.build_dir = Path(self.temp_dir.name) / "build"
        self.source_dir.mkdir()
        (self.source_dir / "CMakeLists.txt").write_text(
            textwrap.dedent(
                f"""\
                cmake_minimum_required(VERSION 3.20.1)
                project(cuda_architecture_policy NONE)
                include("{POLICY_MODULE}")
                cvcuda_detect_cuda_architecture_source()

                set(CMAKE_CUDA_COMPILER_VERSION "${{TEST_CUDA_VERSION}}")
                if(TEST_PROCESSOR STREQUAL "aarch64")
                    set(ARCH_X86_64 OFF)
                    set(ARCH_AARCH64 ON)
                else()
                    set(ARCH_X86_64 ON)
                    set(ARCH_AARCH64 OFF)
                endif()
                cvcuda_configure_cuda_architecture_policy()

                file(WRITE "${{CMAKE_BINARY_DIR}}/policy-result.txt"
                    "architectures=${{CMAKE_CUDA_ARCHITECTURES}}\n"
                    "generated=${{CVCUDA_GENERATED_CUDA_ARCHITECTURES}}\n"
                    "mode=${{CVCUDA_TARGETED_SM8X_CUBINS}}\n"
                    "active=${{CVCUDA_TARGETED_SM8X_CUBINS_ACTIVE}}\n"
                    "source=${{_CVCUDA_CUDA_ARCHITECTURES_SOURCE}}\n")
                """
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def configure(self, *arguments, cudaarchs=None, expect_success=True):
        environment = os.environ.copy()
        if cudaarchs is None:
            environment.pop("CUDAARCHS", None)
        else:
            environment["CUDAARCHS"] = cudaarchs
        command = [
            "cmake",
            "-S",
            str(self.source_dir),
            "-B",
            str(self.build_dir),
            "-DTEST_CUDA_VERSION=12.5",
            *arguments,
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=60,
        )
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return dict(
                line.split("=", maxsplit=1)
                for line in (self.build_dir / "policy-result.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_compact_x86_defaults_follow_toolkit(self):
        result = self.configure()
        self.assertEqual(result["architectures"], "80-real;90-real;75-real")
        self.assertEqual(result["generated"], result["architectures"])
        self.assertEqual(result["active"], "ON")

        result = self.configure("-DTEST_CUDA_VERSION=13.2")
        self.assertEqual(
            result["architectures"],
            "80-real;90-real;100-real;120-real;75-real",
        )
        self.assertEqual(result["source"], "GENERATED")
        self.assertEqual(result["active"], "ON")

    def test_x86_defaults_use_native_turing_without_ptx(self):
        architectures = self.configure()["architectures"].split(";")
        self.assertIn("75-real", architectures)
        self.assertNotIn("75-virtual", architectures)

    def test_aarch64_defaults_follow_platform_and_auto_is_inactive(self):
        result = self.configure("-DTEST_PROCESSOR=aarch64")
        self.assertEqual(
            result["architectures"], "80-real;86-real;89-real;90-real;75-real"
        )
        self.assertEqual(result["active"], "OFF")

        result = self.configure(
            "-DTEST_CUDA_VERSION=13.2",
            "-DTEST_PROCESSOR=aarch64",
            "-DCVCUDA_AARCH64_JETSON=ON",
        )
        self.assertEqual(result["architectures"], "86-real;87-real;89-real")
        self.assertEqual(result["active"], "OFF")

    def test_explicit_cache_values_are_exact(self):
        for architectures in (
            "86-real;89-real",
            "OFF",
            "FALSE",
            "0",
            "native",
            "all",
            "all-major",
        ):
            with self.subTest(architectures=architectures):
                with tempfile.TemporaryDirectory() as build_dir:
                    self.build_dir = Path(build_dir)
                    result = self.configure(
                        f"-DCMAKE_CUDA_ARCHITECTURES={architectures}"
                    )
                    self.assertEqual(result["architectures"], architectures)
                    self.assertEqual(result["generated"], "")
                    self.assertEqual(result["source"], "EXPLICIT_CACHE")
                    self.assertEqual(result["active"], "OFF")

    def test_explicit_canonical_list_activates_auto_deterministically(self):
        for cuda_version, canonical_architectures in (
            ("12.5", "80-real;90-real;75-real"),
            ("13.2", "80-real;90-real;100-real;120-real;75-real"),
        ):
            with self.subTest(cuda_version=cuda_version):
                version_argument = f"-DTEST_CUDA_VERSION={cuda_version}"
                architecture_argument = (
                    f"-DCMAKE_CUDA_ARCHITECTURES={canonical_architectures}"
                )

                with tempfile.TemporaryDirectory() as build_dir:
                    self.build_dir = Path(build_dir)
                    fresh_result = self.configure(
                        version_argument, architecture_argument
                    )

                with tempfile.TemporaryDirectory() as build_dir:
                    self.build_dir = Path(build_dir)
                    self.configure(version_argument)
                    reconfigured_result = self.configure(
                        version_argument, architecture_argument
                    )

                self.assertEqual(fresh_result["architectures"], canonical_architectures)
                self.assertEqual(
                    reconfigured_result["architectures"], canonical_architectures
                )
                self.assertEqual(fresh_result["mode"], "AUTO")
                self.assertEqual(reconfigured_result["mode"], "AUTO")
                self.assertEqual(fresh_result["active"], "ON")
                self.assertEqual(reconfigured_result["active"], "ON")

        self.build_dir = Path(self.temp_dir.name) / "off-build"
        result = self.configure("-DCVCUDA_TARGETED_SM8X_CUBINS=OFF")
        self.assertEqual(result["active"], "OFF")

    def test_auto_compares_effective_architecture_lists(self):
        result = self.configure(
            "-DCMAKE_CUDA_ARCHITECTURES=75-real;90-real;80-real;80-real"
        )
        self.assertEqual(result["architectures"], "75-real;90-real;80-real;80-real")
        self.assertEqual(result["active"], "ON")

    def test_environment_is_exact_only_for_a_fresh_cache(self):
        result = self.configure(cudaarchs="86-real;89-real")
        self.assertEqual(result["architectures"], "86-real;89-real")
        self.assertEqual(result["source"], "ENVIRONMENT")
        self.assertEqual(result["active"], "OFF")

        result = self.configure(cudaarchs="native")
        self.assertEqual(result["architectures"], "86-real;89-real")
        self.assertEqual(result["source"], "EXPLICIT_CACHE")

    def test_cache_false_value_wins_over_environment(self):
        result = self.configure("-DCMAKE_CUDA_ARCHITECTURES=OFF", cudaarchs="native")
        self.assertEqual(result["architectures"], "OFF")
        self.assertEqual(result["source"], "EXPLICIT_CACHE")

    def test_toolchain_cache_value_is_exact(self):
        toolchain_file = self.source_dir / "toolchain.cmake"
        toolchain_file.write_text(
            'set(CMAKE_CUDA_ARCHITECTURES "89-real" CACHE STRING "" FORCE)\n',
            encoding="utf-8",
        )

        result = self.configure(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}")
        self.assertEqual(result["architectures"], "89-real")
        self.assertEqual(result["generated"], "")
        self.assertEqual(result["source"], "EXPLICIT_CACHE")
        self.assertEqual(result["active"], "OFF")

    def test_generated_cache_remains_generated_until_explicitly_replaced(self):
        self.configure()
        result = self.configure(cudaarchs="native")
        self.assertEqual(result["architectures"], "80-real;90-real;75-real")
        self.assertEqual(result["source"], "GENERATED")
        self.assertEqual(result["active"], "ON")

        result = self.configure("-DCMAKE_CUDA_ARCHITECTURES=86-real")
        self.assertEqual(result["architectures"], "86-real")
        self.assertEqual(result["generated"], "")
        self.assertEqual(result["source"], "EXPLICIT_CACHE")
        self.assertEqual(result["active"], "OFF")

        result = self.configure()
        self.assertEqual(result["architectures"], "86-real")
        self.assertEqual(result["source"], "EXPLICIT_CACHE")

    def test_targeted_cubin_mode_is_validated_and_canonicalized(self):
        result = self.configure("-DCVCUDA_TARGETED_SM8X_CUBINS=off")
        self.assertEqual(result["mode"], "OFF")
        self.assertEqual(result["active"], "OFF")

        with tempfile.TemporaryDirectory() as build_dir:
            self.build_dir = Path(build_dir)
            result = self.configure(
                "-DCMAKE_CUDA_ARCHITECTURES=86-real",
                "-DCVCUDA_TARGETED_SM8X_CUBINS=on",
            )
            self.assertEqual(result["mode"], "ON")
            self.assertEqual(result["active"], "ON")

        with tempfile.TemporaryDirectory() as build_dir:
            self.build_dir = Path(build_dir)
            result = self.configure(
                "-DCVCUDA_TARGETED_SM8X_CUBINS=sometimes", expect_success=False
            )
            self.assertIn("must be AUTO, ON, or OFF", result.stdout + result.stderr)

    def test_targeted_source_helper_is_active_gated_and_filter_safe(self):
        helper_source_dir = Path(self.temp_dir.name) / "helper-source"
        helper_source_dir.mkdir()
        (helper_source_dir / "targeted.cu").write_text(
            "__global__ void targeted() {}\n", encoding="utf-8"
        )
        (helper_source_dir / "CMakeLists.txt").write_text(
            textwrap.dedent(
                f"""\
                cmake_minimum_required(VERSION 3.20.1)
                project(targeted_source_helper NONE)
                include("{POLICY_MODULE}")

                set(CVCUDA_TARGETED_SM8X_CUBINS_ACTIVE "${{TEST_ACTIVE}}")
                set(CMAKE_CUDA_ARCHITECTURES "${{TEST_GLOBAL_ARCHITECTURES}}")
                set(CMAKE_CUDA_ARCHITECTURES_ALL "${{TEST_GLOBAL_ARCHITECTURES_ALL}}")
                set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "${{TEST_GLOBAL_ARCHITECTURES_ALL_MAJOR}}")
                set(CMAKE_CUDA_ARCHITECTURES_NATIVE "${{TEST_GLOBAL_ARCHITECTURES_NATIVE}}")
                set(SELECTED_SOURCES unselected.cu)
                if(TEST_SOURCE_SELECTED)
                    list(APPEND SELECTED_SOURCES targeted.cu)
                endif()

                cvcuda_add_targeted_cuda_architectures_to_sources(
                    ARCHITECTURES ${{TEST_ARCHITECTURES}}
                    SOURCES targeted.cu
                    SELECTED_SOURCES_VAR SELECTED_SOURCES)
                # Repeated declarations must not duplicate nvcc flags.
                cvcuda_add_targeted_cuda_architectures_to_sources(
                    ARCHITECTURES ${{TEST_ARCHITECTURES}}
                    SOURCES targeted.cu
                    SELECTED_SOURCES_VAR SELECTED_SOURCES)

                get_property(TARGETED_OPTIONS SOURCE targeted.cu PROPERTY COMPILE_OPTIONS)
                list(LENGTH TARGETED_OPTIONS TARGETED_OPTION_COUNT)
                string(JOIN "|" TARGETED_OPTIONS_JOINED ${{TARGETED_OPTIONS}})
                file(WRITE "${{CMAKE_BINARY_DIR}}/helper-result.txt"
                    "count=${{TARGETED_OPTION_COUNT}}\noptions=${{TARGETED_OPTIONS_JOINED}}\n")
                """
            ),
            encoding="utf-8",
        )

        def configure_helper(
            active,
            source_selected,
            architectures="86-real;89",
            global_architectures="",
            global_architectures_all="",
            global_architectures_all_major="",
            global_architectures_native="",
            expect_success=True,
        ):
            with tempfile.TemporaryDirectory() as build_dir:
                result = subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(helper_source_dir),
                        "-B",
                        build_dir,
                        f"-DTEST_ACTIVE={active}",
                        f"-DTEST_SOURCE_SELECTED={source_selected}",
                        f"-DTEST_ARCHITECTURES={architectures}",
                        f"-DTEST_GLOBAL_ARCHITECTURES={global_architectures}",
                        f"-DTEST_GLOBAL_ARCHITECTURES_ALL={global_architectures_all}",
                        f"-DTEST_GLOBAL_ARCHITECTURES_ALL_MAJOR={global_architectures_all_major}",
                        f"-DTEST_GLOBAL_ARCHITECTURES_NATIVE={global_architectures_native}",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if not expect_success:
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    return result
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return dict(
                    line.split("=", maxsplit=1)
                    for line in (Path(build_dir) / "helper-result.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )

        result = configure_helper("ON", "ON")
        self.assertEqual(result["count"], "2")
        self.assertEqual(
            result["options"],
            "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_86,code=sm_86>"
            "|$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>",
        )
        self.assertNotIn("code=compute_", result["options"])

        result = configure_helper("ON", "ON", global_architectures="86;89-real")
        self.assertEqual(result, {"count": "0", "options": ""})

        result = configure_helper("ON", "ON", global_architectures="86-real;89-virtual")
        self.assertEqual(result["count"], "1")
        self.assertEqual(
            result["options"],
            "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>",
        )

        result = configure_helper(
            "ON", "ON", global_architectures="86-virtual;89-virtual"
        )
        self.assertEqual(result["count"], "2")

        result = configure_helper(
            "ON",
            "ON",
            global_architectures="all",
            global_architectures_all="80;86;89;90",
        )
        self.assertEqual(result, {"count": "0", "options": ""})

        result = configure_helper(
            "ON",
            "ON",
            global_architectures="all-major",
            global_architectures_all_major="80;90",
        )
        self.assertEqual(result["count"], "2")

        result = configure_helper(
            "ON",
            "ON",
            global_architectures="native",
            global_architectures_native="86",
        )
        self.assertEqual(result["count"], "1")
        self.assertEqual(
            result["options"],
            "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>",
        )

        result = configure_helper("ON", "OFF")
        self.assertEqual(result, {"count": "0", "options": ""})

        result = configure_helper("OFF", "ON")
        self.assertEqual(result, {"count": "0", "options": ""})

        result = configure_helper("ON", "ON", "89-virtual", expect_success=False)
        self.assertIn(
            "Targeted CUDA architecture must be 86, 86-real, 89, or 89-real",
            result.stdout + result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
