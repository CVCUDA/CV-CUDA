#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


# This script runs from <build>/bin locally or <prefix>/bin in package CI.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECT_DEV = os.environ.get("CVCUDA_PACKAGE_TEST_EXPECT_DEV") == "1"
CMAKE = shutil.which("cmake")

INSTALL_LIBDIR = "@CMAKE_INSTALL_LIBDIR@"
LIBRARY_ARCHITECTURE = "@CMAKE_LIBRARY_ARCHITECTURE@"
CVCUDA_VERSION_API = "@CVCUDA_VERSION_API@"
CVCUDA_VERSION_FULL = "@CVCUDA_VERSION_FULL@"
NVCV_VERSION_FULL = "@NVCV_VERSION_FULL@"
NVCV_VERSION_MAJOR = "@NVCV_VERSION_MAJOR@"


class PackageConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_script = PACKAGE_ROOT / "cmake_install.cmake"
        is_build_tree = install_script.is_file()
        if CMAKE is None:
            if is_build_tree or EXPECT_DEV:
                raise RuntimeError("cmake is required to test the development package")
            raise unittest.SkipTest("cmake is not available")

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp_dir.cleanup)
        cls.temp_root = Path(cls.temp_dir.name)
        cls.package_prefix = PACKAGE_ROOT
        if is_build_tree:
            cls.package_prefix = cls.temp_root / "prefix"
            for component in ("lib", "dev"):
                cls.run_command(
                    CMAKE,
                    "--install",
                    str(PACKAGE_ROOT),
                    "--prefix",
                    str(cls.package_prefix),
                    "--component",
                    component,
                )

        cls.package_dirs = {
            package: cls.package_prefix / INSTALL_LIBDIR / "cmake" / package
            for package in ("cvcuda", "nvcv_types")
        }
        for package, path in cls.package_dirs.items():
            if not (path / f"{package}-config.cmake").is_file():
                if is_build_tree or EXPECT_DEV:
                    raise RuntimeError(
                        f"Development package files are missing from {path}"
                    )
                raise unittest.SkipTest("development package files are not installed")

    @staticmethod
    def run_command(*command):
        subprocess.run(command, check=True, timeout=180)

    def configure(self, name, cmake_lists):
        source_dir = self.temp_root / f"{name}-source"
        build_dir = self.temp_root / f"{name}-build"
        source_dir.mkdir()
        (source_dir / "CMakeLists.txt").write_text(
            textwrap.dedent(cmake_lists), encoding="utf-8"
        )
        self.run_command(
            CMAKE,
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            # project(NONE) cannot infer the multiarch path needed by CUDA::cudart_static.
            f"-DCMAKE_LIBRARY_ARCHITECTURE={LIBRARY_ARCHITECTURE}",
            "-DCMAKE_FIND_USE_PACKAGE_ROOT_PATH=FALSE",
            "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE",
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE",
            "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE",
        )

    def test_cvcuda_discovers_public_dependencies(self):
        alias_dir = self.temp_root / "cmake"
        alias_dir.mkdir()
        package_aliases = {
            package: alias_dir / f"{package}{NVCV_VERSION_MAJOR}"
            for package in self.package_dirs
        }
        for package, alias in package_aliases.items():
            alias.symlink_to(self.package_dirs[package], target_is_directory=True)

        for layout, package_dirs in {
            "direct": self.package_dirs,
            "package-alias": package_aliases,
        }.items():
            with self.subTest(layout=layout):
                self.configure(
                    f"dependencies-{layout}",
                    f"""\
            cmake_minimum_required(VERSION 3.20.1)
            project(cvcuda_consumer NONE)

            find_package(cvcuda {CVCUDA_VERSION_API} CONFIG REQUIRED
                PATHS "{package_dirs['cvcuda']}" NO_DEFAULT_PATH)
            if(NOT nvcv_types_DIR STREQUAL "{package_dirs['nvcv_types']}")
                message(FATAL_ERROR
                    "nvcv_types resolved outside the package layout under test")
            endif()
            foreach(target IN ITEMS cvcuda nvcv_types CUDA::cudart_static)
                if(NOT TARGET ${{target}})
                    message(FATAL_ERROR "cvcuda did not discover ${{target}}")
                endif()
            endforeach()
            """,
                )

    def test_zero_major_packages_reject_previous_minor(self):
        package_versions = {
            "cvcuda": CVCUDA_VERSION_FULL,
            "nvcv_types": NVCV_VERSION_FULL,
        }
        for package, installed_version in package_versions.items():
            with self.subTest(package=package):
                major, minor = map(int, installed_version.split(".")[:2])
                if major != 0 or minor == 0:
                    self.skipTest("only applies to nonzero 0.x releases")

                requested_version = f"0.{minor - 1}"
                preload = ""
                if package == "cvcuda":
                    # Preload the raw pre-fix export's referenced targets so a
                    # SameMajorVersion regression reaches the assertion below.
                    preload = f"""
                    find_package(CUDAToolkit REQUIRED)
                    find_package(nvcv_types CONFIG REQUIRED
                        PATHS "{self.package_dirs['nvcv_types']}" NO_DEFAULT_PATH)
                    """
                self.configure(
                    f"{package}-previous-minor",
                    f"""\
                    cmake_minimum_required(VERSION 3.20.1)
                    project({package}_version_consumer NONE)

                    {preload}
                    find_package({package} {requested_version} CONFIG QUIET
                        PATHS "{self.package_dirs[package]}" NO_DEFAULT_PATH)
                    if({package}_FOUND)
                        message(FATAL_ERROR
                            "{package} accepted incompatible {requested_version}")
                    endif()
                    if(NOT "{installed_version}" IN_LIST {package}_CONSIDERED_VERSIONS)
                        message(FATAL_ERROR
                            "{package} did not consider the installed version")
                    endif()
                    """,
                )


if __name__ == "__main__":
    unittest.main()
