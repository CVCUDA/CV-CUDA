# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
CXX = shutil.which("g++") or shutil.which("clang++")
FILTERS = (
    "BilinearFilter",
    "BicubicFilter",
    "BoxFilter",
    "HammingFilter",
    "LanczosFilter",
)
PIXEL_TYPES = ("unsigned char", "std::uint16_t", "std::int16_t", "float")


def enclosing_constexpr_guard(source, marker):
    position = source.index(marker)
    guards = list(re.finditer(r"if\s+constexpr\s*\(", source[:position]))
    if not guards:
        raise AssertionError(f"No compile-time guard before {marker}")
    start = guards[-1].end()
    depth = 1
    for offset in range(start, position):
        if source[offset] == "(":
            depth += 1
        elif source[offset] == ")":
            depth -= 1
            if depth == 0:
                return source[start:offset]
    raise AssertionError(f"Unterminated compile-time guard before {marker}")


class PillowResizeTileDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CXX is None:
            raise RuntimeError("A host C++17 compiler (g++ or clang++) is required")
        source = (REPO / "src/cvcuda/priv/OpPillowResize.cu").read_text()
        cls.function = source.split("void pillow_resize_var_shape(", 1)[1].split(
            "// The element types", 1
        )[0]
        markers = (
            "= PillowVarShapeStorageDisjoint<kPlanar>",
            "all_u8_tiles_fit = all_u8_tiles_fit",
            "const PillowVarShapeU8TileAdapter<kPlanar> images",
        )
        declarations = re.findall(
            r"constexpr\s+bool\s+can_tile\s*=[^;]+;", cls.function
        )
        cls.guards = [
            enclosing_constexpr_guard(cls.function, marker) for marker in markers
        ]
        program = [
            "#include <cstdint>",
            "#include <type_traits>",
            "#include <iostream>",
        ]
        program.extend(f"struct {filter_name} {{}};" for filter_name in FILTERS)
        for phase, guard in enumerate(cls.guards):
            program.append(
                "template<class Filter, class elem_type, bool kPlanar> "
                f"constexpr bool eligible_{phase}() {{ "
                + "\n".join(declarations)
                + f"return ({guard}); }}"
            )
        program.append("int main() {")
        for phase in range(len(markers)):
            for pixel_type in PIXEL_TYPES:
                for filter_name in FILTERS:
                    for planar in (False, True):
                        key = f"{phase}|{pixel_type}|{filter_name}|{int(planar)}"
                        program.append(
                            f'std::cout << "{key}:" << eligible_{phase}<'
                            f"{filter_name}, {pixel_type}, {str(planar).lower()}>()"
                            " << '\\n';"
                        )
        program.append("}")
        with tempfile.TemporaryDirectory(prefix="pillow-tile-dispatch-") as directory:
            root = Path(directory)
            probe = root / "probe.cpp"
            binary = root / "probe"
            probe.write_text("\n".join(program))
            result = subprocess.run(
                [
                    CXX,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    str(probe),
                    "-o",
                    str(binary),
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            output = subprocess.run(
                [str(binary)], text=True, capture_output=True, check=True, timeout=10
            ).stdout
        cls.results = dict(line.rsplit(":", 1) for line in output.splitlines())
        assert len(cls.results) == len(markers) * len(PIXEL_TYPES) * len(FILTERS) * 2

    def test_interleaved_keeps_existing_separable_dispatch(self):
        for key, value in self.results.items():
            if key.endswith("|0"):
                with self.subTest(predicate=key):
                    self.assertEqual(value, "0")

    def test_planar_rgb8_linear_and_cubic_remain_eligible(self):
        for phase in range(3):
            for filter_name in FILTERS[:2]:
                key = f"{phase}|unsigned char|{filter_name}|1"
                self.assertEqual(self.results[key], "1", key)

    def test_other_pixel_types_and_filters_remain_ineligible(self):
        for phase in range(3):
            for pixel_type in PIXEL_TYPES:
                for filter_name in FILTERS:
                    if pixel_type == "unsigned char" and filter_name in FILTERS[:2]:
                        continue
                    key = f"{phase}|{pixel_type}|{filter_name}|1"
                    self.assertEqual(self.results[key], "0", key)

    def test_storage_scan_capacity_check_and_launch_agree(self):
        for pixel_type in PIXEL_TYPES:
            for filter_name in FILTERS:
                for planar in (0, 1):
                    suffix = f"{pixel_type}|{filter_name}|{planar}"
                    self.assertEqual(
                        len({self.results[f"{phase}|{suffix}"] for phase in range(3)}),
                        1,
                        suffix,
                    )

    def test_launch_still_requires_channel_capacity_and_nonaliasing_guards(self):
        self.assertRegex(
            self.function,
            r"if\s*\(channels == 3 && all_u8_tiles_fit && storage_disjoint\)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
