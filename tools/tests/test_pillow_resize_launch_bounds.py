# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
NVCC = os.environ.get("NVCC") or shutil.which("nvcc")
ARCHITECTURES = (75, 80, 86, 87, 89, 90, 100, 110, 120, 121)


@unittest.skipUnless(NVCC, "CUDA compiler required; no GPU is used")
class PillowResizeLaunchBoundsTests(unittest.TestCase):
    def test_fused_tile_bounds_compile_for_supported_ci_architectures(self):
        source = (REPO / "src/cvcuda/priv/OpPillowResize.cu").read_text()
        attribute = re.search(
            r"__global__\s+([^{}]+?)\s+void pillow_fused_u8_tile\(", source
        )
        self.assertIsNotNone(attribute, "Actual fused tile declaration must be tested")
        macros = re.findall(
            r"^#if[^\n]*\n(?:(?!^#endif).)*CVCUDA_PILLOW_U8_TILE_BOUNDS(?:(?!^#endif).)*^#endif",
            source,
            re.MULTILINE | re.DOTALL,
        )
        compiler_environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        supported = subprocess.run(
            [NVCC, "--list-gpu-code"],
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
            env=compiler_environment,
        ).stdout.split()
        self.assertIn("sm_75", supported, "The failing CI target must be exercised")
        with tempfile.TemporaryDirectory(prefix="pillow-launch-bounds-") as directory:
            root = Path(directory)
            probe = root / "probe.cu"
            probe.write_text(
                '#include "PillowResizePolicy.hpp"\n'
                + "\n".join(macros)
                + "\ntemplate<int MinBlocks>\n__global__ "
                + attribute.group(1)
                + " void probe(int *output) { output[threadIdx.x] = threadIdx.x; }\n"
                + "template __global__ void probe<4>(int *);\n"
                + "template __global__ void probe<6>(int *);\n"
            )
            for architecture in ARCHITECTURES:
                target = f"sm_{architecture}"
                with self.subTest(architecture=target):
                    if target not in supported:
                        print(f"UNAVAILABLE {target}: compiler does not support target")
                        continue
                    result = subprocess.run(
                        [
                            NVCC,
                            "--cubin",
                            "-std=c++17",
                            "-O3",
                            "-Werror",
                            "all-warnings",
                            "-arch=" + target,
                            "-I" + str(REPO / "src/cvcuda/priv"),
                            str(probe),
                            "-o",
                            str(root / (target + ".cubin")),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=90,
                        env=compiler_environment,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    print(
                        f"PASS {target}: actual source launch bounds, MinBlocks=4 and6"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
