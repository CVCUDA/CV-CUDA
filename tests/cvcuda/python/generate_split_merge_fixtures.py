# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate channel-copy fixtures with Pillow 12.3.0 in an isolated environment."""

import json
from pathlib import Path

import PIL
from PIL import Image


def main():
    if PIL.__version__ != "12.3.0":
        raise RuntimeError("Fixture generation requires Pillow 12.3.0")
    fixtures = []
    representatives = {}
    for mode, representative in {
        "L": "L",
        "LA": "LA",
        "La": "LA",
        "RGB": "RGB",
        "RGBA": "RGBA",
        "RGBa": "RGBA",
        "RGBX": "RGBA",
        "CMYK": "RGBA",
        "YCbCr": "RGB",
        "HSV": "RGB",
    }.items():
        channels = Image.getmodebands(mode)
        source = bytes((index * 37 + 11) % 256 for index in range(6 * channels))
        image = Image.frombytes(mode, (3, 2), source)
        bands = image.split()
        payload = {
            "channels": channels,
            "shape": [2, 3, channels],
            "source": list(image.tobytes()),
            "bands": [list(band.tobytes()) for band in bands],
            "merged": list(Image.merge(mode, bands).tobytes()),
        }
        if mode == representative:
            representatives[mode] = payload
            fixtures.append({"mode": mode, **payload})
        elif payload != representatives[representative]:
            raise RuntimeError(f"Pillow mode {mode} differs from {representative}")
    destination = Path(__file__).with_name("split_merge_pillow.json")
    destination.write_text(
        json.dumps({"pillow": PIL.__version__, "fixtures": fixtures}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
