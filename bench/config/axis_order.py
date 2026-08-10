#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic axis ordering shared by C++ and Python benchmarks."""

import argparse
import json
from pathlib import Path


AXIS_CATEGORIES = ("string_axes", "int64_axes", "float64_axes")


def order_axis_names(axis_mappings):
    """Return unique axis names in case-sensitive UTF-8 byte order."""
    names = {name for mapping in axis_mappings for name in mapping}
    return sorted(names, key=lambda name: name.encode("utf-8"))


def operator_axis_order(config_path, operator):
    """Return the union axis order for one generated C++ benchmark."""
    document = json.loads(Path(config_path).read_text())
    if "configs" in document:
        entries = (
            document["configs"].values()
            if document.get("benchmark") == operator
            else ()
        )
    else:
        entries = (
            entry
            for entry in document.values()
            if isinstance(entry, dict) and entry.get("benchmark") == operator
        )

    mappings = []
    for entry in entries:
        mappings.extend(entry.get(category, {}) for category in AXIS_CATEGORIES)
    return order_axis_names(mappings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    print("\n".join(operator_axis_order(args.config, args.operator)))


if __name__ == "__main__":
    main()
