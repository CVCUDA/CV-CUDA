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

from pathlib import Path
import xml.etree.ElementTree as ET
import os
import sys

outdir = Path(sys.argv[1])

if not os.path.exists(outdir):
    os.makedirs(outdir)

xmlRoot = sys.argv[2]

for i in os.listdir(xmlRoot):
    group_path = os.path.join(xmlRoot, i)
    if os.path.isfile(group_path) and "group__" in i:
        tree = ET.parse(group_path)
        root = tree.getroot()
        for compounddef in root.iter("compounddef"):
            group_name = compounddef.attrib["id"]
            group_label = compounddef.find("compoundname").text
            group_title = compounddef.find("title").text
            outfile = outdir / (group_name + ".rst")
            output = ":orphan:\n\n"
            output += group_title + "\n"
            output += "=" * len(group_title) + "\n\n"
            output += f".. doxygengroup:: {group_label}\n"
            output += "   :project: cvcuda\n"
            outfile.write_text(output)
