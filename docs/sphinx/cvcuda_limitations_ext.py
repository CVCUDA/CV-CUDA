# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Sphinx extension for CV-CUDA Python operator docs.

Two jobs:
1. Inject a "Limitations" table (sourced from the C API Doxygen XML) under
   every cvcuda.* operator page.
2. Provide a ``cvcuda-autofunction`` directive that splits pybind11's
   ``Overloaded function.`` docstring into one ``.. py:function::`` entry per
   overload.  Sphinx autodoc does not split this format natively — it renders
   the numbered list as prose.
"""

import importlib
import os
import re
import textwrap
import xml.etree.ElementTree as ET
from typing import Optional

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.ext.autodoc.directive import AutodocDirective
from sphinx.ext.napoleon import Config as NapoleonConfig, GoogleDocstring
from sphinx.util.docutils import SphinxDirective

# --------------------------------------------------------------------------- #
# Pybind11 overload splitter                                                   #
# --------------------------------------------------------------------------- #

_OVERLOAD_ENTRY = re.compile(r"^\d+\.\s+(\w+\(.*)")
_NAPOLEON_CONFIG = NapoleonConfig(
    napoleon_use_param=True, napoleon_use_rtype=True, napoleon_use_keyword=True
)


def _parse_pybind11_overloads(
    docstring: str,
) -> Optional[list[tuple[str, list[str]]]]:
    """Parse pybind11's ``Overloaded function.`` docstring into (sig, body) pairs."""
    if not docstring or "Overloaded function." not in docstring:
        return None
    overloads: list[tuple[str, list[str]]] = []
    current_sig: Optional[str] = None
    current_body: list[str] = []
    for raw_line in docstring.splitlines():
        m = _OVERLOAD_ENTRY.match(raw_line)
        if m:
            if current_sig is not None:
                overloads.append((current_sig, current_body))
            current_sig = m.group(1)
            current_body = []
        elif current_sig is not None:
            current_body.append(raw_line)
    if current_sig is not None:
        overloads.append((current_sig, current_body))
    return overloads if len(overloads) >= 2 else None


def _body_to_rst(body_lines: list[str]) -> str:
    """Dedent and run Napoleon so Args:/Returns: become :param:/:returns:."""
    text = textwrap.dedent("\n".join(body_lines)).strip()
    return str(GoogleDocstring(text, _NAPOLEON_CONFIG)) if text else ""


class CvcudaAutofunctionDirective(SphinxDirective):
    """Replacement for ``autofunction`` that splits pybind11 overloads.

    * Single-overload functions delegate to the stock ``autofunction``.
    * Overloaded functions emit one ``.. py:function::`` per overload.
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec = {}

    def run(self) -> list[nodes.Node]:
        fullname = self.arguments[0]
        modname, attrname = fullname.rsplit(".", 1)
        try:
            mod = importlib.import_module(modname)
            func = getattr(mod, attrname)
        except (ImportError, AttributeError):
            return self._autofunction(fullname)

        raw_doc = getattr(func, "__doc__", "") or ""
        overloads = _parse_pybind11_overloads(raw_doc)
        if not overloads:
            return self._autofunction(fullname)

        return self._split_overloads(fullname, overloads)

    def _autofunction(self, fullname: str) -> list[nodes.Node]:
        ad = AutodocDirective(
            "autofunction",
            [fullname],
            {},
            StringList([]),
            self.lineno,
            self.content_offset,
            "",
            self.state,
            self.state_machine,
        )
        return ad.run()

    def _split_overloads(
        self, fullname: str, overloads: list[tuple[str, list[str]]]
    ) -> list[nodes.Node]:
        base = _func_base_name(fullname)
        rst_lines: list[str] = []

        for idx, (sig, body) in enumerate(overloads):
            rst_body = _body_to_rst(body)

            rst_lines.append(f".. py:function:: {sig}")
            rst_lines.append("   :module: cvcuda")
            if idx > 0:
                # Avoid duplicate anchor targets — first overload owns the
                # canonical #cvcuda.<funcname> anchor.
                rst_lines.append("   :noindex:")
            rst_lines.append("")

            for line in rst_body.splitlines():
                rst_lines.append(f"   {line}")
            if rst_body:
                rst_lines.append("")

            if base in _cache:
                rst_lines.append("   .. rubric:: Limitations")
                rst_lines.append("")
                for lim_line in _cache[base]:
                    rst_lines.append(f"   {lim_line}")
                rst_lines.append("")

        vl = StringList(rst_lines, source=f"<cvcuda-autofunction:{fullname}>")
        wrapper = nodes.container()
        wrapper.document = self.state.document
        self.state.nested_parse(vl, 0, wrapper)
        return wrapper.children


# --------------------------------------------------------------------------- #
# Mapping: Python function name (bare, no _into) → Doxygen group name suffix  #
# The XML files live at {breathe_xml_dir}/group__NVCV__C__ALGORITHM__{suffix}.xml
# --------------------------------------------------------------------------- #
_FUNC_TO_GROUP: dict[str, str] = {
    "adaptivethreshold": "ADAPTIVETHRESHOLD",
    "advcvtcolor": "__ADV__CVT__COLOR",
    "averageblur": "AVERAGEBLUR",
    "bilateral_filter": "BILATERAL__FILTER",
    "bndbox": "__BND__BOX",
    "boxblur": "__BOX__BLUR",
    "brightness_contrast": "BRIGHTNESS__CONTRAST",
    "center_crop": "CENTER__CROP",
    "channelreorder": "CHANNEL__REORDER",
    "clahe": "__CLAHE",
    "color_twist": "COLOR__TWIST",
    "composite": "COMPOSITE",
    "conv2d": "CONV2D",
    "convertto": "CONVERT__TO",
    "copymakeborder": "COPYMAKEBORDER",
    "copymakeborderstack": "COPYMAKEBORDER",
    "crop_flip_normalize_reformat": "CROP__FLIP__NORMALIZE__REFORMAT",
    "customcrop": "CUSTOM__CROP",
    "cvtcolor": "CVTCOLOR",
    "erase": "ERASE",
    "findhomography": "FIND__HOMOGRAPHY",
    "flip": "FLIP",
    "gamma_contrast": "GAMMA__CONTRAST",
    "gaussian": "GAUSSIAN",
    "gaussiannoise": "GAUSSIAN__NOISE",
    "histogram": "__HISTOGRAM",
    "histogrameq": "__HISTOGRAM__EQ",
    "hq_resize": "HQ__RESIZE",
    "inpaint": "INPAINT",
    "joint_bilateral_filter": "JOINT__BILATERAL__FILTER",
    "label": "LABEL",
    "laplacian": "LAPLACIAN",
    "match": "PAIRWISE__MATCHER",
    "max_loc": "MINMAXLOC",
    "median_blur": "MEDIAN__BLUR",
    "minarearect": "__MIN__AREA__RECT",
    "min_loc": "MINMAXLOC",
    "min_max_loc": "MINMAXLOC",
    "morphology": "MORPHOLOGY",
    "nms": "NON__MAXIMUM__SUPPRESSION",
    "normalize": "NORMALIZE",
    "osd": "__O__S__D",
    "padandstack": "PADANDSTACK",
    "pillowresize": "PILLOW__RESIZE",
    "random_resized_crop": "RANDOMRESIZEDCROP",
    "reformat": "REFORMAT",
    "remap": "REMAP",
    "resize": "RESIZE",
    "resize_crop_convert_reformat": "__RESIZE__CROP",
    "rotate": "ROTATE",
    "sift": "SIFT",
    "stack": "__STACK",
    "threshold": "THRESHOLD",
    "warp_affine": "WARP__AFFINE",
    "warp_perspective": "WARP__PERSPECTIVE",
}

# Module-level cache populated at builder-inited time
_cache: dict[str, list[str]] = {}


# --------------------------------------------------------------------------- #
# XML → RST conversion helpers                                                 #
# --------------------------------------------------------------------------- #


def _table_to_rst(table_elem) -> list[str]:
    rows = table_elem.findall("row")
    if not rows:
        return []
    lines = [".. list-table::", "   :header-rows: 1", "   :widths: auto", ""]
    for row in rows:
        entries = row.findall("entry")
        for j, entry in enumerate(entries):
            text = "".join(entry.itertext()).strip()
            lines.append(f"   * - {text}" if j == 0 else f"     - {text}")
    lines.append("")
    return lines


def _verbatim_to_rst(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    lines = [".. code-block:: text", ""]
    for line in text.splitlines():
        lines.append(f"   {line}")
    lines.append("")
    return lines


def _para_to_rst(para) -> list[str]:
    """Convert a <para> element to RST lines, stopping before <parameterlist>."""
    table = para.find("table")
    if table is not None:
        rst: list[str] = []
        if para.text and para.text.strip():
            rst.append(para.text.strip())
            rst.append("")
        rst.extend(_table_to_rst(table))
        return rst

    lines: list[str] = []
    buf = para.text or ""

    for child in para:
        if child.tag == "parameterlist":
            break
        elif child.tag == "verbatim":
            if buf.strip():
                lines.append(buf.strip())
                lines.append("")
            lines.extend(_verbatim_to_rst(child.text or ""))
            buf = child.tail or ""
        elif child.tag == "table":
            if buf.strip():
                lines.append(buf.strip())
                lines.append("")
            lines.extend(_table_to_rst(child))
            buf = child.tail or ""
        elif child.tag == "ref":
            buf += (child.text or "") + (child.tail or "")
        else:
            buf += "".join(child.itertext()) + (child.tail or "")

    if buf.strip():
        lines.append(buf.strip())
        lines.append("")
    return lines


def _parse_limitations(xml_path: str) -> list[str]:
    """Extract Limitations section from a Doxygen group XML file as RST lines."""
    try:
        tree = ET.parse(xml_path)
    except (FileNotFoundError, ET.ParseError):
        return []

    root = tree.getroot()

    sections: list[tuple[list[str], list[str]]] = []
    for memberdef in root.iter("memberdef"):
        desc = memberdef.find("detaileddescription")
        if desc is None:
            continue
        paras = list(desc)

        lim_start: Optional[int] = None
        for i, para in enumerate(paras):
            if "".join(para.itertext()).strip() == "Limitations:":
                lim_start = i + 1
                break
        if lim_start is None:
            continue

        result: list[str] = []
        for para in paras[lim_start:]:
            has_table = para.find("table") is not None
            has_verbatim = para.find("verbatim") is not None
            has_plist = para.find("parameterlist") is not None
            has_text = bool(para.text and para.text.strip())
            if has_plist and not has_table and not has_verbatim and not has_text:
                break
            result.extend(_para_to_rst(para))

        if result:
            name = (memberdef.findtext("name", default="") or "").strip()
            for names, lines in sections:
                if lines == result:
                    if name and name not in names:
                        names.append(name)
                    break
            else:
                sections.append(([name] if name else [], result))

    if not sections:
        return []
    if len(sections) == 1:
        return sections[0][1]

    combined: list[str] = []
    for names, lines in sections:
        symbols = ", ".join(f"``{name}``" for name in names if name)
        if symbols:
            combined.extend((f"**Applies to {symbols}:**", ""))
        combined.extend(lines)
    return combined


# --------------------------------------------------------------------------- #
# Sphinx event handlers                                                        #
# --------------------------------------------------------------------------- #


def _load_cache(app) -> None:
    """Populate _cache at builder-inited from the Doxygen XML directory."""
    global _cache
    _cache = {}

    breathe_projects = getattr(app.config, "breathe_projects", {})
    xml_dir = breathe_projects.get("cvcuda", "")
    if not xml_dir or not os.path.isdir(xml_dir):
        return

    for func_name, group_suffix in _FUNC_TO_GROUP.items():
        if func_name in _cache:
            continue
        xml_path = os.path.join(
            xml_dir, f"group__NVCV__C__ALGORITHM__{group_suffix}.xml"
        )
        if os.path.isfile(xml_path):
            lines = _parse_limitations(xml_path)
            if lines:
                _cache[func_name] = lines


def _func_base_name(full_name: str) -> Optional[str]:
    """Return the bare function name from 'cvcuda.resize_into' → 'resize'."""
    if not full_name.startswith("cvcuda."):
        return None
    base = full_name.removeprefix("cvcuda.")
    for suffix in ("_into_with_op", "_into"):
        if base.endswith(suffix):
            return base.removesuffix(suffix)
    return base


def _inject(app, what, name, obj, options, lines) -> None:
    """Append Limitations to single-overload operators.

    Overloaded operators are handled entirely by CvcudaAutofunctionDirective
    (which emits ``.. py:function::`` blocks that don't trigger autodoc
    events), so this hook only fires for single-overload functions delegated
    to stock ``autofunction``.
    """
    if what not in ("function", "method") or not name.startswith("cvcuda."):
        return
    base = _func_base_name(name)
    if base not in _cache:
        return

    lines.append("")
    lines.append(".. rubric:: Limitations")
    lines.append("")
    lines.extend(_cache[base])


def setup(app):
    app.connect("builder-inited", _load_cache)
    app.connect("autodoc-process-docstring", _inject)
    app.add_directive("cvcuda-autofunction", CvcudaAutofunctionDirective)
    return {"version": "1.0", "parallel_read_safe": True}
