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

"""Conservative Python-binding API snapshots shared by optimization gates."""

from __future__ import annotations

import re


def _normalize_cpp_surface(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _without_cpp_comments(text):
    """Blank comments while preserving line breaks for declaration parsing."""

    def blank(match):
        return re.sub(r"[^\n]", " ", match.group(0))

    return re.sub(r"//[^\n]*|/\*.*?\*/", blank, text or "", flags=re.S)


def _top_level_commas(text):
    """Return comma offsets outside nested template/declarator delimiters."""
    opening = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closing = set(opening.values())
    stack = []
    commas = []
    for index, char in enumerate(text):
        if char in opening:
            stack.append(opening[char])
        elif char in closing:
            if not stack or stack.pop() != char:
                return None
        elif char == "," and not stack:
            commas.append(index)
    return commas if not stack else None


def _unsupported_typedef_names(declarator):
    """Extract names without pretending to resolve an unsupported declarator."""
    opening = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closing = set(opening.values())
    stack = []
    names = set()
    non_names = {"alignas", "decltype", "noexcept", "sizeof", "typeid"}

    for index, char in enumerate(declarator):
        if char in opening:
            if not stack and char in "([":
                suffix = declarator[index:]
                grouped = None
                if char == "(":
                    grouped = re.match(
                        r"\(\s*(?:(?:[A-Za-z_]\w*\s*::\s*)+)?[*&]+\s*"
                        r"([A-Za-z_]\w*)",
                        suffix,
                    )
                if grouped:
                    names.add(grouped.group(1))
                else:
                    direct = re.search(r"([A-Za-z_]\w*)\s*$", declarator[:index])
                    if direct and direct.group(1) not in non_names:
                        names.add(direct.group(1))
            stack.append(opening[char])
        elif char in closing:
            if not stack or stack.pop() != char:
                return ()

    trailing_pointer = re.search(r"[*&]\s*([A-Za-z_]\w*)\s*$", declarator)
    if trailing_pointer:
        names.add(trailing_pointer.group(1))
    return tuple(sorted(names))


def _local_type_aliases(source):
    """Collect conservative local ``using``/``typedef`` alias definitions.

    Each name maps to all definitions found for it. ``None`` marks syntax this
    lightweight checker cannot resolve safely; a referenced duplicate or
    unresolved definition therefore makes the whole API snapshot unavailable.
    A typedef whose names cannot be identified makes this collection unavailable.
    """
    text = _without_cpp_comments(source)
    aliases = {}

    using_attempts = list(re.finditer(r"(?m)^\s*using\s+([A-Za-z_]\w*)\s*=", text))
    using_defs = list(
        re.finditer(r"(?ms)^\s*using\s+([A-Za-z_]\w*)\s*=\s*([^;{}]+);", text)
    )
    parsed_using_starts = {match.start() for match in using_defs}
    for match in using_defs:
        name, target = match.group(1), _normalize_cpp_surface(match.group(2))
        aliases.setdefault(name, []).append(target or None)
    for match in using_attempts:
        if match.start() not in parsed_using_starts:
            aliases.setdefault(match.group(1), []).append(None)

    typedef_attempts = list(re.finditer(r"(?m)^\s*typedef\b", text))
    typedef_defs = list(re.finditer(r"(?ms)^\s*typedef\s+([^;{}]+);", text))
    parsed_typedef_starts = {match.start() for match in typedef_defs}
    if any(match.start() not in parsed_typedef_starts for match in typedef_attempts):
        return None

    for match in typedef_defs:
        body = _normalize_cpp_surface(match.group(1))
        commas = _top_level_commas(body)
        if commas is None:
            return None
        if commas:
            # Multiple declarators share one base type. Resolving all of them
            # correctly requires a C++ declarator parser, so expose each likely
            # name as unresolved and fail closed only if it reaches a binding.
            starts = [0, *(offset + 1 for offset in commas)]
            ends = [*commas, len(body)]
            pieces = [body[start:end] for start, end in zip(starts, ends, strict=True)]
            for piece in pieces:
                name_match = re.search(r"([A-Za-z_]\w*)\s*$", piece.strip())
                names = set(_unsupported_typedef_names(piece))
                if name_match:
                    names.add(name_match.group(1))
                if not names:
                    return None
                for name in sorted(names):
                    aliases.setdefault(name, []).append(None)
            continue

        simple = re.fullmatch(r"(.+?)\s+([A-Za-z_]\w*)", body)
        if simple:
            target, name = _normalize_cpp_surface(simple.group(1)), simple.group(2)
            aliases.setdefault(name, []).append(target or None)
            continue

        pointer = re.search(r"\(\s*[*&]\s*([A-Za-z_]\w*)\s*\)", body)
        if pointer:
            aliases.setdefault(pointer.group(1), []).append(None)
            continue
        names = _unsupported_typedef_names(body)
        if not names:
            return None
        for name in names:
            aliases.setdefault(name, []).append(None)

    return aliases


def _referenced_type_aliases(source, signatures):
    """Resolve aliases reachable from bound callable signatures.

    Unused private aliases are intentionally omitted when they can be parsed.
    Duplicate definitions, unsupported declarations, or cycles on a reachable
    path return ``None``; an unclassifiable typedef also returns ``None`` because
    its reachability cannot be established safely.
    """
    aliases = _local_type_aliases(source)
    if aliases is None:
        return None
    signature_text = " ".join(signatures)
    roots = sorted(
        name for name in aliases if re.search(rf"\b{re.escape(name)}\b", signature_text)
    )
    state = {}
    resolved = {}

    def visit(name):
        if state.get(name) == "done":
            return True
        if state.get(name) == "visiting":
            return False
        definitions = aliases.get(name, [])
        if len(definitions) != 1 or definitions[0] is None:
            return False

        state[name] = "visiting"
        target = definitions[0]
        dependencies = sorted(
            candidate
            for candidate in aliases
            if re.search(rf"\b{re.escape(candidate)}\b", target)
        )
        if any(not visit(dependency) for dependency in dependencies):
            return False
        state[name] = "done"
        resolved[name] = target
        return True

    if any(not visit(root) for root in roots):
        return None
    return tuple(sorted(resolved.items()))


def binding_api_snapshot(source, op):
    """Return registration, callable signatures, and reachable type aliases.

    Function bodies and private helpers are deliberately excluded. Unsupported
    or ambiguous syntax returns ``None`` so callers can require manual review.
    """
    if not source:
        return None

    export = re.search(
        rf"(?ms)^void\s+ExportOp{re.escape(op)}\s*\(\s*py::module\s*&\s*m\s*\)\s*"
        rf"\{{(?P<body>.*?)^\}}\s*// namespace cvcudapy\s*$",
        source,
    )
    if not export:
        return None

    registration = re.sub(
        r'R"pbdoc\(.*?\)pbdoc"',
        'R"pbdoc()pbdoc"',
        export.group("body"),
        flags=re.S,
    )
    registration_count = len(re.findall(r"\bm\.def\s*\(", registration))
    symbols = re.findall(
        r'\bm\.def\s*\(\s*"[^"]+"\s*,\s*'
        r'(?:&|NvtxTrace\s*\(\s*"[^"]+"\s*,\s*&)'
        r"([A-Za-z_]\w*)",
        registration,
    )
    if registration_count == 0 or len(symbols) != registration_count:
        return None

    callable_signatures = []
    for symbol in symbols:
        matches = list(
            re.finditer(
                rf"(?ms)^(?P<signature>[^\n{{;]*\b{re.escape(symbol)}\s*"
                rf"\([^;{{]*?\))\s*\{{",
                source,
            )
        )
        if len(matches) != 1:
            return None
        callable_signatures.append(
            (symbol, _normalize_cpp_surface(matches[0].group("signature")))
        )

    referenced_aliases = _referenced_type_aliases(
        source, [signature for _, signature in callable_signatures]
    )
    if referenced_aliases is None:
        return None

    return (
        _normalize_cpp_surface(registration),
        tuple(callable_signatures),
        referenced_aliases,
    )
