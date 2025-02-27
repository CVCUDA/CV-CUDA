# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import re
import sys
from typing import List, Tuple

LICENSES = {
    "Apache-2.0": """  # SPDX-License-Identifier: Apache-2.0
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
  # limitations under the License.""",
}

# Define the regular expression for the comment line


def get_tag(file_path, tag="SPDX-License-Identifier"):
    # Define the regular expression to match the tag and extract its value
    rgx_comment = r'^[\s]*[\w\W]+[\s]+[\w\W]*[\s]*["]*'
    rgx = rf"^(?:{rgx_comment})?{tag}:\s*([^\"]*)\"*"

    # Read the file content
    with open(file_path, "r") as file:
        content = file.readlines()

    # Search for the tag in each line and extract the value
    for line in content:
        match = re.search(rgx, line)
        if match:
            return match.group(1).strip()

    return None


def exports_enum(s: str) -> bool:
    return s.lstrip().startswith("py::enum_<")


def get_name_of_enum(s: str) -> str:
    """Name of enum is first string in line"""
    return re.findall('"([^"]*)"', s)[0]


def exports_class(s: str) -> bool:
    return s.lstrip().startswith("py::class_<")


def get_name_of_class_if_documented(s: str) -> Tuple[bool, str]:
    """
    If a class has only one strings in line, it has no documentation to be exported.
    If it has more than one string, it has doc and first string is the title of the class
    """
    found_strings = re.findall('"([^"]*)"', s)  # get all strings
    if len(found_strings) > 1:
        return True, found_strings[0]
    else:
        return False, ""


def exports_def(s: str) -> bool:
    return s.lstrip().startswith("m.def(")


def get_name_of_def(s: str) -> str:
    """Name of def is first string in line"""
    return re.findall('"([^"]*)"', s)[0]


def has_exports(file_path: str, export_calls: List[str]) -> bool:
    with open(file_path, "r") as file_str:
        file_str_read = file_str.read()
        for call in export_calls:
            if call in file_str_read:
                export_calls.remove(call)
                return True
    return False


def create_rst_text(
    template_file: str, name: str, module: str, members: str, license: str
) -> str:
    with open(template_file, "r") as f:
        rst_text = f.read()
    rst_text = rst_text.replace("@OperatorName@", name)
    rst_text = rst_text.replace("@=@", "=" * len(name))
    rst_text = rst_text.replace("@Module@", module)
    rst_text = rst_text.replace("@MemberFunctions@", members)
    rst_text = rst_text.replace(
        LICENSES["Apache-2.0"], license
    )  # template_file has Apache-2.0 by default
    return rst_text


def create_cvcuda_operator_rst_files(
    cvcuda_path: str, outdir: str, python_cvcuda_root: str, export_calls: List[str]
) -> None:
    # search for template rst file
    template_rst_file_path = os.path.join(
        cvcuda_path, "docs", "sphinx", "_python_api", "template.rst"
    )
    if not os.path.isfile(template_rst_file_path):
        raise FileNotFoundError(f"File {template_rst_file_path} not found")

    # iterate through all files
    for i in sorted(os.listdir(python_cvcuda_root)):
        op_file_path = os.path.join(python_cvcuda_root, i)
        # Only work on .cpp files that export operators
        if (
            os.path.isfile(op_file_path)
            and i.endswith(".cpp")
            and i != "Main.cpp"
            and has_exports(op_file_path, export_calls)
        ):
            # Get operators license
            op_license_tag = get_tag(op_file_path)
            if not op_license_tag:
                raise RuntimeError(f"No license tag found for file: {op_file_path}")
            op_license = LICENSES[op_license_tag]

            # Get operator name form .cpp file: remove prefix "Op" and file type
            operator_name = os.path.splitext(i)[0]
            operator_name = operator_name[len("Op") :]  # noqa: E203

            # Look for functions to add to documentation
            # search for all lines that start with "m.def(" (stripping leading white spaces)
            # then pick first string of that line, this is the name of the python function to be exported
            exports = set()
            with open(op_file_path, "r") as fp:
                for line in fp:
                    if exports_def(line):
                        exports.add(get_name_of_def(line))
            if len(exports) == 0:
                raise RuntimeError(f"No exports found in file {op_file_path}")
            exports_str = ", ".join(exports)

            # Create text to put into rst file - starting from a template
            rst_text = create_rst_text(
                template_rst_file_path, operator_name, "cvcuda", exports_str, op_license
            )

            # Write rst file: outdir/_op_<operatorname>.rst
            outfile = os.path.join(outdir, f"_op_{operator_name.lower()}.rst")
            with open(outfile, "w") as f:
                f.write(rst_text)
    return


def create_cvcuda_non_operator_rst_files(
    cvcuda_path: str, outdir: str, python_cvcuda_root: str, export_calls: List[str]
) -> None:
    # search for template rst file
    template_rst_file_path = os.path.join(
        cvcuda_path, "docs", "sphinx", "_python_api", "template.rst"
    )
    if not os.path.isfile(template_rst_file_path):
        raise FileNotFoundError(f"File {template_rst_file_path} not found")

    for i in sorted(os.listdir(python_cvcuda_root)):
        nonop_file_path = os.path.join(python_cvcuda_root, i)
        # Only work on .cpp files that something different than operators
        if (
            os.path.isfile(nonop_file_path)
            and i.endswith(".cpp")
            and i != "Main.cpp"
            and has_exports(nonop_file_path, export_calls)
        ):
            # Get non-operators license
            nonop_license_tag = get_tag(nonop_file_path)
            if not nonop_license_tag:
                raise RuntimeError(f"No license tag found for file: {nonop_file_path}")
            nonop_license = LICENSES[nonop_license_tag]

            # Look for functions to add to documentation
            # Search for all lines that start with "py::enum_<" or "py::class_<"
            with open(nonop_file_path, "r") as fp:
                for line in fp:
                    if exports_enum(line):
                        export = get_name_of_enum(line)
                    elif exports_class(line):
                        has_doc, name = get_name_of_class_if_documented(line)
                        if has_doc:
                            export = name
                        else:
                            continue
                    else:
                        continue

                    # Create text to put into rst file - starting from a template
                    rst_text = create_rst_text(
                        template_rst_file_path, export, "cvcuda", export, nonop_license
                    )

                    # Write rst file: outdir/_aux_<export>.rst
                    outfile = os.path.join(outdir, f"_aux_{export.lower()}.rst")
                    with open(outfile, "w") as f:
                        f.write(rst_text)
    return


def export_found(s: str) -> bool:
    return s.lstrip().startswith("Export")


def get_export_fun_name(s: str) -> str:
    return s.lstrip().split("(", 1)[0]


def exporting_nonops(s: str) -> bool:
    """Everything after that command exports auxiliary operator entities
    (non-operators)"""
    return s.lstrip().startswith("// doctag: Non-Operators")


def exporting_ops(s: str) -> bool:
    """Everything after that command exports operators"""
    return s.lstrip().startswith("// doctag: Operators")


def get_exported_cvcuda(path_to_main: str):
    export_nonop = []  # list for non operators
    export_op = []  # list for operators
    exports = None
    with open(path_to_main, "r") as fp:
        for line in fp:
            if export_found(line):
                # remove everything after first "("
                name = get_export_fun_name(line)
                try:
                    exports.append(name)
                except AttributeError:
                    print(
                        "No comment '// doctag: Non-Operators' or '// doctag: Operators' was found in "
                        f"{path_to_main} prior to 'Export*(m);'-routines."
                    )
                    sys.exit()
            elif exporting_nonops(line):
                exports = export_nonop
            elif exporting_ops(line):
                exports = export_op
    assert len(export_nonop) > 0 and len(export_op) > 0
    return export_nonop, export_op


def generate_py_doc_rsts_cvcuda(cvcuda_path: str, outdir: str):
    python_cvcuda_root = os.path.join(cvcuda_path, "python", "mod_cvcuda")
    export_nonop, export_op = get_exported_cvcuda(
        os.path.join(python_cvcuda_root, "Main.cpp")
    )
    create_cvcuda_operator_rst_files(cvcuda_path, outdir, python_cvcuda_root, export_op)
    create_cvcuda_non_operator_rst_files(
        cvcuda_path, outdir, python_cvcuda_root, export_nonop
    )
    return


if __name__ == "__main__":
    outdir = sys.argv[1]  # path/to/cvcuda/docs/sphinx/_python_api/_cvcuda_api
    cvcuda_path = sys.argv[2]  # path/to/cvcuda
    os.makedirs(outdir, exist_ok=True)
    generate_py_doc_rsts_cvcuda(cvcuda_path, outdir)
