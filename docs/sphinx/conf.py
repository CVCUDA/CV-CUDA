# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "CV-CUDA"
copyright = "2022-2024, NVIDIA."
author = "NVIDIA"
version = "Beta"
release = version

# set python docstring source path
lib_path = os.getenv("SPHINX_PYTHON_SRC", default=".")
sys.path.insert(0, os.path.abspath(lib_path))

# -- Module mocking ----------------------------------------------------------

autodoc_mock_imports = ['nvcv', 'cvcuda']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "docs/manuals/py/**"]

extensions = ["recommonmark"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"

autodoc_inherit_docstrings = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "sphinx_rtd_theme"
html_logo = os.path.join("content", "nv_logo.png")

html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#000000",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": False,
    # 'navigation_depth': 10,
    "sidebarwidth": 12,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_favicon = os.path.join("content", "nv_icon.png")

html_static_path = ["templates"]

html_last_updated_fmt = ""

html_js_files = [
    "pk_scripts.js",
]


def setup(app):
    app.add_css_file("custom.css")


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for sphinx_rtd_theme -----------------------------------------

# Enable the sphinx_rtd_theme extension
extensions.append("sphinx_rtd_theme")

# Enable the sphinx.ext.todo extension
extensions.append("sphinx.ext.todo")

# -- Extensions --------------------------------------------------

# Enable extensions
extensions.append("breathe")
extensions.append("sphinx.ext.autodoc")
extensions.append("sphinx.ext.viewcode")
extensions.append("sphinx.ext.napoleon")

# -- Extension configuration -------------------------------------------------
# Set up the default project for breathe extension
breathe_default_project = "cvcuda"
breathe_doxygen_config_options = {
    "QUIET": "NO",
    "WARNINGS": "NO",
    "WARN_IF_UNDOCUMENTED": "NO",
    "WARN_IF_DOC_ERROR": "NO",
    "WARN_NO_PARAMDOC": "NO",
    "WARN_AS_ERROR": "NO",
}
