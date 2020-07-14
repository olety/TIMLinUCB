# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

docs_path = os.path.abspath(".")
code_path = os.path.abspath(os.path.join(docs_path, os.pardir))
libs_path = os.path.abspath(
    os.path.join(code_path, "venv", "lib", "python3.7", "site-packages")
)
sys.path.insert(0, code_path)
sys.path.insert(0, libs_path)


# -- Project information -----------------------------------------------------

project = "TIMLinUCB"
copyright = "2020, Oleksii Kyrylchuk"
author = "Oleksii Kyrylchuk"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # Numpy style docstrings
    "sphinx_rtd_theme",  # ReadTheDocs theme
]
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "experiments.py",
    "experiments_digg_comparison.py",
    "experiments_facebook_comparison.py",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# Making the theme compatible with RTD
# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
# on_rtd = os.environ.get("READTHEDOCS", None) == "True"
#
# if not on_rtd:  # only import and set the theme if we're building docs locally
#     import sphinx_theme
#
#     html_theme = "stanford_theme"
#     html_theme_path = [sphinx_theme.get_html_theme_path("stanford_theme")]

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
