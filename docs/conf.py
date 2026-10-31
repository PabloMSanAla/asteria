# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path

# Insert package root to sys.path
package_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(package_path))

import asteria

project = "Asteria"
copyright = "2026, Pablo M. Sanchez-Alarcon"
author = "Pablo M. Sanchez-Alarcon"
release = asteria.__version__
version = asteria.__version__

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"Asteria v{version} Documentation"

# Napoleon settings for Google / NumPy docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
