# -*- coding: utf-8 -*-
from __future__ import annotations

master_doc = "index"
project = "cvxcovariance"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "recommonmark",
]


source_dirs = "cvx"
autoapi_type = "python"
autoapi_dirs = ["../../%s" % d for d in source_dirs.split()]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# HTML -----------------------------------------------------------------
html_favicon = "_static/logo.jpeg"
html_logo = "_static/logo.jpeg"
html_title = "cvxcovariance Documentation"
html_show_sourcelink = True
html_css_files = [
    "custom.css",
]
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
