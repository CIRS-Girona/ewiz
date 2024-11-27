import os

project = "eWiz"
copyright = "2024, Jad Mansour, University of Girona. All rights reserved"
author = "Jad Mansour"
release = "1.0.0"

extensions = ["sphinx_rtd_theme", "sphinx.ext.autodoc"]
autodoc_member_order = "bysource"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
"""
html_logo = os.path.join("_static", "logo_no_slogan.svg")
"""
html_theme_options = {
    "logo_only": False
}
