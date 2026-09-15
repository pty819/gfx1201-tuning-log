# Sphinx config for gfx1201 调优实录.
# Theme: sphinx_rtd_theme — left toctree, right body.

project = "gfx1201 调优实录"
author = "pty819"
copyright = "2026"
release = "2026-09-15"
language = "zh_CN"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_block",
    "attrs_inline",
]
myst_heading_anchors = 3
autosectionlabel_prefix_document = True

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "titles_only": False,
    "style_external_links": True,
}
html_title = "gfx1201 调优实录"
html_short_title = "gfx1201"
html_show_sourcelink = True
html_copy_source = False
html_static_path = ["_static"]

templates_path = ["_templates"]
pygments_style = "sphinx"
