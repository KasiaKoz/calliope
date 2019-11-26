# -*- coding: utf-8 -*-
#
# Calliope documentation build configuration file, created by
# sphinx-quickstart on Thu Nov 14 09:10:03 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os

# Append the local _extensions dir to module search path
sys.path.append(os.path.abspath('_extensions'))
sys.path.append(os.path.abspath('helpers'))

from sphinx.builders.html import StandaloneHTMLBuilder, SingleFileHTMLBuilder

import generate_tables  # from helpers

__version__ = None
# Sets the __version__ variable
exec(open('../calliope/_version.py').read())

# Generates the tables and source code files
generate_tables.process()

##
# Mock modules for Read The Docs autodoc generation
##

class Mock(object):
    __all__ = []

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        elif name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        else:
            return Mock()

MOCK_MODULES = [
    'numpy', 'matplotlib', 'matplotlib.pyplot',
    'matplotlib.colors', 'matplotlib.colors.ListedColormap',
    'pyomo', 'pyomo.core', 'pyomo.opt', 'pyomo.environ',
    'pyutilib', 'pyutilib.services',
    'pyutilib.services.TempfileManager', 'yaml', 'pandas',
    'click', 'xarray', 'dask', 'xarray.ufuncs',
    'numpy.random', 'numpy.fft', 'numpy.lib', 'numpy.lib.scimath',
    'scipy', 'scipy.cluster', 'scipy.cluster.vq',
    'scipy.spatial', 'scipy.spatial.distance',
    'sklearn', 'sklearn.metrics',
    'bokeh', 'bokeh.plotting', 'bokeh.models', 'bokeh.core.properties',
    'plotly', 'plotly.offline', 'plotly.graph_objs',
    'natsort', 'IPython'
]

for m in MOCK_MODULES:
    sys.modules[m] = Mock()


# Redefine supported_image_types for the HTML builder to prefer PNG over SVG
image_types = ['image/png', 'image/svg+xml', 'image/gif', 'image/jpeg']
StandaloneHTMLBuilder.supported_image_types = image_types
SingleFileHTMLBuilder.supported_image_types = image_types

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

# Generate RTD base url: if a dev version, point to "latest", else "v..."
if 'dev' in __version__:
    docs_base_url = 'en/latest/'
else:
    docs_base_url = 'en/v{}/'.format(__version__)

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode',
              'sphinx.ext.extlinks',
              # 'numfig',
              # 'sphinx.ext.autosummary',
              'numpydoc']

# Ensure that cdnjs is used rather than the discontinued mathjax cdn
mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

numpydoc_show_class_members = False   # numpydoc: don't do autosummary

nbviewer_url = 'https://nbviewer.ipython.org/url/calliope.readthedocs.io/'

extlinks = {'nbviewer_docs': (nbviewer_url + docs_base_url + '%s', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

#A string of reStructuredText that will be included at the beginning of every
# source file that is read
rst_prolog = """
.. role:: python(code)
   :language: python

.. role:: yaml(code)
   :language: yaml

.. role:: sh(code)
   :language: sh
"""

# A string of reStructuredText that will be included at the end
# of every source file that is read. This is the right place to add
# substitutions that should be available in every file.
# rst_epilog = """
# .. |docs_base_url| replace:: {}
# """.format(docs_base_url)

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Calliope'
copyright = '2013–2018 Calliope contributors listed in AUTHORS (Apache 2.0 licensed)'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'flask_theme_support.FlaskyStyle'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
sys.path.append(os.path.abspath('_themes'))
html_theme_path = ['_themes']
html_theme = 'flask_calliope'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    'index': ['sidebar_title.html', 'sidebar_search.html', 'sidebar_downloads.html', 'sidebar_toc.html'],
    '**': ['sidebar_title.html', 'sidebar_search.html', 'sidebar_toc.html']
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'Calliopedoc'


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'Calliope.tex', 'Calliope Documentation',
     'Calliope contributors', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'calliope', 'Calliope Documentation',
     ['Calliope contributors'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'Calliope', 'Calliope Documentation',
     'Calliope contributors', 'Calliope', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'


#
# Remove module docstrings from autodoc
#

def remove_module_docstring(app, what, name, obj, options, lines):
    if what == "module":
        del lines[:]


autodoc_member_order = "bysource"


def setup(app):
    app.connect("autodoc-process-docstring", remove_module_docstring)
