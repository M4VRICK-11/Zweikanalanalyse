import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project ='Python & Akustik - Zweikanalanalysator'
copyright = '2025, Omar Ben Salem, Gwendal Frühauf'
author = 'Omar Ben Salem, Gwendal Frühauf'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx_autodoc_typehints',
              'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'de'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_title = "Python & Akustik - Zweikanalanalysator"

def setup(app):
    app.add_css_file('custom.css')

