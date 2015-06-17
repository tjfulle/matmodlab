
Developer Guide
###############

*matmodlab* Directory Structure
===============================

The *matmodlab* project has the following directory structure

::

  matmodlab/
    README.md
    __init__.py
    bin/
    core/
    docs/
    examples/
    lib/
    materials/
    matmodlab.py
    tests/
    utils/
    viz/

Coding Style Guide
==================

All new code in *matmodlab* should adhere sctrictly to the style described in
Python's `PEP-0008 <http://www.python.org/dev/peps/pep-0008/>`_ and docstrings
should follow NumPy's `docstring guide
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

It is highly recommended to use tools such as *pylint* and *pep8* to perform
code analysis on all new and existing code.
