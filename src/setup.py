from distutils.core import setup

import py2exe


setup(console=['main.py'],
      options=dict(
          py2exe=dict(
              excludes=["Tkconstants", "Tkinter", "_ssl", 
                        "base64", "bz2", "calendar", "compiler", "ctypes",
                        "distutils", "doctest", "dummy_thread", 
                        "dummy_threading", "email", "ftplib", "io", 
                        "locale", "logging", "multiprocessing", "nose",
                        "numpy.distutils", "numpy.core._dotblas",
                        "os2emxpath", "pdb", "pkg_resources",
                        "posixpath", "pydoc", "pydoc_topics", "repr", "scipy",
                        "select", "stringprep", "strptime",   
                        "tcl", "unicodedata", "xml"],
                        # StringIO and difflib are required with python2.7/64b 
          )
      )
)