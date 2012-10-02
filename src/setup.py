from distutils.core import setup

import py2exe


setup(console=['main.py'],
      options=dict(
          py2exe=dict(
              excludes=["Tkconstants", "Tkinter", "_ssl",
                        "base64", "bz2", "calendar", "compiler", "ctypes",
                        "distutils", "doctest", "dummy_thread",
                        "dummy_threading", "email", "ftplib", "io",
                        "logging", "multiprocessing", "nose",
                        "numpy.distutils", "numpy.core._dotblas",
                        "os2emxpath", "pdb", "pkg_resources",
                        "posixpath", "pydoc", "pydoc_topics", "repr", "scipy",
                        "select", "stringprep", "strptime",
                        "tcl", "unicodedata", "xml"],
                        # locale, StringIO and difflib are required with
                        # python2.7/64b
               # py2exe seems to be unable to find this dll, so we exclude it
               # but than we *must* copy the file from numpy (usually
               # $PYTHONROOT\Lib\site-packages\numpy\) to the "dist" directory.
               # XXX: this might be because I installed numexpr from the
               # binaries from Christoph Gohlke which depend on Numpy-MKL
               # (http://www.lfd.uci.edu/~gohlke/pythonlibs/)
               dll_excludes=["libiomp5md.dll"]
          )
      )
)
