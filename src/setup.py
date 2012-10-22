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
               # When use versions of numpy and numexpr linked to the MKL
               # (for example those provided by Christoph Gohlke), py2exe is
               # unable to find this dll, so we have to exclude it but than we
               # *must* copy the file from numpy (usually
               # $PYTHONROOT\Lib\site-packages\numpy\) to the "dist" directory.
               dll_excludes=["libiomp5md.dll"]
          )
      )
)
