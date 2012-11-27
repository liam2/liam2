import sys
from distutils.extension import Extension

from cx_Freeze import setup, Executable
from Cython.Distutils import build_ext as cython_build_ext
import numpy as np

# cython options
ext_modules = [Extension("groupby", ["groupby.pyx"],
                         include_dirs=[np.get_include()])]
build_ext_options = {}

# Add the output directory of build_ext to sys.path so that build_exe finds
# and copies groupby.pyd
class my_build_ext(cython_build_ext):
    def finalize_options(self):
        cython_build_ext.finalize_options(self)
        sys.path.insert(0, self.build_lib)

# cx_freeze options
excludes = ["Tkconstants", "Tkinter", "Cython", "_ssl",
            "base64", "bz2", "calendar", "compiler", "ctypes",
            "distutils", "doctest", "dummy_thread",
            "dummy_threading", "email", "ftplib", "io",
            "logging", "multiprocessing", "nose",
            "numpy.distutils", "numpy.core._dotblas",
            "os2emxpath", "pdb", "pkg_resources",
            "posixpath", "pydoc", "pydoc_topics", "repr", "scipy",
            "select", "stringprep", "strptime",
            "tcl", "unicodedata", "xml"]
build_exe_options = {"compressed": True,
                     "optimize": 2,
                     "replace_paths": [("*", "")],
                     "excludes": excludes}

setup(name="liam2",
      version="0.5.1",
      description="LIAM2",

      cmdclass={"build_ext": my_build_ext},
      ext_modules=ext_modules,

      options={"build_ext": build_ext_options,
               "build_exe": build_exe_options},
      executables=[Executable("main.py")])