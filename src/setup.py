from __future__ import print_function

import os
import sys
import fnmatch
from os.path import join
from itertools import chain
from distutils.extension import Extension

from cx_Freeze import setup, Executable
from Cython.Distutils import build_ext
import numpy as np


#===============#
# generic tools #
#===============#

def allfiles(pattern, path='.'):
    """
    like glob.glob(pattern) but also include files in subdirectories
    """
    return (join(dirpath, f)
            for dirpath, dirnames, files in os.walk(path)
            for f in fnmatch.filter(files, pattern))


def int_version(release_name):
    """
    converts a release name to a version number with only dots and integers
    :param release_name: the release to convert
    :return: a release name with -pre and -rc parts stripped

    >>> int_version('0.8')
    '0.8'
    >>> int_version('0.8.1')
    '0.8.1'
    >>> int_version('0.8-pre1')
    '0.7.99801'
    >>> int_version('0.8-rc2')
    '0.7.99902'
    >>> int_version('0.8.1-pre2')
    '0.8.0.99802'
    >>> int_version('0.8.1-rc1')
    '0.8.0.99901'
    """
    pre_pos = release_name.find('-pre')
    rc_pos = release_name.find('-rc')
    if pre_pos != -1:
        head, tail = release_name[:pre_pos], release_name[pre_pos + 4:]
        prefix = '8'
    elif rc_pos != -1:
        head, tail = release_name[:rc_pos], release_name[rc_pos + 3:]
        prefix = '9'
    else:
        return release_name
    assert tail.isdigit()
    patch = '.99' + prefix + tail.rjust(2, '0')
    head, middle = head.rsplit('.', 1)
    return head + '.' + str(int(middle) - 1) + patch


#================#
# cython options #
#================#

# Add the output directory of cython build_ext to sys.path so that build_exe
# finds and copies C extensions
class MyBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        sys.path.insert(0, self.build_lib)


ext_modules = [Extension("cpartition", ["cpartition.pyx"],
                         include_dirs=[np.get_include()]),
               Extension("cutils", ["cutils.pyx"],
                         include_dirs=[np.get_include()])]
build_ext_options = {}


#===================#
# cx_freeze options #
#===================#

def vitables_data_files():
    try:
        import vitables
    except ImportError:
        return []

    module_path = os.path.dirname(vitables.__file__)
    files = chain(allfiles('*.ui', module_path),
                  allfiles('*.ini', module_path),
                  allfiles('*', join(module_path, 'icons')))
    return [(fname, join('vitables', fname[len(module_path) + 1:]))
            for fname in files]

build_exe_options = {
    # compress zip archive
    "compressed": True,

    # optimize pyc files (strip docstrings and asserts)
    "optimize": 2,

    # strip paths in __file__ attributes
    "replace_paths": [("*", "")],

    "includes": ["matplotlib.backends.backend_qt4agg"],
    "packages": ["vitables.plugins"],
    # matplotlib => calendar, distutils, unicodedata
    # matplotlib.backends.backend_tkagg => Tkconstants, Tkinter
    # Qt .ui file loading (for PyTables) => logging, xml
    # ctypes, io are required now
    "excludes": [
        # linux-specific modules
        "_codecs", "_codecs_cn", "_codecs_hk", "_codecs_iso2022", "_codecs_jp",
        "_codecs_kr", "_codecs_tw",

        # common modules
        "Tkconstants", "Tkinter", "Cython", "_ssl", "base64", "bz2", "compiler",
        "doctest", "dummy_thread", "dummy_threading", "email", "ftplib",
        "multiprocessing", "nose", "numpy.distutils",
        "numpy.core._dotblas", "os2emxpath", "pdb", "pkg_resources",
        "posixpath", "pydoc", "pydoc_topics", "repr", "scipy", "select",
        "stringprep", "strptime", "tcl"
    ],
    'include_files': vitables_data_files(),
}


#============#
# main stuff #
#============#

setup(name="liam2",
      version=int_version('0.8.2-rc1'),  # cx_freeze wants only ints and dots
      description="LIAM2",
      cmdclass={"build_ext": MyBuildExt},
      ext_modules=ext_modules,
      options={"build_ext": build_ext_options, "build_exe": build_exe_options},
      executables=[Executable("main.py")],
      requires=['numpy', 'numexpr', 'tables', 'bcolz'])
# also recommends 'matplotlib' and 'vitables'
