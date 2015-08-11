#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import platform
import sys
import fnmatch
from os.path import join
from itertools import chain


from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

try:
    from cx_Freeze import Executable, setup
except ImportError:
    Executable = None


# ============= #
# generic tools #
# ============= #

def allfiles(pattern, path='.'):
    """
    like glob.glob(pattern) but also include files in subdirectories
    """
    return (join(dirpath, f)
            for dirpath, dirnames, files in os.walk(path)
            for f in fnmatch.filter(files, pattern))


def int_version(release_name):
    """
    converts a release name to a version string with only dots and integers
    :param release_name: the release name to convert
    :return: a release name with prerelease tags (beta, rc, ...) stripped.
    unrecognised tags are left intact, even if that means returning an invalid
    version string
    >>> int_version('0.8')
    '0.8'
    >>> int_version('0.8.1')
    '0.8.1'
    >>> int_version('0.8alpha1')
    '0.7.99701'
    >>> int_version('0.8rc2')
    '0.7.99902'
    >>> int_version('0.8.1a2')
    '0.8.0.99702'
    >>> int_version('0.8.1beta3')
    '0.8.0.99803'
    >>> int_version('0.8.1rc1')
    '0.8.0.99901'
    """
    if 'pre' in release_name:
        raise ValueError("'pre' is not supported anymore, use 'alpha' or "
                         "'beta' instead")
    if '-' in release_name:
        raise ValueError("- is not supported anymore")
    # 'a' needs to be searched for after 'beta'
    tags = [('rc', 9), ('c', 9),
            ('beta', 8), ('b', 8),
            ('alpha', 7), ('a', 7)]
    for tag, num in tags:
        pos = release_name.find(tag)
        if pos != -1:
            head, tail = release_name[:pos], release_name[pos + len(tag):]
            assert tail.isdigit()
            patch = '.99' + str(num) + tail.rjust(2, '0')
            head, middle = head.rsplit('.', 1)
            return head + '.' + str(int(middle) - 1) + patch
    return release_name


# ============ #
# main options #
# ============ #

optional_kwargs = {}


# ============== #
# cython options #
# ============== #

# Add the output directory of cython build_ext to cxfreeze search path so that
# build_exe finds and copies C extensions
class MyBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)

        if Executable is not None:
            # need to be done in-place, otherwise build_exe_options['path'] will use
            # the unmodified version because it is computed before build_ext is
            # called
            cxfreeze_searchpath.insert(0, self.build_lib)


ext_modules = [Extension("cpartition", ["liam2/cpartition.pyx"],
                         include_dirs=[np.get_include()]),
               Extension("cutils", ["liam2/cutils.pyx"],
                         include_dirs=[np.get_include()])]
build_ext_options = {}


# ================= #
# cx_freeze options #
# ================= #

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

if Executable is not None:
    cxfreeze_searchpath = sys.path + ['liam2']

    build_exe_options = {
        # path to find Python modules (we could have modified sys.path but this
        # is a bit cleaner)
        "path": cxfreeze_searchpath,

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
            "Tkconstants", "Tkinter", "scipy", "tcl"
        ],
        'include_files': vitables_data_files(),
    }

    if Executable is not None:
        optional_kwargs['executables'] = [Executable("liam2/main.py")]
else:
    build_exe_options = {}


# ========== #
# main stuff #
# ========== #

setup(
    name="liam2",
    # cx_freeze wants only ints and dots
    version=int_version('0.10.2'),
    description="LIAM2",
    cmdclass={"build_ext": MyBuildExt},
    ext_modules=ext_modules,
    options={"build_ext": build_ext_options, "build_exe": build_exe_options},
    install_requires=[
        'numexpr',
        'numpy',
        'tables >= 3',
    ],
    extras_require=dict(
        interpolation=['bcolz'],
        plot=['matplotlib'],
        view=['vitables'],
        ),
    **optional_kwargs
)
