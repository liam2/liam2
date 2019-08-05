#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import re
import sys
import fnmatch
from os.path import join
from itertools import chain

from setuptools import setup
from setuptools.extension import Extension

# not using the try-except here as cython is not really optional now
# if we ever make it optional again, we should uncomment it
# try:
from Cython.Distutils import build_ext
# except ImportError:
#     build_ext = None
import numpy as np


command = sys.argv[1] if len(sys.argv) > 1 else None
build_exe = command == 'build_exe'

if build_exe:
    # when building executables, we need to use "build" so that C extensions are
    # built too ("build" does both "build_ext" and "build_exe"). We do not use
    # build in all cases because we do not want to change the normal "build"
    # behavior when not building executables.
    sys.argv[1] = "build"
    from cx_Freeze import Executable, setup


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
    >>> int_version('0.12.0a1')
    '0.11.99701'
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
            head = head.rstrip('.0')
            patch = '.99' + str(num) + tail.rjust(2, '0')
            head, middle = head.rsplit('.', 1)
            return head + '.' + str(int(middle) - 1) + patch
    return release_name


def read_local(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


# ============ #
# main options #
# ============ #

options = {}
extra_kwargs = {}


# ============== #
# cython options #
# ============== #

# Add the output directory of cython build_ext to cxfreeze search path so that
# build_exe finds and copies C extensions
# XXX: this is always the case currently
if build_ext is not None:
    class MyBuildExt(build_ext):
        def finalize_options(self):
            build_ext.finalize_options(self)

            if build_exe:
                # need to be done in-place, otherwise build_exe_options['path'] will use
                # the unmodified version because it is computed before build_ext is
                # called
                cxfreeze_searchpath.insert(0, self.build_lib)
    extra_kwargs['cmdclass'] = {"build_ext": MyBuildExt}

    ext_modules = [Extension("liam2.cpartition", ["liam2/cpartition.pyx"],
                             include_dirs=[np.get_include()]),
                   Extension("liam2.cutils", ["liam2/cutils.pyx"],
                             include_dirs=[np.get_include()])]
    extra_kwargs['ext_modules'] = ext_modules
    options["build_ext"] = {}


# ================= #
# cx_freeze options #
# ================= #

if build_exe:
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

        "includes": ["matplotlib.backends.backend_qt4agg", "matplotlib.backends.backend_qt5agg"],
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

    options["build_exe"] = build_exe_options
    extra_kwargs['executables'] = [Executable("liam2/main.py")]


# ========== #
# main stuff #
# ========== #

def get_version(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            m = re.match(r'__version__ = "([^"]+)"\s*', line)
            if m:
                return m.group(1)
        return None


version = get_version('./liam2/version.py')


classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science / Research",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Healthcare Industry",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific / Engineering",
]

setup(
    name="liam2",
    # cx_freeze wants only ints and dots (full version number)
    version=int_version(version),
    author="Gaëtan de Menten",
    author_email="gdementen@gmail.com",
    url="http://liam2.plan.be",
    license='GNU General Public License v3 (GPLv3)',
    description="Microsimulation platform",
    long_description=read_local('README.rst'),
    classifiers=classifiers,
    options=options,
    packages=['liam2'],
    include_package_data=True,
    entry_points={
        'console_scripts': ['liam2=liam2.main:main'],
    },
    install_requires=[
        # not specifying cython here because we need it to be installed
        # *before* this script executes, if we want it to be of any use.

        # we fix a precise version of larray because we monkey-patch a private method of it and this is probably
        # going to be more brittle than usual. See larray_monkey.py for more details.
        'larray == 0.32.*',
        'numexpr >= 2.6.6',
        'numpy >= 1.8',
        'tables >= 3',
        'pandas',
        'pyyaml',
    ],
    extras_require=dict(
        interpolation=['bcolz'],
        plot=['matplotlib'],
        view=['vitables'],
        test=['flake8','nose','matplotlib'],
    ),
    **extra_kwargs
)
