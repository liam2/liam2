#! /usr/bin/env python
import os
import re

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np

# ============= #
# generic tools #
# ============= #

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


np_include_dir = np.get_include()
extensions = [
    Extension("liam2.cpartition", ["liam2/cpartition.pyx"],
              include_dirs=[np_include_dir]),
    Extension("liam2.cutils", ["liam2/cutils.pyx"],
              include_dirs=[np_include_dir])
]
version = get_version('./liam2/version.py')


classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science / Research",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Healthcare Industry",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Programming Language :: Python :: 3.14',
    "Topic :: Scientific / Engineering",
]

setup(
    name="liam2",
    version=int_version(version),
    author="Gaëtan de Menten",
    author_email="gdementen@gmail.com",
    url="http://liam2.plan.be",
    license='GPL-3.0',
    license_files=['COPYING'],
    description="Microsimulation platform",
    long_description=read_local('README.rst'),
    classifiers=classifiers,
    packages=['liam2'],
    include_package_data=True,
    entry_points={
        'console_scripts': ['liam2=liam2.main:main'],
    },
    install_requires=[
        # not specifying cython here because we need it to be installed
        # *before* this script executes, if we want it to be of any use.
        'numexpr >= 2.6.6',
        'numpy >= 1.8',
        'tables >= 3',
        'pyyaml',
    ],
    extras_require=dict(
        interpolation=['bcolz'],
        plot=['matplotlib'],
        view=['vitables'],
        test=['flake8', 'nose', 'matplotlib'],
    ),
    ext_modules=cythonize(extensions),
)
