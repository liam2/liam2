import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np


sys.argv.extend(['build_ext', '--inplace'])

ext_modules = [Extension("groupby", ["groupby.pyx"],
                         include_dirs=[np.get_include()])]

setup(
    name='Groupby',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
