from __future__ import print_function

from distutils.core import setup

import py2exe
import sys

args = sys.argv
if len(args) < 2:
    print("Usage: %s script.py" % args[0])
    sys.exit()

script = args[1]
args[1] = 'py2exe'

setup(console=[script],
      #zipfile=None,      # uncomment to have all files within the executable
      options=dict(
          py2exe=dict(
              excludes=["Tkconstants", "Tkinter", "tcl", "_ssl", "pdb",
                        "pydoc", "pydoc_topics"],
              # 1 = bundle everything, including the Python interpreter
              # 2 = bundle everything but the Python interpreter
              # 3 = don't bundle (default)
              bundle_files=3,
          )
      )
)
