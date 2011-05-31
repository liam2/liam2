from distutils.core import setup

import py2exe
import sys

args = sys.argv
if len(args) < 2:
    print "Usage: %s script.py" % args[0]
    sys.exit()

script = args[1]
args[1] = 'py2exe'

setup(console=[script],
      options=dict(
          py2exe=dict(
              excludes=["Tkconstants", "Tkinter", "tcl", "_ssl", "pdb", "pydoc",
                        "pydoc_topics", "difflib"],
          )
      )
)