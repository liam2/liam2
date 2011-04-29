from distutils.core import setup

import py2exe


setup(console=['merge_h5.py'],
      options=dict(
          py2exe=dict(
              excludes=["Tkconstants", "Tkinter", "tcl", "_ssl", "pdb", "pydoc",
                        "pydoc_topics", "difflib"],
          )
      )
)