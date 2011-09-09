from distutils.core import setup

import py2exe


setup(console=['main.py'],
      options=dict(
          py2exe=dict(
              excludes=["Tkconstants", "Tkinter", "tcl", "_ssl", "pdb", "pydoc",
                        "pydoc_topics"],
                        # difflib is required with python2.7/64b 
          )
      )
)