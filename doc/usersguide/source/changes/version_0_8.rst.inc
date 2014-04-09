New features
------------

* added a few functions to create charts (courtesy of `matplotlib
  <http://matplotlib.org>`_): bar, plot, pie, stackplot, boxplot and scatter.
  As with all other functions in liam2, they are available both during a
  simulation and in the interactive console. The charts can either be
  visualized directly or saved to a file. See the :ref:`charts section <charts>`
  for details.

* added a "view" command line option to LIAM2 to open ViTables (an hdf5
  viewer) as well as a corresponding menu entry and keyboard shortcut (F9) in
  Notepad++. It is meant to be used when editing a *model* file, and it will
  open both the input dataset and the result file (if any).

* document boolean aggregate functions: all() and any() which were
  added in 0.7 but were not yet documented.

* added *assertFalse* assert function.

Miscellaneous improvements
--------------------------

* improved the first few demonstration models quite a bit. A recommended read
  for all users.

* added precisions in the documentation of align() based on Alexis Eidelman
  suggestions.

* made a few more error messages a bit more useful by displaying the line
  where the error occurred.

* sped up global[array_expr].

* give a hint to use assertNanEqual when it would make a failing assertEqual
  pass.

* implemented global[slice_expr] (eg. MINR[period: period+2]). When the
  slice bounds are arrays (different for each individual) and the slice
  length is not constant (not the same for all individals),
  it returns a special array with an extremely limited set of supported
  operations: only aggregates on axis=1 are implemented.

* include the documentation only in HTML Help format (.chm) in the bundle.
  .pdf and "normal" html are still available as separate downloads on the
  website.

* removed the predictor keyword support (it now raises an exception instead
  of a warning).

* adapted the release script since our move to git and converted it to Python.

Fixes
-----

* fixed the "syntax upgrade" script by removing a special case for grpmin and
  grpmax as it was in fact not needed and caused problems when the expression
  being aggregated contained parentheses.