New features
------------

* allow loading globals from csv files directly (closes :issue:`30`).

* added a new simulation option category: "logging". This is the new home for
  the existing "timings" option, and for a new "level" option, which allows
  to sets the verbosity of logging to one of `periods`, `procedures` (this is
  the default) or `processes` (which corresponds to the behavior in
  earlier versions).

Miscellaneous improvements
--------------------------
* decreased memory usage. In our models~20%
  flushes output_index/id_to_rownum to disk for periods < period - 1 (closes
  :issue:`130`).

* show top times now also show the % of total.

* after each period, a very rough estimate of the remaining time for the
  simulation is displayed (closes :issue:`127` in combination with the
  logging/level option).

Fixes
-----

* Using links or other "non-simple"
  variables in the score expression
  TODO: of what? matching?
  made things a lot slower than necessary
  because the "system temporary variables" kept accumulating and context_subset
  was increasingly costly.