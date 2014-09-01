New features
------------

* allow loading globals from csv files directly (closes :issue:`30`).

* added command line argument (-p/--progress) to enable reporting of
  global/simulation progress, it does not play well with (interactive) charts
  though (closes :issue:`127`).

Miscellaneous improvements
--------------------------
* decreased memory usage. In our models~20%
  flushes output_index/id_to_rownum to disk for periods < period - 1 (closes
  :issue:`130`).

* show top times now also show the % of total.

Fixes
-----

* Using links or other "non-simple"
  variables in the score expression
  TODO: of what? matching?
  made things a lot slower than necessary
  because the "system temporary variables" kept accumulating and context_subset
  was increasingly costly.