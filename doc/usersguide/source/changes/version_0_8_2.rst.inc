New features

------------

* added the rank_matching() function as an other method to match two sets of 
individuals.

* add the possibility to generate automatically an order in matching. The two
possible new options are 'EDtM' and 'SDtOM'.

* add an new 'pool_size' optional parameter in matching. If an integer is
entered the best match for an individual is looked for in a random subset
of size pool_size.

--------------------------

Fixes

-----
=======
------------

* allow loading globals from csv files directly. One can
  mix both approaches (have some globals in the "input" HDF file while other
  globals in .csv files). This makes it more practical in case you have a
  relatively stable input dataset for entities, but need to change the globals
  often, or have several variants with different globals.
  This closes :issue:`30`.

* added a new simulation option category: "logging". This is the new home for
  the existing "timings" option, and for a new "level" option, which allows
  to sets the verbosity of logging to one of `periods`, `procedures` (this is
  the default) or `processes` (which corresponds to the behavior in
  earlier versions).

Miscellaneous improvements
--------------------------
* decreased memory usage for models with many periods. It used to keep the
  index of individuals of past periods forever in memory, which had
  approximately the same effect than adding one column for each period.
  Since it is only useful when going back in time more than one period, it now
  flushes it to disk for periods < period - 1. This as two consequences: it
  will decrease memory use for models with many periods but will slow down
  models going back in time more than one period. This closes :issue:`130`.

* the "top times" at the end of the simulation now also include the % of total.

* after each period, a very rough estimate of the remaining time for the
  simulation is displayed (closes :issue:`127` in combination with the
  logging/level option).

Fixes
-----

* Using links or other "non-simple" variables in the score expression of the
  matching() function was a lot slower and memory-hungry than necessary
  because some "system temporary variables" kept accumulating. It is still a
  lot slower than it should though, see :issue:`128`.
