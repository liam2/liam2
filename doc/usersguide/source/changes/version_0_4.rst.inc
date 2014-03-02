New features
------------

* added grpgini function.

* added grpmedian function.

* implemented filter argument in grpsum().

* implemented N-dimensional alignment (alignment can be done on more than two
  variables/dimensions in the same file).

* added keyword arguments to csv():

  - 'fname' to allow defining the exact name of the csv file. 
  - 'mode' to allow appending to a csv file instead of overwriting it.

* reworked csv() function to support several arguments, like show. It also 
  supports non-table arguments.

* added 'skip_shows' simulation option, to make all show() functions do nothing.

* allowed expressions in addition to variable names in alignment files.

* added keyword arguments to dump():

  - 'missing' to convert nans into the given value.
  - 'header' to determine whether column names should be in the dump or not.

* improved import functionality:

  - compression is now configurable.
  - any csv file can be transposed, not just globals.
  - globals fields can be selected, renamed and inverted like in normal
    entities.
    
* added "explore" command to the main executable, to launch the interactive
  console on a completed simulation without re-simulating it.     

Miscellaneous improvements
--------------------------

* expressions do not need to be quoted anymore.

* reverted init to old semantic: it happens in "start_period - 1", so that 
  lag(variable_set_in_init) works even for the first period.

* purge all local variables after each process to lower memory usage.

* allowed the result of new() to not be stored in a variable.

* allowed using temporary variables in matching() function.

* using a string for matching expressions is deprecated.

* added a tolerance of 1e-6 to the sum of choice's probabilities to be equal 1.0

* added explicit message about alignment over and underflows.

* nicer display for small (< 5ms) and large (>= 1 hour) timings.

* improved error message on missing parenthesis around operands of boolean
  operators.

* improved error message on duplicate fields.

* improved error message when a variable which is not computed yet is used.

* added more information to the console log:

  - number of individuals at the start and end of each period.
  - more stats at the end of the simulation.

* excluded unused components in the executable to make it smaller.

Fixes
-----

* fixed logit_regr(align=float).

* fixed grpavg(bool, filter=cond).

* fixed groupby(a, b, c, expr=grpsum(d), percent=True).

* fixed having several grpavg with a filter argument in the same expression.

* fixed calling the main executable without argument (simply display usage).

* fixed dump with (some kind of) aggregate values in combination with a filter.

* fixed void data source.
