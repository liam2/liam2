.. highlight:: yaml

Change log
##########

Version 0.2.1
=============

Released on 2011-06-16.

Miscellaneous improvements:
---------------------------

* simplified and cleaned up the demonstration models.

* improved the error message when a link points to an unknown entity. 

Fixes:  
------

* fixed "new" function, which created individuals correctly but returned values
  which did not correspond to the ids of the newly created individuals, due to
  a bug in numpy.


Version 0.2
===========

Released on 2011-06-07.

New features:
-------------

* added support for retrospective simulation (ie simulating periods for which we
  already have some data): at the start of each simulated period, if there is 
  any data in the input file for that period, it is "merged" with the result of
  the last simulated period. If there is any conflict, the data in the input
  file has priority.

* added "clone" function which creates new individuals by copying all fields 
  from their "origin" individuals, except for the fields which are given a value
  manually.  

* added breakpoint function, which launches the interactive console during 
  a simulation. Two more console commands are available in that mode:
   
  - "s(tep)" to execute the next process
  - "r(esume)" to resume normal execution

  The breakpoint function takes an optional period argument so that it triggers
  only for that specific period.

* added "tsum" function, which sums an expression over the whole 
  lifetime of individuals. It returns an integer when summing integer or 
  boolean expressions, and a float for float expressions.

* implemented using the value of a periodic global at a specific period. That
  period can be either a constant (eg "MINR[2005]") or an expression 
  (eg "MINR[period - 10]" or "MINR[year_of_birth + 20]")

* added "trunc" function which takes a float expression and returns an int 
  (dropping everything after the decimal point) 

Miscellaneous improvements:
---------------------------

* made integer division (int / int) return floats. eg 1/2 = 0.5 instead of 0.

* processes which do not return any value (csv and show) do not need to be named
  anymore when they are inside of a procedure.

* the array used to run the first period is constructed by merging the
  individuals present in all previous periods.

* print timing for sub-processes in procedures. This is quite verbose but makes
  debugging performance problems/regressions easier.

* made error messages more understandable in some cases.

* manually flush the "console" output every time we write to it, not only within
  the interactive console, as some environments (namely when using the notepad++
  bundle).

* disable compression of the output/simulation file, as it hurts performance
  quite a bit (the simulation time can be increased by more than 60%).
  Previously, it was using the same compression settings as the input file.

* allowed align() to work on a constant. eg: ::

    align(0.0, fname='al_p_dead_m.csv')

* made the "tavg" function work with boolean and float expressions in addition
  to integer expressions

* allowed links to be used in expression given in the "new" function to 
  initialise the fields of the new individuals.

* using "__parent__" in the new() function is no longer necessary.

* made the "init" section optional (it was never intended to be mandatory).

* added progress bar for copying table

* optimised some parts for speed, making the whole simulation roughly as fast as
  0.1 even though more work is done.

Fixes:  
------

* fixed "tavg" function:

  - the result was wrong because the number of values (used in the division)
    was one less than it should.
  - it yielded "random" values when some individuals were present in a past
    period, but not in the current period.

* fixed "duration" function:

  - it crashed when a past period contained no individuals.
  - it yielded "random" values when some individuals were present in a past
    period, but not in the current period.

* fixed "many2one" links returning seemingly random values instead of "missing"
  when they were pointing to an individual which was not present anymore
  (usually because the individual was dead).

* fixed min/max functions.

* fields which are not given an explicit value in new() are initialised to
  missing, instead of 0.

* the result of the new() function (which returns the id of the newly created
  individuals) is now -1 (instead of 0) for parents which are not in the
  filter.

* fixed some expressions crashing when used within a lag.

* fixed the progress bar to display correctly even when there are only very few
  iterations.


Version 0.1
===========

First semi-public release, released on 2011-02-24.