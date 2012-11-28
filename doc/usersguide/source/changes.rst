.. highlight:: yaml

Change log
##########

Version 0.5.1
=============

Released on 2012-11-26.

Miscellaneous improvements:
---------------------------

* if there is only one entity defined in a model (like in demo01.yml) and the
  interactive console is launched, start directly in that entity, instead of
  requiring the user to set it manually.  

* improved introduction comments in demo models.

* display whether C extensions are used or not in --versions.

* use default_entity in demos (from demo03 onward).

* do not display python version in normal execution but only in --versions.

* use cx_freeze instead of py2exe to build executables for Windows so that
  we can use the same script to build executables across platforms and tweaked
  further our build script to minimise the executable size. 
  
* compressed as many files as possible in the 32 bit Windows bundle with UPX
  to make the archive yet smaller (UPX does not support 64 bit executables
  yet).

Fixes:
------

* fixed the "explore" command.

* fixed integer fields on 64 bit platforms other than Windows.

* fixed demo06: WEMRA is an int now.

* fixed demo01 introduction comment (bad file name).


Version 0.5.0
=============

Released on 2012-10-25.

New features:
-------------

* added a way to import several files for the same entity. A few comments are
  in order:

  - Each file can have different data points. eg if you have historical data
    for some fields data going back to 1950 for some individuals, and other
    fields going back to only 2000, the import mechanism will merge those data
    sets. 
  - It can also optionally fill missing data points. Currently it only
    supports filling with the "previous value" (the value the individual had
    (if any) for that field in a previous period). In the future, we will add
    more ways to fill those by interpolating existing data. Note that
    *currently* only data points which are entirely missing are filled, not
    those which are set to the special value corresponding to "missing" for the
    field type (i.e. False for booleans, -1 for integers and "nan" for floats).
    This will probably change in the future.
  - As a consequence of this new feature, it is now possible to import liam1
    files using the "normal" import file syntax.

* added an optional "default_entity" key to the "simulation" block of
  simulation files, so that the interactive console starts directly in that
  entity.

* added function to compute the Nth percentile: grppercentile(expr, percent[,
  filter]).

* implemented an optional filter argument for many functions. The behaviour is
  different depending on the kind of function:

  - for functions that change an existing variable (clip() and round()), the
    value for filtered individuals is not modified.
  - for functions which create a new variable (uniform(), normal() and
    randint()), the value for filtered individuals is the missing value
    corresponding with the type of the column (-1 for randint(), nan for
    uniform() and normal()).
  - for aggregate functions (grpmin(), grpmax(), grpstd(), grpmedian() and
    grppercentile()), the aggregate is computed over the individuals who
    satisfy the filter.

* added new functions for testing: assertTrue and assertEqual:

  - assertTrue(expr) evaluates its expression argument and check that it is
    True.
  - assertEqual(expr1, expr2) evaluates its two expressions and check that
    they are equal.

* The behaviour when an assertion fails is configurable through the
  "assertions" option in the "simulation" block. This option can take three
  values:

  - "raise": interrupt the simulation (this is the default).
  - "warn": display a warning message.
  - "skip": do not run the assertion at all. 

* added commands to the console:

  - entities: prints the list of available entities.
  - periods: prints the list of available periods for the current entity.

* added new command line arguments to override paths specified in the
  simulation file:

  - --input-path: override the input path
  - --input-file: override the input file
  - --output-path: override the output path
  - --output-file: override the output file
                        
* added --versions command line argument to display versions of all the
  libraries used.

Miscellaneous improvements:
---------------------------

* performance optimisations:

  - fields which are used in lag expressions are cached (stored in memory) to
    avoid fetching them from disk. This considerably speeds up lag expressions
    at the expense of a bit more memory used.
  - implemented a few internal functions in Cython to get C-level performance.
    This considerably speeds up alignment and groupby expressions, especially
    when the number of "alignment categories" (the number of possible
    combinations of values for the variables used to partition) is high.
    The down side is that if someone wants to recreate liam2 binaries from the
    source code and benefit from this optimisation (there is a pure-python
    fallback), he needs to have cython and a C compiler installed.
  - other minor optimisations to groupby and alignments with take or leave
    filters.
  - slightly sped up initial data loading for very large datasets with a lot of
    historical data. 

* choices() arguments (options and probabilities) now accept expressions
  (ie. they can be computed at run time).

* improved the interactive console:

  - made the interactive console start in the last simulated period by default.
  - changed the behaviour of the "entity" command without argument to print the
    current entity.
  - the "period" command can now be called without argument to print the
    current period.

* added more explicit checks for bad input:

  - check for duplicate headers in alignment files.
  - check all arguments to groupby() are valid instead of only the first one.
  - check for invalid keyword arguments to dump().
  - check for invalid keyword arguments to csv().
  - check the type of arguments to choice().
  - validate globals at load time to make sure the declared globals are
    actually present in the dataset.

* disallow strings for the score expression in the matching() function.

* improved the test coverage:  There is still a long way for full test coverage, but the changes in this
  version is already a first step in the right direction:

  - automated many tests by using the new assertions functions.
  - added more tests.

* only copy declared globals to the output file, and do not create a "globals"
  node at all if there is no declared global.

* manually close input and output files when an error happens during
  initialisation, so that the user only sees the real error message.

* globals can be entirely missing from the input file if they are not used in
  the simulation file.

* made the usual code clean-ups.

Fixes:
------

* fixed typo in the code outputting durations ("hourss" instead of "hours").

* fixed a bug which prevented to define constants without quoting them in some
  cases.

* fixed a crash when all groups were empty in a groupby(xxx, expr=grpcount(),
  percent=True).

* fixed aggregate functions (grpmin, grpmax, grpstd, grpmedian and
  grppercentile) to accept a scalar as argument (even though it is not very
  useful to do that).

* fixed a bug which prevented to use a simulation output file as input in some
  cases.


Version 0.4.1
=============

Released on 2011-12-02.

Miscellaneous improvements:
---------------------------

* validate both import and simulation files, i.e. detect bad structure and
  invalid and missing keywords.

* improved error messages (both during import and the simulation), by stripping
  any information that is not useful to the user. For some messages, we only
  have a line number and column left, this is not ideal but should be better
  than before. The technical details are written to a file (error.log) instead.

* improved "incoherent alignment data" error message when loading an alignment
  file by changing the wording and adding the path of the file with the error.

* reorganised bundle files so that there is no confusion between directories
  for Notepad++ and those of liam2.
   
* tweaked Notepad++ configuration:

  - added explore command as F7
  - removed more unnecessary features.

Fixes:  
------

* disallowed using one2many links like many2one (it was never intended this way
  and produced wrong results).

* fixed groupby with a scalar expression (it does not make much sense, but it is
  better to return the result than to fail).

* re-enabled the code to show the expressions containing errors where possible
  (in addition to the error message). This was accidentally removed in a
  previous version.

* fixed usage to include the 'explore' command.


Version 0.4.0
=============

Released on 2011-11-25.

New features:
-------------

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

Miscellaneous improvements:
---------------------------

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

Fixes:  
------

* fixed logit_regr(align=float).

* fixed grpavg(bool, filter=cond).

* fixed groupby(a, b, c, expr=grpsum(d), percent=True).

* fixed having several grpavg with a filter argument in the same expression.

* fixed calling the main executable without argument (simply display usage).

* fixed dump with (some kind of) aggregate values in combination with a filter.

* fixed void data source.


Version 0.3.0
=============

Released on 2011-06-29.

New features:
-------------

* added ability to import csv files directly with the main executable. 

Miscellaneous improvements:
---------------------------

* made periodic globals optional.

* improved a few sections of the documentation.

Fixes:  
------

* fixed non-assignment "actions" in interactive console (csv, remove, ...).

* fixed error_var argument to cont_regr, clip_regr and log_regr.


Version 0.2.1
=============

Released on 2011-06-20.

Miscellaneous improvements:
---------------------------

* simplified and cleaned up the demonstration models.

* improved the error message when a link points to an unknown entity.

* the evaluator creates fewer internal temporary variables in some cases. 

Fixes:  
------

* added log and exp to the list of available functions (they were already
  implemented but not usable because of that).

* fixed log_regr, cont_regr and clip_regr which were comparing their result with
  0.5 (like logit_regr when there is no alignment).
 
* fixed new() function, which created individuals correctly but in some cases
  returned values which did not correspond to the ids of the newly created
  individuals, due to a bug in numpy.


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

* processes which do not return any value (csv and show) do not need to be
  named anymore when they are inside of a procedure.

* the array used to run the first period is constructed by merging the
  individuals present in all previous periods.

* print timing for sub-processes in procedures. This is quite verbose but makes
  debugging performance problems/regressions easier.

* made error messages more understandable in some cases.

* manually flush the "console" output every time we write to it, not only within
  the interactive console, as some environments (namely when using the notepad++
  bundle) do not flush the buffer themselves.

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

* added progress bar for copying table.

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