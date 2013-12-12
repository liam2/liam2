.. highlight:: yaml

Change log
##########

Version 0.8.pre1
================

Released on 2013-12-12.

New features:
-------------

* added a few functions to create charts (courtesy of `matplotlib
  <http://matplotlib.org>`_): bar, plot, pie, stackplot and boxplot. As with
  all other functions in liam2, they are available both during a simulation
  and in the interactive console. The charts can either be visualized directly
  or saved to a file. See the :ref:`charts section <charts>` for details.

* added a "view" command line option to Liam2 to open ViTables (an hdf5
  viewer) as well as a corresponding menu entry and keyboard shortcut (F9) in
  Notepad++. It is meant to be used when editing a *model* file, and it will
  open both the input dataset and the result file (if any).

* added new boolean aggregate functions: all() and any(). In fact they were
  added in 0.7 but were not yet documented.

* added *assertFalse* assert function.

Miscellaneous improvements:
---------------------------

* added precisions in the documentation of align() based on Alexis Eidelman
  suggestions.

* made a few more error messages a bit more useful by displaying the line
  where the error occurred.

* adapted the release script since our move to git and converted it to Python.

Fixes:
------

* fixed the upgrade script by removing a special case for grpmin and grpmax as
  it was in fact not needed and caused problems when the expression being
  aggregated contained parentheses.


Version 0.7
===========

Released on 2013-06-18.

New features:
-------------

* implemented imports so that simulation files can be split and reused. 
  This can be used to simply split a large model file into smaller files,
  or (more interestingly) to create simulation variants without having to
  duplicate the common parts. This feature was inspired by some code
  from Alexis Eidelman. For details see the :ref:`import_models` section.

* added new logit and logistic functions. They were previously used
  internally but not available to modellers.  

* added two new debugging features: autodump and autodiff. autodump will dump
  all (non-scalar) variables (including temporaries) at the end of each
  procedure in a separate hdf5 file. It can be used stand-alone for debugging,
  or in combination with autodiff. Autodiff will gather all variables at the
  end of each procedure and compare them with the values stored previously by
  autodump in another run of the model (or a variant of it). This can be used
  to precisely compare two versions/variants of a model and see exactly
  where they start to differ.

* added new assert functions:

  - assertIsClose to check that two results are "almost" equal tolerating
    small value differences (for example due to rounding differences).
  - assertEquiv to check that two results are equal tolerating differences in
    shape (though they must be compatible).
  - assertNanEqual to check that two arrays are equal even in the presence of
    nans (because normally nan != nan).
  
* added a new "timings" option to hide timings from the simulation log, so
  that two simulation logs are more easily comparable (for example with "diff"
  tools like WinMerge).

* added a menu entry in notepad++ to run a simulation in "debug mode".

Miscellaneous improvements:
---------------------------

* improved the performance and memory usage by changing the internal memory
  layout. Most operations are now faster. new(), remove(), "merging data"
  (for retrospective simulations) and writing data at the end of each period
  are now slower. In our model, this translates to a peak memory usage 20%
  smaller and a 35% overall speed increase. However, if your model has a low
  processes/variables ratio, it may very well be slower overall with this
  version. If it is your case, please contact us.

* changed the syntax for all aggregate functions: grpxxx(...) should now be
  xxx(...). For example, grpsum(age) should now be: sum(age). The old syntax is
  still valid but it is deprecated (it will be removed in a later version).
  A special note for grpmin() and grpmax() which becomes min() and max() 
  respectively even though those functions already existed. The meaning is 
  deduced from the number of "non-keyword" arguments:
   
  min(expr1, expr2)
    minimum between expr1 and expr2 (for each individual)

  min(expr)
    (aggregate) minimum value of "expr" over all individuals

  min(expr1, filter=expr2)
    (aggregate) minimum value of "expr" over individuals satisfying the filter 
  
  A tool to automatically upgrade models to the new syntax is provided. In
  notepad++, you should use the **Liam2: upgrade model** command in the
  **Macro** menu. 
  
  You can also run it via the command line: ::
  
    main upgrade model.yml [output.yml]
    
  see main upgrade --help for details.

* changed the syntax for all one2many link functions: xxxlink(link_name, ...)
  should now be link_name.xxx(...). For example, countlink(persons) should now
  be: persons.count(). The old syntax is still valid but it is deprecated (it
  will be removed in a later version). As for aggregate functions, one can
  upgrade its models automatically with the "upgrade" command.

* the "period" argument of value_for_period can now be a *scalar* expression
  (it must have the same value for all individuals).
  
* when the output directory does not exist, Liam2 will now try to create it.

* when debug mode is on, print the position in the random sequence before and
  after operations which use random numbers.
  
* entities are loaded/stored for each period in alphabetical order instead of
  randomly. This has no influence on the results but produces nicer log files.

* deprecated the "predictor" keyword. If you need several processes to write
  to the same variable, you should use procedures instead.

Fixes:
------

* using invalid indexes in "global arrays" do not crash anymore if they are
  properly enclosed in an if() expression. For example if you have an array
  "by_age" with values for indices from 0 to 99, the following code will now
  work as expected: ::
  
    if(age < 50, by_age[age + 50], 0.5)

  Periodic globals are unaffected (they always return "missing" when out of
  bounds).

* fixed link expressions which span 3 (or more) *different* entities.

* fixed using show() on a scalar created by summing a "global array".

* fixed the progress bar of matching() when the number of individuals is
  different in the two sets.


Version 0.6.2
=============

Released on 2013-05-21.

Fixes:
------

* fixed storing a copy of a (declared) field (without any modification) in a
  temporary "backup" variable. The temporary variable was not a copy but an
  alias to the same data, so if the field was modified afterwards, the
  temporary variable was also modified implicitly.
  
  As an example, the following code failed before the fix: ::

    # age is a field
    - backup: age
    # modify age (this also modified backup!)
    - age: age + 1
    # failed because "backup" was equal to "age"
    - assertEqual(age, backup + 1)

  This only affected assignment of "pure" fields, not expressions nor temporary
  variables, for example, the following code worked fine (because backup
  stores an expression, not a simple field): ::

    - backup: age * 1
    - age: age + 1
    - assertEqual(age, backup + 1)
    
  and this code worked too (because temp is a temporary variable, not a field):
  ::
  
    - temp: age + 1
    - backup: temp
    - temp: temp + 1
    - assertEqual(temp, backup + 1)


Version 0.6.1
=============

Released on 2013-03-27.

Miscellaneous improvements:
---------------------------

* when importing an nd-array skip cells with only spaces in addition to empty
  cells.

Fixes:
------

* fixed using non-scalar values (eg fields) as indices of n-dimensional arrays,
  and generally made indexing n-dimensional arrays more robust.
  
* fixed choice which did not refuse to run when the sum of probability is != 1
  and the "error" is > 1e-6, as it should. This was the case in past versions
  but the test was accidentally removed in version 0.5.

* fixed choice to warn when the sum of probabilities is > 1 (and the error is 
  <= 1e-6). Previously, it only warned if the sum was < 1.


Version 0.6
===========

Released on 2013-03-15.

New features:
-------------

* globals handling has been vastly improved:

  - *multiple tables*: one can now define several tables in globals and not
    only the "periodic" table. 
    
    These should be imported in the import file and declared in the simulation
    file in the exact same way that periodic globals are.
    
    Their usage within a simulation is a bit different though: whereas periodic
    global variables can be used without prefixing, others globals need to
    be prefixed with the name of their table. For example, if one has declared
    a global table named "othertable": ::
    
      othertable:
          fields:
              - INTFIELD: int
              - FLOATFIELD: float

    its fields can be used like this: ::
    
      my_variable: othertable.INTFIELD * 10

    These other global tables need not contain a PERIOD column. When using such
    a table, LIAM2 will not automatically subtract the "base period"
    from the index, which means that to access a particular row, you have to
    use its row index (0 based). 

  - *n-dimensional globals*: in addition to tables, globals can now be
    n-dimensional arrays. The file format for those should be the same than
    alignment files. They should be declared like this: ::
    
      MYARRAY: {type: float}

  - globals can now be used in all situations instead of only in simple
    expressions and only for the "current" period. Namely, it makes globals
    available in: link functions, temporal functions (lag, value_for_period,
    ...), matching(), new() and in (all the different flavours of) the
    interactive console.
  
* alignment has been vastly improved:

  - *align_abs* is a new function with the same arguments than align which
    can be used to align to absolute numbers per category, instead of
    proportions. Combined with other improvements in this release, this allows
    maximum flexibility for computing alignment targets on the fly (see below).  
  
  - *align on a linked entity* (a.k.a immigration): additionally to the
    arguments of align, align_abs has also an optional "link" argument, which
    makes it work on the linked entities. The link argument must a one2many
    link. For example, it can be used to take as many *household*s as necessary
    trying to get as close as possible to a particular distribution of
    *persons*. When the link argument is in effect, the function uses the
    "Chenard" algorithm.
    
    In this form, align_abs also supports two extra arguments:
    
    + secondary_axis: name of an axis which will influence rel_need when the
      subtotal for that axis is exceeded. See total_by_sex in Chenard. 
      secondary_axis must be one of the alignment columns.  
    + errors: if set to 'carry', the error for a period (difference between 
      the number of individuals aligned and the target for each category) is
      stored and added to the target for the next period.

  - renamed the "probabilities" argument of align to "proportions"

  - the "proportions" argument of align() is now much more versatile, as all
    the following are now accepted:

    + a single scalar, for aligning with a constant proportion.
    + a list of scalars, for aligning with constant proportions per category.
      (this used to be the only supported format for this argument)
    + an expression returning a single scalar.
    + an expression returning an n-dimensional array. expressions and
      possible_values will be retrieved from that array, so you can simply
      use: ::

        align(score, array_expr)

    + a list of expressions returning scalars [expr1, expr2].
    + a string (in which case, it is treated as a filename). The "fname"
      argument is still provided for backward compatibility.

  - added an optional "frac_need" argument to align() to control how
    "fractional needs" are handled. It can take any of three values: "uniform"
    (default), "cutoff" or "round".

    + "uniform" draws a random number (u) from an uniform distribution and
      adds one individual if u < fractional_need. "uniform" is the default
      behavior.
    + "round" simply rounds needs to the nearest integer. In other words, one
      individual is added for a category if the fractional need for that
      category is >= 0.5.
    + "cutoff" tries to match the total need as closely as possible (at the
      expense of a slight loss of precision for individual categories) by 
      searching for the "cutoff point" that yields: ::

        count(frac_need >= cutoff) == sum(frac_need)

  - changed the order of align() arguments: proportions is now the second
    argument, instead of filter, which means you can omit the "fname" or
    "proportions" keywords and write something like: ::
    
      align(score, 'my_csv_file.csv')

  - made align() (and by extension logit_regr) always return False for
    individuals outside the filter, instead of trying to modify the target
    variable only where the filter is True. That feature seemed like a good
    idea on paper but had a very confusing side-effect: the result was
    different when it was stored in an existing variable than in a new
    temporary variable.

  - it is no longer possible to use expressions in alignment files. If you
    need to align on an expression (instead of a simple variable), you should
    specify the expression in the alignment function. eg: ::

      align(0.0, fname='al_p_dead.csv', expressions=[gender, age + 1])
  
* the result of a groupby can be used in expressions. This can be used, for
  example, to compute alignment targets on the fly.

* implemented explore on data files (.h5), so that one can, for example,
  explore the input dataset.

* added skip_na (defaults to True) argument to all aggregate functions to
  specify whether or not missing values (nan for float expressions, -1 for
  integer expressions) should be ignored.

* macros can now be used in the interactive console.

* added "globals" command in the interactive console to list the available
  globals.

* added qshow() command to show an expression "textual form" in addition to its
  value. Example: ::
  
    qshow(grpavg(age))
  
  will display: ::
  
    grpavg(age): 38.5277057298
  
* added optional "pvalues" argument to groupby() to manually provide the "axis"
  values to compute the expression on, instead of having groupby compute the
  combination of all the unique values present in the dataset for each column.

Miscellaneous improvements for users:
-------------------------------------

* improved the documentation, in part thanks to the corrections and
  suggestions from Alexis Eidelman.

* added a "known issues" section to the documentation.

* grpmin and grpmax ignore missing values (nan and -1) by default like other
  aggregate functions.

* grpavg ignore -1 values for integer expressions like other aggregate
  functions.

* made the operator precedence for "and", "or" and "not" more sensible, which
  means that, for example: ::

    age > 10 and age < 20

  is now equivalent to: ::

    (age > 10) and (age < 20)

  instead of raising an error.

* many2one links are now ~30% faster for large datasets.

* during import, when a column is entirely empty and its type is not specified
  manually, assume a float column instead of failing to import.

* allow "id" and "period" columns to be defined explicitly (even though they
  are still implicit by default).

* allow "period" in any dimension in alignment files, not only in the last one.

* disabled all warnings for x/0 and 0/0. This is not an ideal situation, but it
  is still an improvement because they appeared in LIAM2 code and not in user
  code and as such confused users more than anything.

* the "num_periods" argument of lag: lag(age, num_periods) can now be a
  *scalar* expression (it must have the same value for all individuals).
  
* changed output format of groupby to match input format for alignments.

* added Warning in grpgini when all values (for the filter) are zeros.

* when an unrecoverable error happens, save the technical error log to the
  output directory (for run and explore commands) instead of the directory
  from where liam2 was run and display on the console where the file has been
  saved.

* better error message when an input file has inconsistent row lengths.

* better error message when using a one2many function in a groupby expression.

Miscellaneous improvements for developers:
------------------------------------------

* added a "code architecture" section to the documentation.

* python tracebacks can be re-activated by setting the DEBUG environment
  variable to True. 

* added a script to automate much of the release process.

* added source files for creating liam2 bundle (ie add our custom version of
  notepad++ to the source distribution).

* updated INSTALL file, and include sections on how to build the documentation
  and the C extensions.

* added many tests, fixed a few existing ones and generally greatly improved
  our test suite.

Fixes:
------

* fixed "transposed" option on import. The number of lines to copy was computed
  on the untransposed data which meant too few data points were copied if the
  number columns was greater than the number of lines and it crashed if it was
  smaller.

* fixed all aggregate functions (except grpcount and grpsum) with a filter
  argument equal to a simple variable (eg filter=gender) in the presence of
  "missing" (nan) values in the expression being aggregated: the filter
  variable was modified.

* fixed duration() on a simple variable (eg duration(work)): the variable was
  modified by the function.

* fixed a nasty bug which made that each variable that needed to be read on
  disk (lag of more than one period, duration, value_for_period, ...) was
  read 2 or 3 times instead of just once, greatly slowing down the function.

* fixed accessing columns for the next-to-last period in the interactive
  console after a simulation: it was either giving bad results or returning an
  error.

* fixed all aggregate functions (except grpcount, grpsum and grpavg which
  worked) on boolean expressions. This is actually only (remotely) useful for
  grpgini and grpstd.

* fixed groupby with both filter and expr arguments.

* fixed groupby(expr=scalar).

* fixed sumlink(link, scalar).

* fixed new(number=...).

* fixed non-aligned regressions with a filter (it was ignored).

* fixed the editor shortcuts (to launch liam2) to work when the directory
  containing the model contains spaces.

* fixed handling of comments in the first cell of a row in alignments files
  (the entire row is ignored now).

* fixed "textual form" of choice expressions when bins or choices are dynamic.

* fixed using numpy 1.7

Experimental new features:
--------------------------

* implemented optional periodicity for simulation processes.


Version 0.5.1
=============

Released on 2012-11-28.

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
  
* improved our build system to automate much of the release process.

Fixes:
------

* fixed the "explore" command.

* fixed integer fields on 64 bit platforms other than Windows.

* fixed demo06: WEMRA is an int now.

* fixed demo01 introduction comment (bad file name).


Version 0.5
===========

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

* improved the test coverage:  There is still a long way for full test coverage,
  but the changes in this version is already a first step in the right
  direction:

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


Version 0.4
===========

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


Version 0.3
===========

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