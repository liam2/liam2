New features
------------

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

Miscellaneous improvements for users
------------------------------------

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

Miscellaneous improvements for developers
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

Fixes
-----

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

Experimental new features
-------------------------

* implemented optional periodicity for simulation processes.
