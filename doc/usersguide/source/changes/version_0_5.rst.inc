New features
------------

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

Miscellaneous improvements
--------------------------

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

Fixes
-----

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
