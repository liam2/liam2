New features
------------

* added support for most of the random number generators provided by Numpy
  which were not already supported by LIAM2: beta, chisquare, dirichlet,
  exponential, f, gamma, geometric, hypergeometric, laplace, lognormal,
  multivariate_normal, noncentral_chisquare, noncentral_f, pareto, power,
  rayleigh, standard_cauchy, standard_exponential, standard_gamma,
  standard_normal, standard_t, triangular, vonmises, wald, weibull, zipf,
  binomial, logseries, negative_binomial, poisson and multinomial (see the
  :ref:`random functions <random_functions>` section for details). Closes
  :issue:`137`.

* added the :ref:`rank_matching() <rank_matching>` function as an alternative
  method to match two sets of individuals. Based on :PR:`136` from Alexis
  Eidelman.

* added an optional "algo" argument to the :ref:`matching() <matching>`
  function, which can be set to either "onebyone" or "byvalue".

  + "onebyone" is the current default and should give the same result as
    previous versions.

  + "byvalue" groups individuals by their value for all the variables involved
    in both the score and orderby expressions, and match groups together.
    Depending on whether all individuals have different combination of values or
    not, this can be much faster than matching each individual in turn. It is
    highly encouraged to use this option as it is much faster in most cases and
    it scales better (O (N1g*N2g) instead of O(N1*N2) where N1g and N2g are the
    number of combination of values in each set and N1 and N2 are the number of
    individuals in each set). However, the results are **NOT** exactly the same
    than in previous versions, even though they are both correct. This means
    that simulation results will be harder to compare against simulation results
    obtained using previous versions. This will be the new default value for
    version 0.10. Please also note that this new option is only available if the
    C extensions are installed. In our test models on actual data, this version
    ran from 50% faster to 3x faster.

  This code is based on the optimized_matching work from :PR:`144` by Alexis
  Eidelman.

* added the possibility to automatically generate an order in matching() by
  using the special value 'EDtM' for its orderby argument. Based on :PR:`136`
  from Alexis Eidelman.

* added an optional 'pool_size' argument to :ref:`matching() <matching>`. If
  used, the best match for an individual is looked for in a random subset of
  size pool_size. Based on :PR:`136` from Alexis Eidelman.

Miscellaneous improvements
--------------------------

* updated bundled dependencies to their latest version. The numpy upgrade to
  version 1.9 brings some performance improvements in various areas (our test
  simulation runs approximately 15% faster overall).

* large internal refactoring

  - it is now easier to define new functions (there is much less code to write).

  - all arguments to all functions can now be expressions. Closes :issue:`5`.

  - cleaner variable scopes. Eliminates a whole class of potential problems
    when using two fields with the same name but a different entity (via a
    link) in the same expression. Closes :issue:`41`.

* cache some internal structure so that it is not recomputed over and over,
  which improves overall performance by a few percents in some cases, especially
  when computing many "small" expressions as is often the case in
  one-by-one matching() (which improved in our tests by 10-20%).

* remove() can now be called without filter argument (it removes all
  individuals)

* better and more consistent error messages when calling functions with
  incorrect arguments (too few, too many, ...)

* use input/path as the base directory for loading .csv globals (those using an
  explicit "path") instead of using the directory of the HDF input file.

* nicer string representation of some expressions (this only affects qshow and
  groupby).

* the --versions command-line argument now also shows versions for optional
  dependencies (if present).

* improved many tests, especially the ones for matching().

Fixes
-----

* fixed the "view" command (to launch ViTables -- via F9 for example) in the 64
  bit bundle. This was a regression in 0.8.2. Closes :issue:`147`.

* fixed computing most expressions involving arrays with more than one
  dimension. It only worked if all the arrays involved were based on the same
  "source" array (which was the case in our tests).

* assertEqual fails gracefully when comparing two arrays with different shapes.

* fixed global fields colliding with fields with the same name from other
  (global) tables.

* fixed expressions like: ::

    if(filter_expr, align(..., array[scalar], ...), False)

  and made all if(expr, GLOBAL[scalar_value], ...) expressions faster in the
  process.

* fixed a rare problem with some expressions using scalars returned by
  aggregate functions.