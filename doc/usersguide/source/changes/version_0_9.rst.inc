New features
------------

* added the rank_matching() function as an other method to match two sets of
  individuals.

* add the possibility to automatically generate an order in matching() by
  using the special value 'EDtM' for its orderby argument.

* add a new 'pool_size' optional parameter in matching(). If an integer is
  entered the best match for an individual is looked for in a random subset
  of size pool_size.

* added most of the random generators provided by Numpy.

Miscellaneous improvements
--------------------------

* massive internal refactoring
  - it is now much easier to define new functions (there is much less code to
    write).

  - all arguments to all functions can now be expressions.

  - cleaner variable scopes. Eliminates a whole class of potential problems
    when using two fields with the same name but a different entity (via a
    link) in the same expression.

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

* fixed links in lag (specific to devel?)

* fixed O2M functions on a different entity with a filter (specific to devel?)

* fixed expressions between arrays with more than one dimension. It only worked
  if all the arrays involved in the expression were based on the same "source"
  array (which was the case in our tests).

* do not expand scalar keys of (global) arrays when the contextual filter is
  an array.

* assertEqual fails gracefully when the comparing two arrays with different
  shapes.

* convert 0d arrays returned by numexpr to scalars.

* fixed global fields colliding with fields of the same name from other
  (global) tables.
