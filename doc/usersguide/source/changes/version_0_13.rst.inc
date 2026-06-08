Breaking changes
----------------

* This release main purpose was to update the versions of Python (and the whole
  stack of Python packages LIAM2 is built on) to something recent. This version
  is 100% compatible with previous versions in the syntax it supports for the
  models (simulation files) **BUT** results will probably differ from previous
  versions. Closes :issue:`257` and probably a few others.


New features
------------

* added the ability for simulations to delete fields, macros, globals
  and functions defined in an imported simulation. The trick is to set them
  to ~ (yaml syntax for None)

  See the :ref:`Importing other models <import_models>` section for details.


Miscellaneous improvements
--------------------------

* when dumping floating point numbers to CSV, use the nicest (shortest)
  representation of those numbers which would be loaded back as the same value,
  like it was already the case when displaying them (closes :issue:`225`).

* use a safer way to load .yml simulation files. This should make it *a little*
  less dangerous to run a simulation file from an untrusted source. It is still
  a **VERY BAD IDEA** to do so because LIAM2 was never designed with untrusted
  input in mind.


Fixes
-----

* fixed "globals" console command in the presence of scalar globals.

* fixed duration if output file is unspecified (closes :issue:`300`).

* fixed functions with a name ending by "if" (closes :issue:`280`).

* updated numexpr version requirement to fix ``if(constant, expr1, expr2)``.
  Closes :issue:`185`.

* fixed links to numpy documentation (closes :issue:`325`).

* fixed corner case in groupby: when grouping by a single boolean column and
  there were no individuals matching the filter, the result was wrong.

* fixed the error message when a field definition contains a non-string value
  where a keyword is expected (type, initialdata, etc.). See :pr:`303`.
