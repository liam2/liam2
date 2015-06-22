New features
------------

* allowed to override all simulation options except "init" and "processes"
  using the command-line. Run "main run -h" for details (closes :issue:`173`).

Fixes
-----

* fixed the error message when using an unknown variable to be more explicit
  like it was in versions before 0.9 (closes :issue:`167`).

* the output of the demonstration models is no longer included in the bundles
  (closes :issue:`169`).

* made simulations which terminate in an error exit with the correct error code
  even without --debug (closes :issue:`174`).
