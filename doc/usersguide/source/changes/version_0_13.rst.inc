New features
------------

* used larray

* implemented an :ref:`if keyword <if_keyword>` for code which should be only executed if a *global* condition is true.
  Note that, like for the :ref:`while keyword <while_keyword>`, this only works if the condition is a single value
  for all individuals. For per-individual conditional code, one should continue using the :ref:`if() function
  <if_function>`.

* implemented choice from an Array and from an external file (closes :issue:`214` and :issue:`215`)

* implemented ext_expr using a global array & use LArray

* added view()


Miscellaneous improvements
--------------------------

* allowed the use of while as a keyword and a function to avoid silent mistakes (closes :issue:`314`).

* support Python 3 and use the latest versions of all the third party libraries (closes :issue:`257`).

* allowed importing import files from within import files like in simulation files (closes :issue:`302`).


Fixes
-----

* added explicit check and error message when trying to store an array of incorrect length in an entity field.

* fixed parsing error when a function name ends with ìf (closes :issue:`280`)

* fixed using an array (as a lookup table) in the score expression of matching().
  Thanks to Mahdi Ben Jelloul (fixes :issue:`294`).

* fixed error message when checking fields definitions (:pr:`133`).

* fixed "globals" console command in the presence of non np.ndarray globals

* Allow the use of while as a keyword and a function to avoid silent mistakes (closes :issue:`314`).

* fixed duration() function when no output file is specified (fixes :issue:`300`).
