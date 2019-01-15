New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* improved something.

Fixes
-----

* added explicit check and error message when trying to store an array of incorrect length in an entity field.

* Fix parsing error of function when their name termination is ìf (closes :issue:`280`)

* fixed using an array (as a lookup table) in the score expression of matching().
  Thanks to Mahdi Ben Jelloul (fixes :issue:`294`).

* fixed error message when checking fields definitions (:pr:`133`).

* allowed importing import files from within import files like in simulation files (closes :issue:`302`).

* Allow the use of while as a keyword and a function to avoid silent mistakes (closes :issue:`314`).

* fixed duration() function when no output file is specified (fixes #300).
