New features
------------

* implemented user-defined functions (see the :ref:`bla bla section <bla>` for
  details).

* implemented :ref:`while loops <while>` (closes :issue:`151`).

* added support for using the "inf" (infinite) constant.
.. _bla:

Miscellaneous improvements
--------------------------

* made bcolz optional (only useful for interpolating the dataset during import).
  Thanks to Mahdi Ben Jelloul (in :pr:`161`).

* allow simulations with no processes section (but an init section)

Fixes
-----

* fixed end-of-simulation stats when simulating 0 periods (closes
  :issue:`158`).

* allow overriding (detected) field types of period and id fields in import
  (closes :issue:`154`).

* fixed autodump (broken in 0.9). Note that it is currently incompatible with
  while loops and recursive (user) functions.