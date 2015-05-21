New features
------------

* implemented support for calling :ref:`user-defined functions
  <functions>` (previously called procedures) from within user code,
  optionally passing arguments to them and retrieving their return value.
  As a consequence of this, every mention of "procedures" in the documentation
  has been replaced by "functions". Closes :issue:`152`.

* implemented :ref:`while loops <while>` (closes :issue:`151`).

* added support for using the "inf" (infinite) constant.

Miscellaneous improvements
--------------------------

* made bcolz optional (which is only useful for interpolating the dataset
  during import). Thanks to Mahdi Ben Jelloul (:pr:`161`).

* allow simulations with no processes section (but an init section).

* reworked a few sections of the documentation.

Fixes
-----

* fixed end-of-simulation stats when simulating 0 periods (closes :issue:`158`).

* allow overriding (detected) field types of period and id fields in import
  (closes :issue:`154`).

* fixed autodump (broken in 0.9). Note that it is currently incompatible with
  while loops and recursive (user) functions.