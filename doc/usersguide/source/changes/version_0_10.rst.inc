New features
------------

* implemented support for calling :ref:`user-defined functions
  <functions>` (previously called procedures) from within user code,
  optionally passing arguments to them and retrieving their return value.
  As a consequence of this, every mention of "procedures" in the documentation
  has been replaced by "functions", including the name of the "procedures"
  logging level which is now deprecated. Closes :issue:`152`.

* implemented :ref:`while loops <while>` (closes :issue:`151`).

* added support for using the "inf" (infinite) constant.

Miscellaneous improvements
--------------------------

* improved our error handling code to display the (part of the) line where the
  error occurred in more cases and to not strip the traceback (error.log) of
  some important information in the cases where the line was already displayed.

* configured the bundle editor (Notepad++) to display some warning and
  error lines in red in the console log. Additionally, when the error message
  contains a filename and/or a line number (this is currently very rare
  unfortunately), it is now click-able (to jump directly at the file/line).

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
  while loops and recursive (user-defined) functions.