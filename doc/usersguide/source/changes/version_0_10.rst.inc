New features
------------

* implemented support for calling :ref:`user-defined functions
  <functions>` (previously called procedures) from within user code,
  optionally passing arguments to them and retrieving their return value.
  As a consequence of this, every mention of "procedures" in the documentation
  has been replaced by "functions", including the name of the "procedures"
  logging level which is now deprecated. Closes :issue:`152`.

* implemented :ref:`while loops <while>` (closes :issue:`151`).

* added support for :ref:`not storing some fields in the output file
  <fields_output>` (closes :issue:`58`).

* added support for using the "inf" (infinite) constant.

Miscellaneous improvements
--------------------------

* improved our error handling code to display the (part of the) line where the
  error occurred in more cases and to not strip the traceback (error.log) of
  some important information in the cases where the line was already displayed.

* configured the bundle editor (Notepad++) to display some warning and
  error lines in red in the console log. Additionally, when the error message
  contains a filename and/or a line number (this is currently too rare
  unfortunately), it is now click-able (to jump directly at the file/line).

* defining a process outside of a function has been deprecated because it is
  ambiguous. For example, this code will now trigger a warning and will be
  an error in a future version: ::

    entities:
        person:
            fields:
                agegroup: int
            processes:
                agegroup: 10 * trunc(age / 10)
    simulation:
        processes:
            - person: [agegroup]

  It should be replaced by (or possibly moved into another existing function):
  ::

    entities:
        person:
            fields:
                agegroup: int
            processes:
                compute_agegroup:
                    - agegroup: 10 * trunc(age / 10)
    simulation:
        processes:
            - person: [compute_agegroup]

  If this construct was used to have a temporary field (ie the field was not
  declared in the fields section) accessible from several functions, like ::

    entities:
        person:
            processes:
                tempfield: 0

                func1:
                    tempfield: count()

                func2:
                    otherfield: tempfield + 1

  One should now declare that field with `output: False` instead. ::

    entities:
        person:
            fields:
                tempfield: {type: int, initialdata: False, output: False}

            processes:
                func1:
                    tempfield: count()

                func2:
                    otherfield: tempfield + 1

  This closes :issue:`124`.

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