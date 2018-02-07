Syntax changes
--------------

* defining functions without any argument without using parentheses is now a warning. Closes :issue:`162`.
  In a future release, this will be an error. The goal of this change is both to make models more explicit and
  consistent (a function definition *always* has parentheses) and to make LIAM2 internal code simpler in the long run.

  Given that literally all models ever written need to be updated and doing so by hand would take a lot of time,
  the LIAM2 model upgrader can be used to automatically upgrade model files to the new syntax (the original version of
  your model(s) will be saved in a .bak file).

  If you only have a few model files and you use the Windows bundle, you could open each of your model in turn and use
  the ``LIAM2: upgrade model`` command in the ``Macro`` menu of Notepad++.

  If you do not use the bundle, or have many models to upgrade, you should rather use the command line: ::

     liam2 upgrade <pattern_for_your_models>

  For example: ::

     liam2 upgrade examples/*.yml
     liam2 upgrade */*.yml

  Note that if you are using the Windows bundle the executable is not ``liam2`` but ``liam2/main.exe``


New features
------------

* implemented choice with a different probability per individual (closes :issue:`211`): ::

    - p0: if(gender, 0.1, 0.3)
    - p5: if(gender, 0.2, 0.4)
    - p10: if(gender, 0.7, 0.3)
    - intchoice: choice([0, 5, 10], [p0, p5, p10])

    # the same using a global array (choices must be the first dimension)
    # this particular case will become easier/nicer in a future release
    - global_choice: choice(ARRAY2D.pvalues[0], ARRAY2D[:, gender * 1])

* implemented simulation without any user-visible .h5 output file (closes :issue:`220`). One can use an output
  section without declaring any "file", or with file: '': ::

    output:
        file: ''

  In that case, LIAM2 will create an hidden minimal output file and delete it automatically at the end of the
  simulation. For people who do not use the .h5 output file, this can substantially improve disk usage and slightly
  improve performance when using large datasets. The minimal output file will contain only the fields used in lag
  expression going back in time more than one period (because those are not kept in memory).

* implemented load() function which can load both arrays and tables in the middle of a simulation. ::

    - array: load('param/mig.csv', type=float)
    - table: load('param/othertable.csv',
                  fields=[('PERIOD', int), ('INTFIELD', int), ('FLOATFIELD', float)])

  the above `array` and `table` variable can then be used, within the function, exactly like if they were
  respectively a global array or a global table. The only difference is that they are local to the function and
  thus are discarded when the function terminates. This can be used as a way to transfer groupby arrays from one
  entity to the other, which is otherwise not possible at the moment.

* implemented align(link=) to use proportions in combination with the Chenard algorithm (closes :issue:`216`).

* implemented a way of declaring global constants (based on :pr:`206` by Mahdi Ben Jelloul). For example: ::

    globals:
        MY_BOOL_CONSTANT: True
        MY_FLOAT_CONSTANT: 3.1415
        MY_INT_CONSTANT: 42
        MY_STRING_CONSTANT: "hello"

* chart functions gained *xmin*, *xmax*, *ymin*, *ymax* optional arguments to provide manual bounds for axes.
  By default, they are automatically inferred from the data as before (closes :issue:`209`).

* implemented totals argument to groupby to specify whether or not totals should be computed. Defaults to True like
  before.

* added support for a new "msg" argument to all assertion functions. It specifies a custom message to append to the
  normal message displayed when the assertion fails. This is most useful when the condition of the assertion is complex.
  The message can be **any** expression or tuple of expressions, and it is only evaluated if the assertion fails
  (closes :issue:`208`). ::

    - assertTrue(all(age >= 0), msg="we have persons with negative age!")

  will display: ::

    AssertionError: all((age >= 100)) is not True: we have persons with negative age!

  instead of just: ::

    AssertionError: all((age >= 100)) is not True

  Using dump(), csv(dump()) or breakpoint() as the msg argument can be useful. ::

    - assertTrue(all(age < 150), msg=("we have abnormally old persons", dump(filter=age >= 150))
    - assertTrue(all(total_income >= 0), breakpoint())

* implemented assertRaises to check for expected errors.

* implemented access to the error function (erf) function (if the scipy package is installed, which is **NOT** the case
  in the Windows bundle). Thanks to Mahdi Ben Jelloul (:pr:`234`).


Miscellaneous improvements
--------------------------

* improve installation/getting started instructions and include installation instructions in the documentation
  (instead of only providing them in the source archive). It now includes a section concerning installation on Mac OS X.
  Closes :issue:`192`. Thanks to Paul Williamson for his help concerning installation on Macs.

* use "Read the Docs" theme for the documentation as it is nicer.

* better error messages when trying to use data that is not ordered by period (display row and period) or which
  contains duplicated ids for a period (show row in addition to period and id).

* better error message when a field default value is not of the field type.

* When installing LIAM2 via `python setup.py install`, a liam2 script will be created in the Python installation
  Scripts directory (which is in the system PATH in most cases), so that one can use liam2 on the command line
  without specifying its path. For example: ::

    liam2 run model.yml

* the view command can be called without any file argument. This will launch the embedded ViTables without opening any
  file. In other words, one can now use ``liam2 view`` or ``python main.py view`` without another
  extra argument (closes :issue:`194`).

* made charts (matplotlib) work even without PyQt installed (i.e. fallback to the Tk backend).

* avoid evaluating assertions arguments when using ``assertions: skip``. Previously, only the final test was skipped.

* improved check on the sum of probabilities in sidewalk alignment (:pr:`197`). Thanks to Mahdi Ben Jelloul.

* misc improvements to the code, test models and the documentation, some of which done by Mahdi Ben Jelloul.

* messages when an assertion concerning float values fails now contain all available decimals, instead of rounding
  at the 12th decimal.


Fixes
-----

* fixed running the bundled LIAM2 when another Python distribution is installed in the PATH of the system (closes
  :issue:`222`).

* fixed csv(dump()) rounding float values at the 12th decimal instead of using all available precision when the missing
  argument is used (closes :issue:`252`).

* fixed bug which made it impossible to override an existing field or global definition when importing another model.
  It was using the original/imported field definition and ignoring the overridden definition (closes :issue:`264`).

* fixed fields declared as initialdata: False to load data from the input file anyway if a field with
  the same name existed in the corresponding table (closes :issue:`227`). Additionally, those fields used the
  type from the input table instead of the one declared (if different).

* fixed output: False fields acting like temporary globals (i.e. being wiped at the end of each period) instead of like
  a field (closes :issue:`230`). This had the indirect consequence of having both a field and a temporary variable with
  the same name, which confused dump().

* fixed importing models using relative paths in some cases. Also makes the display of the imported model path nicer
  in that case (:pr:`200`). Thanks to Mahdi Ben Jelloul.

* fixed skip_shows: True in simulation file being ignored.

* fixed ``--skiptimings=False`` being ignored if ``timings: True`` was specified in the simulation file.

* fixed subsetting an array created by indexing a global with a field when the result is an array. ::

   - array: global[field1, :, field2]
   - first_item: array[0]

* fixed show() and csv() on arrays created using some combinations of groupby and global arrays.

* fixed using .transpose() without argument on a LabeledArray (a global or the result of a groupby).

* fixed running LIAM2 in a debugger in some cases.

* fixed some random number generator functions being referenced twice in the documentation index.

* fixed alignment when take or leave filters are a single constant for all individuals (set as a Python bool).

* fixed remove() failing if an array temporary variable is defined in the same function (closes :issue:`222`).

* fixed expressions with both a "non-simple" operation (everything except + - * / and where) and a function call failing
  to evaluate (closes :issue:`186`).
