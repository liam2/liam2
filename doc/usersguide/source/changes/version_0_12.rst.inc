New features
------------

* implemented choice with a different probability per individual (closes :issue:`211`)

    - p0: if(gender, 0.1, 0.3)
    - p5: if(gender, 0.2, 0.4)
    - p10: if(gender, 0.7, 0.3)
    - intchoice: choice([0, 5, 10], [p0, p5, p10])

    # the same using a global array (choices must be the first dimension)
    # this particular case will become easier/nicer in a future release
    - global_choice: choice(ARRAY2D.pvalues[0], ARRAY2D[:, gender * 1])

* implemented simulation without any user-visible .h5 output file (closes :issue:`220`). One can use an output
  section without declaring any "file", or with file: '':

    output:
        file: ''

  In that case, LIAM2 will create an hidden minimal output file and delete it automatically at the end of the
  simulation. For people who do not use the .h5 output file, this can substantially improve disk usage and slightly
  improve performance when using large datasets.  The minimal output file will contain only the fields used in lag
  expression going back in time more than one period (because those are not kept in memory).

* implemented load() function which can load both arrays and tables in the middle of a simulation.

    - array: load('param/mig.csv', type=float)
    - table: load('param/othertable.csv',
                  fields=[('PERIOD', int), ('INTFIELD', int), ('FLOATFIELD', float)])

  the above `array` and `table` variable can then be used, within the function, exactly like if they were
  respectively a global array or a global table. The only difference is that they are local to the function and
  thus are discarded when the function terminates. This can be used as a way to transfer groupby arrays from one
  entity to the other, which is otherwise not possible at the moment.

* chart functions gained *xmin*, *xmax*, *ymin*, *ymax* optional arguments to provide manual bounds for axes.
  By default, they are automatically inferred from the data as before (closes :issue:`209`).

* implemented totals argument to groupby to specify whether or not totals should be computed. Defaults to True like
  before.

* implemented experimental align(link=) to use proportions in combination with the Chenard algorithm. It needs more
  testing though.

* implemented assertRaises to check for expected errors.


Miscellaneous improvements
--------------------------

* the view command can be called without any file argument. This will launch the embedded ViTables without opening any
  file. In other words, one can now use `liam2/main.exe view` (on Windows) or `python main.py view` without another
  extra argument (closes :issue:`194`).

* better error messages when trying to use data that is not ordered by period (display row and period) or which
  contains duplicated ids for a period (show row in addition to period and id).

* better error message when a field default value is not of the field type.

* improved check on the sum of probabilities in sidewalk alignment (:pr:`197`). Thanks to Mahdi Ben Jelloul.

* When installing LIAM2 via `python setup.py install`, a liam2 script will be created in the Python installation
  Scripts directory (which is in the system PATH in most cases), so that one can simply call:
  `liam2 run model.yml` from anywhere.

* made charts (matplotlib) work even without PyQt installed (i.e. fallback to the Tk backend).

* avoid evaluating assertions arguments when using `assertions: skip`. Previously, only the final test was skipped.

* misc improvements to the code, test models and the documentation, some of which done by Mahdi Ben Jelloul.


Fixes
-----

* fixed importing models using relative paths in some cases. Also makes the display of the imported model path nicer
  in that case (:pr:`200`). Thanks to Mahdi Ben Jelloul.

* fixed running the bundled LIAM2 when another Python distribution is installed in the PATH of the system (closes
  :issue:`222`).

* fixed skip_shows: True in simulation file being ignored.

* fixed --skiptimings=False being ignored if timings: True was specified in the simulation file.

* fixed subsetting an array created by indexing a global with a field when the result is an array.

   - array: global[field1, :, field2]
   - first_item: array[0]

* fixed show() and csv() on arrays created using some combinations of groupby and global arrays.

* fixed using .transpose() without argument on a LabeledArray (a global or the result of a groupby).

* fixed running LIAM2 in a debugger in some cases.

* fixed some random number generator functions being referenced twice in the documentation index.
