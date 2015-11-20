New features
------------


* A default value can be assigned to fields that are not present
  in the initial data (see the :ref:`miscellaneous section <model>` 
  for details).

* added support for the "sidewalk" method in :ref:`align_abs() <align_abs>`
  in addition to the existing (and default) "bysorting" method. Thanks to
  Alexis Eidelman and Mahdi Ben Jelloul (:pr:`189`).

* added "functions" command to the interactive console. It displays the
  list of available functions.

* added "help [function]" console command. It displays the signature of the
  function (ie the list of possible arguments and their default value if any).
  For random generator functions it also displays a detailed description
  (from numpy).

* added a 'limit' argument to `dump()` to limit the number of rows of the
  output.

Miscellaneous improvements
--------------------------

* improved the logging message for align with take or leave filters: the take
  and leave filters are combined with the "normal" filter before computing the
  take and leave numbers displayed. It does not change the results in any way,
  only the numbers logged to the console.

* made the demonstration models use 3d alignment files in some cases instead of
  separate alignment files for men and women.

Fixes
-----

* fixed various problems with the demonstration models.

* fixed something (closes :issue:`1`).