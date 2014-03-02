Miscellaneous improvements
--------------------------

* validate both import and simulation files, i.e. detect bad structure and
  invalid and missing keywords.

* improved error messages (both during import and the simulation), by stripping
  any information that is not useful to the user. For some messages, we only
  have a line number and column left, this is not ideal but should be better
  than before. The technical details are written to a file (error.log) instead.

* improved "incoherent alignment data" error message when loading an alignment
  file by changing the wording and adding the path of the file with the error.

* reorganised bundle files so that there is no confusion between directories
  for Notepad++ and those of liam2.
   
* tweaked Notepad++ configuration:

  - added explore command as F7
  - removed more unnecessary features.

Fixes
-----

* disallowed using one2many links like many2one (it was never intended this way
  and produced wrong results).

* fixed groupby with a scalar expression (it does not make much sense, but it is
  better to return the result than to fail).

* re-enabled the code to show the expressions containing errors where possible
  (in addition to the error message). This was accidentally removed in a
  previous version.

* fixed usage to include the 'explore' command.
