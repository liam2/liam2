Fixes
-----

* Alignment under and overflows are displayed again, even when debug is
  not set (closes :issue:`155`).

* Always load all declared entities even if there is no process executed for
  them (but produce a warning in that case). This used to cause a problem if
  an entity had no process but was used through links (closes :issue:`89`).

* Fixed dump when the filter is False (a scalar), for example in
  :code:`dump(filter=period == 2100)` (closes :issue:`142`).

* Fixed the declared return type of many random distributions, which
  caused a problem if they were used with a filter or within an if()
  expression. This change also fixed cont_regr when both mult and filter
  arguments are used (closes :issue:`153`).
