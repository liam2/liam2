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

* Alignment under and overflows are displayed again, even when debug is
  not set (closes :issue:`155`).

* Always load all declared entities even if there is no process executed for
  them (but produce a warning in that case). This used to cause a problem if
  an entity had not process but was used through links (closes :issue:`89`).
