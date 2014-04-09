Miscellaneous improvements
--------------------------

* simplified and cleaned up the demonstration models.

* improved the error message when a link points to an unknown entity.

* the evaluator creates fewer internal temporary variables in some cases. 

Fixes
-----

* added log and exp to the list of available functions (they were already
  implemented but not usable because of that).

* fixed log_regr, cont_regr and clip_regr which were comparing their result with
  0.5 (like logit_regr when there is no alignment).
 
* fixed new() function, which created individuals correctly but in some cases
  returned values which did not correspond to the ids of the newly created
  individuals, due to a bug in numpy.
