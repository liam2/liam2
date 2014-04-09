Miscellaneous improvements
--------------------------

* when importing an nd-array skip cells with only spaces in addition to empty
  cells.

Fixes
-----

* fixed using non-scalar values (eg fields) as indices of n-dimensional arrays,
  and generally made indexing n-dimensional arrays more robust.
  
* fixed choice which did not refuse to run when the sum of probability is != 1
  and the "error" is > 1e-6, as it should. This was the case in past versions
  but the test was accidentally removed in version 0.5.

* fixed choice to warn when the sum of probabilities is > 1 (and the error is 
  <= 1e-6). Previously, it only warned if the sum was < 1.
