New features
------------

* added the gumbel() function to draw random numbers from the Gumbel
  distribution (also known as the Smallest Extreme Value (SEV) distribution).

Miscellaneous improvements
--------------------------

Fixes
-----

* fixed a performance regression in 0.8 when using very large negative
  numbers as indices for a global (eg MINR[bad_index] where bad_indix
  contains several -2147483648 values). Closes issue #121.

* added missing documentation for assertFalse and assertNanEqual.