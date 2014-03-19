New features
------------

* added the gumbel() function to draw random numbers from the Gumbel
  distribution (also known as the Smallest Extreme Value (SEV) distribution).

Fixes
-----

* fixed a performance regression in 0.8 when using very large negative
  numbers as indices for a global (eg MINR[bad_index] where bad_indix
  contains several -2147483648 values). Closes issue #121.

* fixed the (debug) interactive console to not produce a useless warning if a
  global temporary was run before entering the console (regression in 0.8).
  Closes issue #120.

* added missing documentation for assertFalse and assertNanEqual.