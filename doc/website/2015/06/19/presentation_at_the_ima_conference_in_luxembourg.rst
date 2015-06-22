Presentation at the IMA conference in Luxembourg
================================================

I will be giving a presentation titled "LIAM2 - overview and recently added
features" at the Fifth World Congress of the International Microsimulation
Association, 2-4 September 2015, Luxembourg.

Abstract
--------

This presentation will be divided into two parts. First a quick overview of what
is LIAM2, then a description of all the new features since the LIAM2 course at
the IMA conference in 2013 and more briefly those since our initial presentation
in 2011. Each of those features will be briefly explained with examples of how
they were used in real models.

Overview
~~~~~~~~

LIAM2 is a free, open source, user-friendly modelling and simulation framework.
It is made as generic as possible so that it can be used to develop almost any
type of discrete-time dynamic microsimulation model with cross-sectional dynamic
ageing (i.e. all individuals are simulated at the same time for one period, then
for the next period, etc.). LIAM2 is clearly aiming to free "modellers" from
having to develop or care about having state-of-the-art methods for
data-handling or expression evaluation and yet be able to handle relatively
large datasets at a reasonable speed. For example, a model like MIDAS in Belgium
simulated over 60 years with 2.2 million individuals initially could be
developed in a user-friendly environment and is run in less than 4 hours. To
date, LIAM2 has been adopted by modellers in at least 7 countries.

New features since 2011
~~~~~~~~~~~~~~~~~~~~~~~

* model imports and model variants
* many new functions including:

  - charting functions
  - aggregate functions: gini, percentile, ...
  - new alignment options (using absolute values or on a linked entity using
    Chenard's algorithm)
  - debugging functions (assertions, ...)

* data viewer (HDF5 viewer)
* syntax changes
* officially open source and hosted on GitHub

New features since 2013
~~~~~~~~~~~~~~~~~~~~~~~

* user defined functions
* while loops
* improved handling of external data
* new matching algorithms
* more random number generators


.. author:: default
.. tags:: presentation
.. comments::