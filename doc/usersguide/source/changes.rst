.. highlight:: yaml

Change log
##########

Version 0.9.1.1
===============

Released on 2015-04-14.

.. include:: changes/version_0_9_1_1.rst.inc


Version 0.9.1
=============

Released on 2015-03-31.

.. include:: changes/version_0_8_1.rst.inc


Version 0.8
===========

* give a hint to use assertNanEqual when it would make a failing assertEqual
  pass.

* removed the predictor keyword support (it now raises an exception instead
  of a warning).

* sped up global[array_expr].

* implemented global[slice_expr] (eg. MINR[period: period+2]). When the
  slice bounds are arrays (different for each individual) and the slice
  length is not constant (not the same for all individals),
  it returns a special array with an extremely limited set of supported
  operations: only aggregates on axis=1 are implemented.

Fixes:
------
=======
Released on 2014-02-05.

.. include:: changes/version_0_8.rst.inc


Version 0.7
===========

Released on 2013-06-18.


New features:
-------------

* implemented imports so that simulation files can be split and reused. 
  This can be used to simply split a large model file into smaller files,
  or (more interestingly) to create simulation variants without having to
  duplicate the common parts. This feature was inspired by some code
  from Alexis Eidelman. For details see the :ref:`import_models` section.

* added new logit and logistic functions. They were previously used
  internally but not available to modellers.  

* added two new debugging features: autodump and autodiff. autodump will dump
  all (non-scalar) variables (including temporaries) at the end of each
  procedure in a separate hdf5 file. It can be used stand-alone for debugging,
  or in combination with autodiff. Autodiff will gather all variables at the
  end of each procedure and compare them with the values stored previously by
  autodump in another run of the model (or a variant of it). This can be used
  to precisely compare two versions/variants of a model and see exactly
  where they start to differ.

* added new assert functions:

  - assertIsClose to check that two results are "almost" equal tolerating
    small value differences (for example due to rounding differences).
  - assertEquiv to check that two results are equal tolerating differences in
    shape (though they must be compatible).
  - assertNanEqual to check that two arrays are equal even in the presence of
    nans (because normally nan != nan).
  
* added a new "timings" option to hide timings from the simulation log, so
  that two simulation logs are more easily comparable (for example with "diff"
  tools like WinMerge).

* added a menu entry in notepad++ to run a simulation in "debug mode".

Miscellaneous improvements:
---------------------------

* improved the performance and memory usage by changing the internal memory
  layout. Most operations are now faster. new(), remove(), "merging data"
  (for retrospective simulations) and writing data at the end of each period
  are now slower. In our model, this translates to a peak memory usage 20%
  smaller and a 35% overall speed increase. However, if your model has a low
  processes/variables ratio, it may very well be slower overall with this
  version. If it is your case, please contact us.

* changed the syntax for all aggregate functions: grpxxx(...) should now be
  xxx(...). For example, grpsum(age) should now be: sum(age). The old syntax is
  still valid but it is deprecated (it will be removed in a later version).
  A special note for grpmin() and grpmax() which becomes min() and max() 
  respectively even though those functions already existed. The meaning is 
  deduced from the number of "non-keyword" arguments:
   
  min(expr1, expr2)
    minimum between expr1 and expr2 (for each individual)

  min(expr)
    (aggregate) minimum value of "expr" over all individuals

  min(expr1, filter=expr2)
    (aggregate) minimum value of "expr" over individuals satisfying the filter 
  
  A tool to automatically upgrade models to the new syntax is provided. In
  notepad++, you should use the **LIAM2: upgrade model** command in the
  **Macro** menu. 
  
  You can also run it via the command line: ::
  
    main upgrade model.yml [output.yml]
    
  see main upgrade --help for details.

* changed the syntax for all one2many link functions: xxxlink(link_name, ...)
  should now be link_name.xxx(...). For example, countlink(persons) should now
  be: persons.count(). The old syntax is still valid but it is deprecated (it
  will be removed in a later version). As for aggregate functions, one can
  upgrade its models automatically with the "upgrade" command.

* the "period" argument of value_for_period can now be a *scalar* expression
  (it must have the same value for all individuals).
  
* when the output directory does not exist, LIAM2 will now try to create it.

* when debug mode is on, print the position in the random sequence before and
  after operations which use random numbers.
  
* entities are loaded/stored for each period in alphabetical order instead of
  randomly. This has no influence on the results but produces nicer log files.

* deprecated the "predictor" keyword. If you need several processes to write
  to the same variable, you should use procedures instead.

Fixes:
------

* using invalid indexes in "global arrays" do not crash anymore if they are
  properly enclosed in an if() expression. For example if you have an array
  "by_age" with values for indices from 0 to 99, the following code will now
  work as expected: ::
  
    if(age < 50, by_age[age + 50], 0.5)

  Periodic globals are unaffected (they always return "missing" when out of
  bounds).

* fixed link expressions which span 3 (or more) *different* entities.

* fixed using show() on a scalar created by summing a "global array".

* fixed the progress bar of matching() when the number of individuals is
  different in the two sets.
=======
.. include:: changes/version_0_7.rst.inc


Version 0.6.2
=============

Released on 2013-05-21.

.. include:: changes/version_0_6_2.rst.inc


Version 0.6.1
=============

Released on 2013-03-27.

.. include:: changes/version_0_6_1.rst.inc


Version 0.6
===========

Released on 2013-03-15.

.. include:: changes/version_0_6.rst.inc


Version 0.5.1
=============

Released on 2012-11-28.

.. include:: changes/version_0_5_1.rst.inc


Version 0.5
===========

Released on 2012-10-25.

.. include:: changes/version_0_5.rst.inc


Version 0.4.1
=============

Released on 2011-12-02.

.. include:: changes/version_0_4_1.rst.inc


Version 0.4
===========

Released on 2011-11-25.

.. include:: changes/version_0_4.rst.inc


Version 0.3
===========

Released on 2011-06-29.

.. include:: changes/version_0_3.rst.inc


Version 0.2.1
=============

Released on 2011-06-20.

.. include:: changes/version_0_2_1.rst.inc


Version 0.2
===========

Released on 2011-06-07.

.. include:: changes/version_0_2.rst.inc


Version 0.1
===========

First semi-public release, released on 2011-02-24.