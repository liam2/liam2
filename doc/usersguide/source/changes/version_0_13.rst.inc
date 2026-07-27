About this version
------------------

Even though this version shares the version number that I had intended for the
"big new release" that I was working on a few years ago, this version does not
contain any of those large new features. It "only" updates Python and all the
libraries we use to recent versions and, with one exception, only introduces
small bug fixes and improvements which were already done in my development
branch and could be applied easily to this smaller-scoped version.


Breaking changes
----------------

Even though this version of LIAM2 is 100% compatible with previous versions
regarding the syntax it supports for the model/simulation files, the results
will probably be slightly different from version 0.12, even with the same
random seed.

 - One class of difference is that random numbers are generated differently in
   some cases. Since both LIAM2 versions respect the asked for random
   distribution, both versions are in fact **correct**.
 - The other difference concerns sorting individuals during matching and
   alignment when several individuals have exactly the same score.
   Version 0.12 always picked the same individuals across runs (which is
   correct since the score was respected but not ideal because it lacked some
   randomness), while version 0.13 improves that by choosing the individuals
   at random in that case.

There are two sections below with more details about
:ref:`the differences in results <012_013_results_differences>` and an
explanation of :ref:`how to modify your models to have the same results in
version 0.13 than in version 0.12 <reproducing_012_results>` if you are
interested in that. Closes :issue:`257` and probably a few others.


New features
------------

* added the ability for simulations to delete fields, macros, globals
  and functions defined in an imported simulation. The trick is to set them
  to ~ (yaml syntax for None)

  See the :ref:`Importing other models <import_models>` section for details.


Miscellaneous improvements
--------------------------

* when dumping floating point numbers to CSV, use the nicest (shortest)
  representation of those numbers which would be loaded back as the same value,
  like it was already the case when displaying them. This makes the CSV output
  (visually) different than before but a computer would interpret it the same
  way (closes :issue:`225`).

* use a safer way to load .yml simulation files. This should make it *a little*
  less dangerous to run a simulation file from an untrusted source. It is still
  a **VERY BAD IDEA** to do so because LIAM2 was never designed with untrusted
  input in mind.


Fixes
-----

* fixed "globals" console command in the presence of scalar globals.

* fixed duration if output file is unspecified (closes :issue:`300`).

* fixed functions with a name ending by "if" (closes :issue:`280`).

* updated numexpr version requirement to fix ``if(constant, expr1, expr2)``.
  Closes :issue:`185`.

* fixed links to numpy documentation (closes :issue:`325`).

* fixed corner case in groupby: when grouping by a single boolean column and
  there were no individuals matching the filter, the result was wrong.

* fixed the error message when a field definition contains a non-string value
  where a keyword is expected (type, initialdata, etc.). See :pr:`303`.


.. _012_013_results_differences:

Why are LIAM2 version 0.13 results different than 0.12?
-------------------------------------------------------

Because of the different versions of Python and Numpy (a very important package
in our dependencies), the results from version 0.13 are often different from
the results of version 0.12. If you do not care about reproducing results
from earlier versions, you might want to skip both this section and the next
one, but you might be interested in the slightly related newly added
:ref:`warning in logit_score's documentation <logit_score_safe_zone>` which is
probably interesting for many users.

The differences come from two root causes:

 1) the Python dictionary iteration order changed. This affects:

    * ``new()`` and ``clone()`` when those have several random-using expressions
      (for example a ``choice()``). Since the order of evaluation of the
      expressions was deterministic but completely arbitrary and now uses the
      order in which a model defines them, the random values are different
      because they come from some kind of "sequence of random numbers", which
      is consumed by the random-using expressions.  Both versions of LIAM2 are
      correct.

    * ``matching(algo='byvalue')`` with some random-using component in the score
      expression.

 2) The default (Numpy) indirect sort algorithm changed slightly and sorts
    equal values differently than before. This affects:

    * ``matching(algo='onebyone')`` (which is the default matching algorithm).
      There is only a problem when several individuals have exactly the same
      value for orderby.

    * ``align()`` (with or without the ``link`` argument), when several
      individuals have the same value for the score. This also affects
      ``logit_regr()`` with an ``align`` argument.

    Version 0.12 and earlier always picked the same individuals in that case,
    which is not ideal because it artificially restricts the randomness of the
    outcome. In other words, if you ran your model many times (without
    fixing the random seed) and your score was always the same, the same
    individuals will always be picked while there is no justification for that
    (other individuals with the same score should be just as likely to be
    picked).

    Since the order changed in that case anyway because of the change in Numpy
    (which made results differ), I introduced the shuffling of individuals with
    the same score to improve randomness.


.. _reproducing_012_results:

How to reproduce LIAM2 0.12 results using version 0.13?
-------------------------------------------------------

If you absolutely want to produce the same results with version 0.12 and
version 0.13, when using the same random seed, it is possible to modify your
models (and their results) to achieve that. Be warned that this can be a tedious
process for large models and it is **NOT necessary** and **not recommended**
if you do not need exact reproducibility between the two versions.

If your model hits any of the problematic cases cited above (``new()``,
``clone()``, ``matching()``, ``logit_regr()`` and ``align()``), it is **NOT**
possible to reproduce with version 0.13 the exact same results that your
original model produced using version 0.12. It is only possible to change the
model so that both versions produce the same results (but different from what
you had). If you still want to do so, you need to modify the model in the
following ways :

 * add ``v012_partial_compatibility: True`` and (if not present already)
   ``seed: some_constant_number_of_your_choice`` in the simulation block of
   your simulation file. The ``v012_partial_compatibility: True`` flag has two
   effects: it turns off the automatic shuffling of individuals when they have
   the same score (so you temporarily get worse/less random results) and
   displays a warning in that case so that you can fix that manually instead
   of relying on the automatic solution. See below for an explanation of what
   is necessary.

 * if you have ``new()`` and ``clone()`` using several random-using expressions,
   you need to compute a temporary variable with the random-using expression for
   all individuals outside of the ``new()``/``clone()`` function.
   You would then have different results than before (because the model would
   "consume" more values from the "random sequence") but the result should be
   the same in both versions. For example, in the demo08 model, we have: ::

    - new('person', filter=to_give_birth,
          mother_id = id,
          education_level = choice([LOWER_SECONDARY_EDU,
                                    UPPER_SECONDARY_EDU,
                                    TERTIARY_EDU],
                                   [0.25, 0.39, 0.36]),
          age = 0,
          gender = choice([MALE, FEMALE], [0.51, 0.49])
          # more fields omitted for brevity
          )

   The two ``choice()`` there are making results differ between versions. To
   avoid this, the code can be modified as such: ::

    - baby_edu: choice([LOWER_SECONDARY_EDU, UPPER_SECONDARY_EDU, TERTIARY_EDU],
                       [0.25, 0.39, 0.36])
    - baby_gender: choice([MALE, FEMALE], [0.51, 0.49])
    - new('person', filter=to_give_birth,
          mother_id = id,
          age = 0,
          education_level = baby_edu,
          gender = baby_gender
          # more fields omitted for brevity
          )

 * if you use ``matching()`` with ``algo='byvalue'`` and the score expression
   uses some random component, the easiest workaround to make the results the
   same across versions is to use ``algo='onebyone'`` instead (which is the
   default). The procedure will be slower though. See below for how to solve
   the second problem with ``matching()`` (non-unique values in orderby).

 * for both ``matching(algo='onebyone')`` and ``align()``, when the score is a
   simple “mathematical expression” NOT involving any ``logit_score()`` but which
   produces some non-unique values, the simplest workaround is to add some
   small uniform random component to the score/orderby expression.
   The trick is to make the “high” bound of ``uniform()`` smaller than the
   smallest difference between any two values (that is why the warning from
   ``v012_partial_compatibility`` computes and displays that value for you), so
   that the relative order of individuals only changes between individuals
   which had the same value. For example, with code like: ::

    - partner_id: if(to_couple,
                     matching(set1filter=FEMALE, set2filter=MALE,
                              score=<some long expression omitted for brevity>,
                              orderby=difficult_match),
                     partner_id)"

   In that model's case, ``difficult_match`` is an integer value and thus the
   minimum difference between any two values is 1 and the code could thus be
   changed to: ::

    - partner_id: if(to_couple,
                     matching(set1filter=FEMALE, set2filter=MALE,
                              score=<some long expression omitted for brevity>,
                              orderby=difficult_match + uniform(high=0.5)),
                     partner_id)

 * When the score uses ``logit_score()`` either explicitly or via
   ``logit_regr()``, this gets more complicated.
   This was not mentioned anywhere so far but ``logit_score()`` gives best
   results only if its input is roughly between -6 and 6. When that is not the
   case, you risk having it return the same value even if you pass different
   values. See :ref:`the warning about this in logit_score's documentation
   <logit_score_safe_zone>` for more details. Now to fix the non-unique issue
   in this case is a bit trickier and more tedious. Since the output values of
   ``logit_score()`` are sometimes already very close to each other (especially
   when outside the logit_score "safe zone"), we cannot use the ``uniform()``
   trick to make the output of ``logit_score()`` "unique enough" for sorting
   (because there are not enough floating point numbers available between the
   output values). As a consequence, we instead need to modify the input values
   going into the ``logit_score()`` function. The simplest way to do this is
   just to rescale the input expression to make sure it fits the “safe range”.

   For example, in one model, we had code like this: ::

        - married_score: -0.030816 * agediff **2
                         + 0.0013858 * agediff **3
                         + 1.576726 * (educ_level - partner.educ_level)
                         # (more terms omitted for brevity)
        - married: logit_regr(married_score,
                              filter = FEMALE and just_matched,
                              align = 'al_p_mar_married.csv')

   Adding the following line in between: ::

        - married_score: (married_score * 6) / max(married_score)

   solved the issue in that model's case. Note that for that model, there was
   no problem with the lower bound, so it was sufficient to only make sure the
   input values are below the 6 upper bound...

 * If you use "soft-takes" and "soft-leaves" in alignment, like in : ::

      if(some_condition, constant_higher_than_one,
         if(some_other_contition, constant_lower_than_zero,
            logit_score(expr)))

   which is meant to force taking those higher_than_one individuals
   in priority but not necessarily take them all; and select those
   lower_than_zero individuals last but not necessarily avoid them entirely.

   In that case, using a ``uniform()`` in place of the constant works well.
   For example, using ``uniform(2, 3)`` instead of the constant ``2`` and
   ``uniform(-1, 0)`` instead of ``-1`` worked in our case.

 * Interestingly ``logit_regr(0.0, align='my_alignment_file.csv')`` is not
   affected by the problem because ``0.0`` is within the safe range and is
   optimized to ``align(uniform(), 'my_alignment_file.csv')`` anyway.
