.. index:: known issues

Known issues
############

Contextual filter is inconsistent
=================================

First, what is a contextual filter? It is the name we gave to the feature which
propagates the filter of an *if* function to the "True" side of the function,
and the opposite filter to the "False" side. So, for example, in:

  aligned: if(gender, align(0.0, fname='al_p_dead_m.csv')
                      align(0.0, fname='al_p_dead_f.csv'))

the "gender" filter is automatically propagated to the align functions. Which
means, the above code is exactly equivalent to:

  aligned_m: align(0.0, filter=gender, fname='al_p_dead_m.csv')
  aligned_f: align(0.0, filter=not gender, fname='al_p_dead_f.csv')
  aligned: if(gender, aligned_m, aligned_f)

One might wonder what happens if an explicit filter is used in addition to the
contextual filter? Both filters are combined (using "and"), as for example:

  aligned: if(gender, align(0.0, fname='al_p_dead_m.csv', filter=age > 10)
                      align(0.0, fname='al_p_dead_f.csv'))

which is in fact evaluated as:

  aligned_m: align(0.0, filter=gender and age > 10, fname='al_p_dead_m.csv')
  aligned_f: align(0.0, filter=not gender, fname='al_p_dead_f.csv')
  aligned: if(gender, aligned_m, aligned_f)
  
What is the inconsitency anyway?
--------------------------------

This contextual filter propagation is implemented for new(), align(),
logit_regr(), matching() and some (but not all) aggregate functions.
Specifically, it is implemented for grpsum and grpgini, but not for other
aggregate functions (grpcount, grpavg, grpmin, grpmax, grpstd, grpmedian and
grppercentile).

Implementing it would be very easy, but I am unsure whether it makes senses to
have this feature for aggregate functions.

It becomes a bit weird because it means we work at both
the individual level and at the "aggregate" levels in the same expression, or,
in more technical terms, we work with both vectors and scalars, and it might be
confusing: do users realize they are assigning a value for each individual, even
if that is only one of two values?

  age_sum: if(gender, grpsum(age), grpsum(age))

Also, implemeting it for all aggregate functions means we cannot have a
different filter inside an expression, so for example:

    - avg_age_men: grpavg(age, filter=to_marry and gender)
    - difficult_match: if(to_marry and not gender,
                          abs(age - avg_age_men),
                          nan)

could not be rewritten as:

    - difficult_match: if(to_marry and not gender,
                          abs(age - grpavg(age, filter=to_marry and gender)),
                          nan)


The problem is that implementing it for grpavg would
   break difficult_match for example.

   I think the best way out of this is to ignore the contextual filter in all
   aggregate functions. But in that case, what about ctx filter in new(),
   matching(), ...?

I would greatly appreciate more input on the subject, so *please* make your
voice heard if you have an opinion about this, [on the -dev mailing list].
    
31 *different* variables per expression
=======================================

Within a single expression, one may only use 31 *different* variables. There is
a simple workaround though: split your expression in several pieces, each one
using less than 31 variables. Example:

  result: a1 + a2 + ... + a31 + a32 + a33

could be rewritten as:

  tmp: a1 + a2 + ... + a31
  result: tmp + a32 + a33