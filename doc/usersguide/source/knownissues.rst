.. index:: known issues

Known issues
############

Contextual filter is inconsistent
=================================

First, what is a contextual filter? It is the name we gave to the feature which
propagates the filter of an *if* function to the "True" side of the function,
and the opposite filter to the "False" side. So, for example, in: ::

  - aligned: if(gender, align(0.0, 'al_p_dead_m.csv')
                        align(0.0, 'al_p_dead_f.csv'))

the "gender" filter is automatically propagated to the align functions. Which
means, the above code is exactly equivalent to: ::

  - aligned_m: align(0.0, 'al_p_dead_m.csv', filter=gender)
  - aligned_f: align(0.0, 'al_p_dead_f.csv', filter=not gender)
  - aligned: if(gender, aligned_m, aligned_f)

One might wonder what happens if an explicit filter is used in addition to the
contextual filter? Both filters are combined (using "and"), as for example: ::

  - aligned: if(gender, align(0.0, 'al_p_dead_m.csv', filter=age > 10)
                        align(0.0, 'al_p_dead_f.csv'))

which is in fact evaluated as: ::

  - aligned_m: align(0.0, 'al_p_dead_m.csv', filter=gender and age > 10)
  - aligned_f: align(0.0, 'al_p_dead_f.csv', filter=not gender)
  - aligned: if(gender, aligned_m, aligned_f)
  
What is the inconsistency anyway?
---------------------------------

This contextual filter propagation is implemented for new(), align(),
logit_regr(), matching() and **some** (but not all) aggregate functions.
Specifically, it is implemented for sum and gini, but not for other
aggregate functions (count, avg, min, max, std, median and
percentile). This situation needs to be changed, but I am unsure in which
way: implementing it for all aggregate functions or not contextutal filter
for any aggregate function (or any function at all)?

While this features feels natural for new, align and logit_regr, it feels 
out of place for aggregate functions because it means we work at both
the individual level and at the "aggregate" levels in the same expression, or,
in more technical terms, we work with both vectors and scalars, and it might be
confusing: do users realize they are assigning a value for each individual,
even if that is only one of two values?

In an expression like the following: :: 

  - age_sum: if(gender, sum(age), sum(age))
  
do users realize they are assigning a different value for both branches? When I
see an expression like this, I think: "it returns the same value whether the
condition is True or not, let's simplify it by removing the condition": ::   
  
  - age_sum: sum(age)
  
which will not have the same result.

Another (smaller) point, is that implementing this contextual filter feature
means one cannot "escape" the filter of an if function, so for example: ::

  - difficult_match: if(to_marry and not gender,
                        abs(age - avg(age, filter=to_marry and gender)),
                        nan)

would not work, and would need to be rewritten as: :: 

  - avg_age_men: avg(age, filter=to_marry and gender)
  - difficult_match: if(to_marry and not gender,
                        abs(age - avg_age_men),
                        nan)

I would greatly appreciate more input on the subject, so *please* make your
voice heard if you have an opinion about this, [on the -dev mailing list].
    
31 *different* variables per expression
=======================================

Within a single expression, one may only use 31 *different* variables. There is
a simple workaround though: split your expression in several pieces, each one
using less than 31 variables. Example: ::

  - result: a1 + a2 + ... + a31 + a32 + a33

could be rewritten as: ::

  - tmp: a1 + a2 + ... + a31
  - result: tmp + a32 + a33
