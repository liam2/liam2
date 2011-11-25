.. highlight:: yaml
.. index::
    single: processes;

.. _processes_label:

Processes
#########

The processes are the core of a model. LIAM2 supports two kinds of processes: 
*assignments*, which change the value of a variable (predictor) using an
expression, and *actions* which don't (but have other effects).

For each entity (for example, "household" and "person"), the block of processes
starts with the header "processes:". Each process then starts at a new line with
an indentation of four spaces.

Assignments
===========

Assignments have the following general format: ::

    processes:
        variable1_name: expression1
        variable2_name: expression2
        ...

The variable_name will usually be one of the variables defined in the **fields**
block of the entity but, as we will see later, it is not always necessary.

In this case, the name of the process equals the name of the *endogenous
variable*. *Process names* have to be **unique** for each entity. See the
section about procedures if you need to have several processes which modify the
same variable.

To run the processes, they have to be specified in the "processes" section of
the simulation block of the file. This explains why the *process names* have
to be unique for each entity.

*example* ::

    entities:
        person:
            fields:
                - age: int
            processes:
                age: age + 1
    simulation:
        processes:
            - person: [age]
        ...

Temporary variables
===================

All fields declared in the "fields" section of the entity are stored in the
output file. Often you need a variable only to store an intermediate result
during the computation of another variable.

In LIAM2, you can create a temporary variable at any point in the simulation by
simply having an assignment to an undeclared variable. Their value will be
discarded at the end of the period.

*example* ::

    person:
        fields:
            # period and id are implicit
            - age:      int
            - agegroup: int

    processes:
        age: age + 1
        agediv10: trunc(age / 10)
        agegroup: agediv10 * 10
        agegroup2: agediv10 * 5

In this example, *agediv10* and *agegroup2* are temporary variables. In this
particular case, we could have bypassed the temporary variable, but when a long
expression occurs several times, it is often cleaner and more efficient to
express it (and compute it) only once by using a temporary variable.

Actions
=======

Since actions don't return any value, they do not need a variable to store that
result, and they only ever need the condensed form: ::

    processes:
        process_name: action expression
        ...

*example* ::

    processes:
        remove_deads: remove(dead)

Procedures
==========

A process can consist of sub-processes, in that case we call it a *procedure*.
Processes within a procedure are executed in the order they are declared.

Sub-processes each start on a new line, again with an indentation of four spaces
and a -.

So the general setup is: ::

    processes:
        variable_name: expression
        process_name2: action_expression
        process_name3:
            - subprocess_31: expression
            - subprocess_32: expression

In this example, there are three processes, of which the first two do not have
sub-processes. The third process is a procedure which consists of two
sub-processes. If it is executed, subprocess_31 will be executed and then
subprocess_32.

Contrary to normal processes, sub-processes (processes inside procedures) names
do not need to be unique. In the above example, it is possible for subprocess_31
and subprocess_32 to have the same name, and hence simulate the same variable.
Procedure names (process_name3) does not directly refer to a specific endogenous
variable.

*example* ::

    processes:
        ageing:
            - age: age * 2 # in our world, people age strangely
            - age: age + 1
            - agegroup: trunc(age / 10) * 10

The processes on *age* and *agegroup* are grouped in *ageing*. In the simulation
block you specify the *ageing*-process if you want to update *age* and
*agegroup*.

By using procedures, you can actually make *building blocks* or modules in the
model.

Temporary variables
-------------------

Temporary variables defined/computed within a procedure are local to that
procedure: they are only valid within that procedure. If you want to pass
variables between procedures you have to define them in the **fields** section.

*(bad) example* ::

    person:
        fields:
            - age: int

        processes:
            ageing:
                - age: age + 1
                - isold: age >= 150   # isold is a local variable

            rejuvenation:
                - age: age – 1
                - backfromoldage: isold and age < 150  # WRONG !

In this example, *isold* and *backfromoldage* are local variables. They can only
be used in the procedure where they are defined. Because we are trying
to use the local variable *isold* in another procedure in this example, LIAM 2
will refuse to run, complaining that *isold* is not defined.

Actions
-------

Actions inside procedures don't even need a process name.

*example* ::

    processes:
        death_procedure:
            - dead: age > 150
            - remove(dead)

.. index::
    single: expressions;

Expressions
===========

Deterministic changes
---------------------

Let us start with a simple increment; the following process increases the value
of a variable by one each simulation period.

    age: age + 1

The name of the process is *age* and what it does is increasing the variable
*age* of each individual by one, each period.

.. index::
    single: simple expressions;


simple expressions
~~~~~~~~~~~~~~~~~~

- Arithmetic operators: +, -, \*, /, \** (exponent), % (modulo)

Note that an integer divided by an integer returns a float. For example "1 / 2"
will evaluate to 0.5 instead of 0 as in many programming languages. If you are
only interested in the integer part of that result (for example, if you know the
result has no decimal part), you can use the *trunc* function: ::

    agegroup5: 5 * trunc(age / 5)

- Comparison operators: <, <=, ==, !=, >=, >
- Boolean operators: and, or, not

Note that you have to use parentheses when you mix *boolean operators* with
other operators. ::

    inwork: (workstate > 0) and (workstate < 5)
    to_give_birth: not gender and (age >= 15) and (age <= 50)

- Conditional expressions:
    if(condition, expression_if_true, expression_if_false)

*example* ::

    agegroup_civilstate: if(age < 50,
                            5 * trunc(age / 5),
                            10 * trunc(age / 10))

Note that an *if*-statement has always three arguments. If you want to leave a
variable unchanged if a condition is not met, specify its value in the
*expression_if_false* ::

    # retire people (set workstate = 9) when aged 65 or more
    workstate: if(age >= 65, 9, workstate)

You can nest if-statements. The example below retires men (gender = True) over
64 and women whose age equals at least the parameter/periodic global "WEMRA"
(Women Retirement Age). ::

    workstate: if(gender,
                  if(age >= 65, 9, workstate),
                  if(age >= WEMRA, 9, workstate))


.. index::
    single: mathematical functions;

mathematical functions
~~~~~~~~~~~~~~~~~~~~~~

- log(expr): natural logarithm (ln)
- exp(expr): exponential
- abs(expr): absolute value
- round(expr[, n]): returns the rounded value of expr to specified n (number of
  digits after the decimal point). If n is not specified, 0 is used.
- trunc(expr): returns the truncated value (by dropping the decimal part) of
  expr as an integer.
- clip(x, a, b): returns a if x < a, x if a < x < b, b if x > b.
- min(x, a), max(x, a): the minimum or maximum of x and a.


.. index::
    single: aggregate functions;

aggregate functions
~~~~~~~~~~~~~~~~~~~

- grpcount([condition]): count the objects in the entity. If filter is given, only
                      count the ones satisfying the filter.
- grpsum(expr[, filter=condition]): sum the expression
- grpavg(expr[, filter=condition]): average
- grpstd(expr): standard deviation
- grpmax(expr), grpmin(expr): max or min
- grpmedian(expr): median
- grpgini(expr[, filter=condition]): gini

**grpsum** sums any expression over all the individuals of the current entity.
For example *grpsum(earnings)* will produce the sum of the earnings of all
persons in the sample. The expression *grpsum(nch0_11)* will
result in the total number of children 0 to 11 in the sample.

**grpcount** counts the number of individuals in the current entity, optionally
satisfying a (boolean) condition. For example, *grpcount(gender)* will produce
the total number of men in the sample. Contrary to **grpsum**, the grpcount
does not require an argument: *grpcount()* will return the total number of
individuals in the sample.

Note that, grpsum and grpcount are exactly equivalent if their only argument
is a boolean variable (eg. grpcount(ISWIDOW) == grpsum(ISWIDOW)).

*example* ::

    macros:
        WIDOW: civilstate == 5
    processes:
        cnt_widows: show(grpcount(WIDOW))

.. index:: countlink, sumlink, avglink, minlink, maxlink

link functions
~~~~~~~~~~~~~~
(one2many links)

- countlink(link[, filter])
- sumlink(link, expr[, filter])
- avglink(link, expr[, filter])
- minlink/maxlink(link, expr[, filter])

*example* ::

    entities:
        household:
            fields:
                # period and id are implicit
                - nb_persons: {type: int, initialdata: false}
            links:
                persons: {type: one2many, target: person, field: household_id}

            processes:
                household_composition:
                    - nb_persons: countlink(persons)
                    - nb_students: countlink(persons, workstate == 1)
                    - nch0_11: countlink(persons, age < 12)
                    - nch12_15: countlink(persons, (age > 11) and (age < 16))

.. index:: temporal functions, lag, value_for_period, duration, tavg, tsum

temporal functions
~~~~~~~~~~~~~~~~~~

- lag: value at previous period
- value_for_period: value at specific period
- duration: number of consecutive period the expression was True
- tavg: average of an expression since the individual was created
- tsum: sum of an expression since the individual was created

If an item did not exist at that period, the returned value is -1 for a
int-field, nan for a float or False for a boolean. You can overide this
behaviour when you specify the *missing* parameter.

*example* ::

    lag(age, missing=0) # the age each person had last year, 0 if newborn
    grpavg(lag(age))    # average age that the current population had last year
    lag(grpavg(age))    # average age of the population of last year

    value_for_period(inwork and not male, 2002)

    duration(inwork and (earnings > 2000))
    duration(educationlevel == 4)

    tavg(income)

.. index:: random, uniform, normal, randint

random functions
~~~~~~~~~~~~~~~~

- uniform: random numbers with a uniform distribution
- normal: random numbers with a normal distribution
- randint: random integers between bounds

*example* ::

    # a random variable with the stdev derived from errsal
    normal(loc=0.0, scale=grpstd(errsal))
    randint(0, 10)

Stochastic changes I: probabilistic simulation
----------------------------------------------

.. index:: choice

choice
~~~~~~

Monte Carlo or probabilistic simulation is a method for iteratively evaluating a deterministic model using sets of random numbers
as inputs. In microsimulation, the technique is used to simulate changes of state dependent variables. Take the simplest example:
suppose that we have an exogenous probability of an event happening, P(x=1), or not P(x=0). Then draw a random number u from an
uniform (0,1) distribution. If, for individual i, ui<p(1), then xi=1. If not, then xi=0. The expected occurrences of x after,
say, 100 runs is then P(x=1)*100 and the expected value is 1xP(1)+0xP(0)=P(1). This type of simulation hinges on the
confrontation between a random variable and an exogenous probability. In the current version of LIAM 2, it is not possible to
combine a choice with alignment.

In LIAM 2, such a probabilistic simulation is called a **choice** process. Suppose i=1..n choice options, each with a probability
prob_option_i. The choice process then has the following form: ::

    choice([option_1, option_2, ..., option_n],
           [prob_option_1, prob_option_2, ..., prob_option_n])

Note that both lists of options and pertaining probabilities are between []’s. Also, the variable containing the options can be
of any numeric type.

A simple example of a choice process is the simulation of the gender of newborns (51% males and 49% females), as such: ::

    gender=choice([True, False], [0.51, 0.49])

The code below illustrates a more complex example of a choice process (called *collar process*). Suppose we want to
simulate the work status (collar=1 (blue collar worker), white collar worker) for all working individuals. We however have
knowledge one’s level of education (education_level=2, 3, 4).

The process *collar_process* has collar as the key endogenous variable and has four sub-processes.

The first sub-process defines a local variable filter-bw, which will be used to separate those that the procedure should apply
to. These are all those that do not have a value for collar, and who are working, or who are in education or unemployed, which
means that they potentially could work.

The next three "collar" sub-processes simulate whether one is a white or blue collar worker, depending on the
level of education. If one meets the above filter_bw and has the lowest educational attainment level, then one has a
probability of about 84% (men) and 69% (women) of being a blue collar worker. If one has ‘education_level’ equal to 3, the
probability of being a blue collar worker is of course lower (64% for men and 31% for women), and the probability of becoming a
blue collar worker is lowest (8 and 4%, respectively) for those having the highest educational attainment level. ::

    collar_process:  # working, in education, unemployed or other inactive
        - filter_bw: (
                       ((workstate > 0) and (workstate < 7))
                       or
                       (workstate == 10)
                      ) and (collar == 0)
        - collar: if(filter_bw and (education_level == 2),
                      if(gender,
                         choice([1, 2], [0.83565, 0.16435]),
                         choice([1, 2], [0.68684, 0.31316]) ),
                      collar)
        - collar: if(filter_bw and (education_level == 3),
                      if(gender,
                         choice([1, 2], [0.6427, 1 - 0.6427]),
                         choice([1, 2], [0.31278, 1 - 0.31278]) ),
                      collar)
        - collar: if(filter_bw and (education_level == 4),
                      if(gender,
                         choice([1, 2], [0.0822, 1 - 0.0822]),
                         choice([1, 2], [0.0386, 1 - 0.0386]) ),
                      collar)

.. index:: logit, alignment

Stochastic changes II: behavioural equations
--------------------------------------------

- Logit: 
    * logit_regr(expr[, filter=None, align='filename'])
    * logit_regr(expr[, filter=None, align=percentage])

- Alignment :
    * align(expr[, take=take_filter, leave=leave_filter], fname='filename.csv')
- Continuous (expr + normal(0, 1) * mult + error_var): cont_regr(expr[, filter=None, mult=0.0, error_var=None])
- Clipped continuous (always positive): clip_regr(expr[, filter=None, mult=0.0, error_var=None])
- Log continuous (exponential of continuous): log_regr(expr[, filter=None, mult=0.0, error_var=None])


*example* ::

    divorce: logit_regr(0.6713593 * household.nch12_15
                        - 0.0785202 * dur_in_couple
                        + 0.1429621 * agediff,
                        filter=FEMALE and (civilstate == 2),
                        align='al_p_divorce.csv')

    wage_earner: if((age > 15) and (age < 65) and inwork,
                    if(MALE,
                       align(wage_earner_score,
                             fname='al_p_wage_earner_m.csv'),
                       align(wage_earner_score,
                             fname='al_p_wage_earner_f.csv')),
                    False)

.. index:: logit_regr

logit_regr
~~~~~~~~~~

Suppose that we have a logit regression that relates the probability of some
event to explanatory variables X.

    p*i=logit-1(ßX + EPSi)

This probability consists of a deterministic element (as before), completed by a
stochastic element, EPSi, a log-normally distributed random variable. The
condition for the event occurring is p*i > 0.5.

Instead, suppose that we want the proportional occurrences of the event to be equal to an overall proportion X. In that
case, the variable p*i sets the rank of individual i according to the risk that the relevant event will happen. Then only
the first X*N individuals in the ranking will experience the event. This process is known as ‘alignment’.

In case of one logit with one alignment process -or a logit without alignment-,
*logit_regr* will result in the logit returning a Boolean whether the event is
simulated. In this case, the setup becomes: ::

    - single_align: logit_regr(<logit arguments>,
                               [filter=<filter arguments>,
                               align='name.csv'])

*example* ::

    birth:
        - to_give_birth: logit_regr(0.0,
                                    filter=FEMALE and
                                           (age >= 15) and (age <= 50),
                                    align='al_p_birth.csv')

The above generic setup describes the situation where one logit pertains to one
alignment process.

.. index:: logit_score

logit_score
~~~~~~~~~~~

In many cases, however, it is convenient to use multiple logits with the same alignment process. In this case, using  a **logit_score** instead of
**logit_regr** will result in the logit returning intermediate scores that - for all conditions together- are the inputs of the
alignment process. A typical behavioural equation with alignment has the following syntax: ::

        name_process:
            # initialise the score to -1
            - score_variable: -1

            # first condition
            - score_variable: if(condition_1,
                                 logit_score(logit_expr_1),
                                 score_variable)
            # second condition
            - score_variable: if(condition_2,
                                 logit_score(logit_expr_2),
                                 score_variable)

            # ... other conditions ...

            # do alignment based on the scores calculated above
            - name_endogenous_variable:
                if(condition,
                   if(gender,
                      align(score_variable,
                            [take=conditions,]
                            [leave=conditions,]
                            fname='filename_m.csv'),
                      align(score_variable,
                            [take=conditions,]
                            [leave=conditions,]
                            fname='filename_f.csv')),
                   False)

The equation needs to simulate the variable *name_endogenous_variable*. It starts however by creating a score that reflects
the event risk p*i. In a first sub-process, a variable *name_score* is set equal to -1, because this makes it highly
unlikely that the event will happen to those not included in the conditions for which the logit is applied. Next, subject to
conditions *condition_1* and *condition_2*, this score is simulated on the basis of estimated logits. The specification
*logit_score* results in the logit not returning a Boolean but instead a score.

Note that by specifying the endogenous variable *name_score* without any transformations under the ‘ELSE’ condition makes
sure that the score variable is not manipulated by a sub-process it does not pertain to.


.. index:: align, take, leave

align
~~~~~

After this step, the score is known and this is the input for the alignment process. Suppose -as is mostly the case- that
alignment data exists for men and women separately. Then the alignment process starts by a *if* to gender. Next comes the
align command itself. This takes the form ::

    align(score_variable,
          filter=conditions,
          [take=conditions,]
          [leave=conditions,]
          fname='name.csv')

The file *name.csv* contains the alignment data. A standard setup is that the file starts with the prefix *al_* followed by
the name of the endogenous variable and a suffix *_m* or *_f*, depending on gender.

The optional *take* and *leave* commands forces inclusion or exclusion of objects with specified characteristics in
the selection of the event. The individuals with variables specified in the *take* command will a priori be selected for the
event. Suppose that the alignment specifies that 10 individuals should experience a certain event, and that there are 3
individuals who meet the conditions specified in the *take*. Then these 3 individuals will be selected a priori and the
alignment process will select the remaining 7 candidates from the rest of the sample. The *leave* command works the other
way around: those who match the condition in that statement, are a priori excluded from the event happening. The *take* and
*leave* are absolute conditions, which mean that the individuals meeting these conditions will always (*take*) or never
(*leave*) experience the event.

Their *soft* counterparts can easily be included by manipulating the score of individuals.
If this score is set to a strong positive or negative number, then the individual will a priori have a high of low
probability of the event happening. These *soft take* and ‘*soft leave*’s will implement a priority order in the sample of
individuals, but will not under all circumstances conditionally include or exclude.

Note that even if the score is -1 an item can be selected by the alignment procedure. This happens when there are not enough
candidates (selected by the score) to meet the alignment needs.

The below application describes the process of being (or remaining) a wage-earner or employee. It illustrates a *soft
leave* by setting the a priori score variable *wage_earner_score* to -1. This makes sure that the a priori
selection probability for those not specified in the process is very low (but not zero, as in the case of *leave*
conditions).

Next come three sub processes setting a couple of common conditions, in the form of local (temporary) variables. These three sub-
processes are followed by six subsequent *if* conditions, separating the various behavioural equations to the sub-sample
they pertain to. The first three sub conditions pertain to women and describe the probability of being a wage-earner from in
work and employee previous year (1) from in work but not employee previous year (2), and from not in work previous year
(3). The conditions 4 to 6 describe the same transitions but for women. ::

    wage_earner_process:
        - wage_earner_score: -1
        - lag_public: lag((workstate == 2) or (workstate == 3))
        - inwork: (workstate > 0) and (workstate < 5)
        - lag_inwork: lag((workstate > 0) and (workstate < 5))
        - men_inwork: gender and (age > 15) and (age < 65) and inwork

        # === MEN ===
        # Probability of being employee from in work and employee previous year
        - wage_earner_score:
            if(men_inwork and ((lag(workstate) == 1) or (lag(workstate) == 2)),
               logit_score(0.0346714 * age + 0.9037688 * (collar == 1)
                           - 0.2366162 * (civilstate == 3) + 2.110479),
               wage_earner_score)
        # Probability of becoming employee from in work but not employee
        # previous year
        - wage_earner_score:
            if(men_inwork and ((lag(workstate) != 1) and (lag(workstate) != 2)),
               logit_score(-0.1846511 * age - 0.001445 * age **2
                           + 0.4045586 * (collar == 1) + 0.913027),
               wage_earner_score)
        # Probability of becoming employee from not in work previous year
        - wage_earner_score:
            if(men_inwork and (lag(workstate) > 4),
               logit_score(-0.0485428 * age + 1.1236 * (collar == 1) + 2.761359),
               wage_earner_score)

        # === WOMEN ===
        - women_inwork: not gender and (age > 15) and (age < 65) and inwork

        # Probability of being employee from in work and employee previous year
        - wage_earner_score:
            if(women_inwork and ((lag(workstate) == 1) or (lag(workstate) == 2)),
               logit_score(-1.179012 * age + 0.0305389 * age **2
                           - 0.0002454 * age **3
                           - 0.3585987 * (collar == 1) + 17.91888),
               wage_earner_score)
        # Probability of becoming employee from in work but not employee
        # previous year
        - wage_earner_score:
            if(women_inwork and ((lag(workstate) != 1) and (lag(workstate) != 2)),
               logit_score(-0.8362935 * age + 0.0189809 * age **2
                           - 0.000152 * age **3 - 0.6167602 * (collar == 1)
                           + 0.6092558 * (civilstate == 3) + 9.152145),
               wage_earner_score)
        # Probability of becoming employee from not in work previous year
        - wage_earner_score:
            if(women_inwork and (lag(workstate) > 4),
               logit_score(-0.6177936 * age + 0.0170716 * age **2
                           - 0.0001582 * age**3 + 9.388913),
               wage_earner_score)

        - wage_earner: if((age > 15) and (age < 65) and inwork,
                           if(gender,
                              align(wage_earner_score,
                                    fname='al_p_wage_earner_m.csv'),
                              align(wage_earner_score,
                                    fname='al_p_wage_earner_f.csv')),
                           False)

The last sub-procedure describes the alignment process. Alignment is applied to individuals between the age of 15 and 65 who
are in work. The reason for this is that those who are not working obviously cannot be working as a wage-earner. The input-
files of the alignment process are 'al_p_wage_earner_m.csv' and 'al_p_wage_earner_f.csv'. The alignment process sets the
Boolean *wage earner*, and uses as input the scores simulated previously, and the information it takes from the alignment
files. No ‘take’ or ‘leave’ conditions are specified in this case.

Note that the population to align is the population specified in the first condition, here *(age>15) and (age<65) and (inwork)* and not the
whole population.

.. index:: lifecycle functions

Lifecycle functions
-------------------

.. index:: new

new
~~~

**new** creates items initiated from another item of the same entity (eg. a
women gives birth) or another entity (eg. a marriage creates a new houshold).

*generic format* ::

    new('entity_name', filter=expr,
        *set initial values of a selection of variables*)

The first parameter defines the entity in which the item will be created (eg
person, household, ...).

Then, the filter argument specifies which items of the current entity will serve
as the origin for the new items (for persons, that would translate to who is
giving birth, but the function can of course be used for any kind of entity).

Any subsequent argument specifies values for fields of the new individuals. Any
field which is not specified there will receive the missing value corresponding
to the type of the field ('nan' for floats, -1 for integers and False for
booleans). Those extra arguments can be given constants, but also any
expression (possibly using links, random functions, ...). Those expressions are
evaluated in the context of the origin individuals. For example, you could write
"mother_age = age", which would set the field "mother_age" on the new item to
the age of their mother.

*example 1* ::

    birth:
        - to_give_birth: logit_regr(0.0,
                                    filter=not gender and
                                           (age >= 15) and (age <= 50),
                                    align='al_p_birth.csv')
        - new('person', filter=to_give_birth,
              mother_id = id,
              father_id = partner.id,
              household_id = household_id,
              partner_id = -1,
              age = 0,
              civilstate = 1,
              collar = 0,
              education_level = -1,
              workstate = 5,
              gender=choice([True, False], [0.51, 0.49]) )

The first sub-process (*to_give_birth*) is a logit regression over women (not
gender) between 15 and 50 which returns a boolean value whether that person
should give birth or not. The logit itself does not have a deterministic part
(0.0), which means that the ‘fertility rank’ of women that meet the above
condition, is only determined by a logistic stochastic variable). This process
is also aligned on the data in 'al_p_birth.csv'.

In the above case, a new person is created for each time a woman is scheduled to
give birth. Secondly, a number of links are established: the value for the
*mother_id* field of the child is set to the id-number of his/her mother, the
child receives the household number of his/her mother, the child's father is set
to the partner of the mother, ... Finally some variables of the child are set to
specific initial values: the most important of these is its gender, which is the
result of a simple choice process.

**new** is not limited to items of the same entity; the below procedure
*get a life* makes sure that all those who are single when they are 24 year old,
leave their parents’ household for their own household. The region of this
household is created through a simple choice-process.

*example 2* ::

    get_a_life:
        - household_id:
            if((age == 24) and (civilstate != 2) and (civilstate != 3),
               new('household',
                   start_period=period,
                   region_id=choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
               ),
               household_id)

.. index:: clone

clone
~~~~~

**clone** is very similar to **new** but is intended for cases where
most or all variables describing the new individual should be copied from
his/its parent/origin instead of being set to "missing". With clone, you cannot
specify what kind of entity you want to create, as it is always the same as the
origin item. However, similarly to **new**, **clone** also allows fields to be
specified manually by any expression evaluated on the parent/origin.

Put differently, a **new** with no fields mentioned will result in a new item
of which the initial values of the fields are all set to missing and have to be
filled through simulation; on the contrary, a **clone** with no fields mentioned
will result in a new item that is an exact copy of the origin except for its
id number which is always set automatically.

*example* ::

    make_twins:
        - clone(filter=new_born and is_twin,
                gender=choice([True, False], [0.51, 0.49]))

.. index:: remove

remove
~~~~~~

**remove** removes items from an entity dataset. With this command you can
remove obsolete items (eg. dead persons, empty households) thereby ensuring they
are not simulated anymore. This will also save some memory and, in some cases,
improve simulation speed.


The procedure below simulates whether an individual survives or not, and what
happens in the latter case. ::

    dead_procedure:
        # decide who dies
        - dead: if(gender,
                   logit_regr(0.0, align='al_p_dead_m.csv'),
                   logit_regr(0.0, align='al_p_dead_f.csv'))
        # change the civilstate of the suriving partner
        - civilstate: if(partner.dead, 5, civilstate)
        # break the link to the dead partner
        - partner_id: if(partner.dead, -1, partner_id)
        # remove the dead
        - remove(dead)

The first sub-procedure *dead* simulates whether an individual is ‘scheduled for
death’, using again only a logistic stochastic variable and the
age-gender-specific alignment process. Next some links are updated for the
surviving partner.
The sub-procedure *civilstate* puts the variable of that name equal to 5 (which
means that one is a widow(er) for those individuals whose partner has been
scheduled for death. Also, in that case, the partner identification code is
erased. All other procedures describing the heritage process should be included
here. Finally, the *remove* command is called to removes the *dead* from the
simulation dataset.

.. index:: matching

Matching functions
------------------

**matching**: (aka Marriage market) matches individuals from set 1 with
individuals from set 2. For each individual in set 1 following a particular
order (given by the expression in the *orderby* argument), the function computes
the score of all (unmatched) individuals in set 2 and take the best scoring one.

You have to specify the boolean filters which provide the two sets to match
(set1filter and set2filter), the criterion to decide in which order the
individuals of the first set are matched and the expression that will be used
to assign a score to each individual of the second set (given a particular
individual in set 1).

In the score expression the fields of the set 1 individual can be used normally
and the fields of its possible partners can be used by prefixing them by
"**other.**".

*generic setup* ::

    matching(set1filter=boolean_expr,
             set2filter=boolean_expr,
             orderby=difficult_match,
             score=coef1 * field1 + coef2 * other.field2 + ...)

The generic setup of the marriage market is simple; one needs to have selected
those individuals who are to be coupled (*to_couple*=true). Furthermore, one
needs to have a variable (*difficult_match*) which can be used to rank
individuals according how easy they are to match. Finally, we need a function
(*score*) matching potential partners.

In the first step, and for those persons that are selected to be coupled, potential partners are matched in the order set by
*difficult_match* and each woman is matched with the potential partner with the highest matching score. Once this is done,
both individuals become actual partners and the partner identification numbers are set so that the partner number of each
person equals the identification number of the partner.

*example* ::

    marriage:
        - in_couple: MARRIED or COHAB
        - to_couple: if((age >= 18) and (age <= 90) and not in_couple,
                        if(MALE,
                           logit_regr(0.0, align='al_p_mmkt_m.csv'),
                           logit_regr(0.0, align='al_p_mmkt_f.csv')),
                        False)
        - avg_age_males_to_couple: grpavg(age, filter=to_couple and MALE)
        - difficult_match: if(to_couple and FEMALE,
                              abs(age - avg_age_males_to_couple),
                              nan)
        - work: (workstate > 0) and (workstate < 5)
        - partner_id: if(to_couple,
                         matching(set1filter=FEMALE, set2filter=MALE,
                                  orderby=difficult_match,
                                  score=- 0.4893 * other.age
                                        + 0.0131 * other.age ** 2
                                        - 0.0001 * other.age ** 3
                                        + 0.0467 * (other.age - age)
                                        - 0.0189 * (other.age - age) ** 2
                                        + 0.0003 * (other.age - age) ** 3
                                        - 0.9087 * (other.work and not work)
                                        - 1.3286 * (not other.work and work)
                                        - 0.6549 * (other.work and work)),
                         partner_id)
        - coupled: to_couple and (partner_id != -1)
        - newhousehold: new('household', filter=coupled and FEMALE,
                            start_period=period,
                            region_id=choice([0, 1, 2, 3],
                                             [0.1, 0.2, 0.3, 0.4]))
        - household_id: if(coupled,
                           if(MALE, partner.newhousehold, newhousehold),
                           household_id)


The code above shows an application. First of all, individuals eligible for
marriage are all those between 18 and 90 who are not a part of a couple; the
actual decision who is eligible is left to the alignment process. Next, for
every women eligible to coupling, the variable *difficult_match* is the
difference between her age and the average age of men eligible for coupling.

In a third step, for each eligible woman in turn (following the order set by
*difficult_match*), all eligited men are assigned a score and the man with the
best score is matched with that woman. This score depends on his age, his
difference in age with the woman and the the work status of the potential
partners.

In a next step, a new household is created for women who have just become a part
of a couple. Their household number, as well as their new partners is then
updated to reflect their new household.


Output
======

LIAM 2 produces simulation output in three ways. First of all, by default, the
simulated datasets are stored in hdf5 format. These can be accessed at the end
of the run. You can use several tools to inspect the data.

You can display information during the simulation using *show* or *groupby*. You
can *dump* data to csv-file for further study.

If you run LIAM 2 in interactive mode, you can type in output functions in the
console to inspect the data.

.. index::  show

show
----

*show* evaluates expressions and prints the result to the console. ::

    show(expr1[, expr2, expr3, ...])


*example 1* ::

    show(grpcount(age >= 18))
    show(grpcount(not dead), grpavg(age, filter=not dead))

The first process will print out the number of persons of age 18 and older in
the dataset. The second one displays the number of living people and their
average age.

*example 2* ::

    show("Count:", grpcount(),
         "Average age:", grpavg(age),
         "Age std dev:", grpstd(age))

    gives

    Count: 19944 Average age: 42.7496991576 Age std dev: 21.9815913417

Note that you can use the special character "\n" to display the rest of the
result on the next line.

*example 3* ::

    show("Count:", grpcount(),
         "\nAverage age:", grpavg(age),
         "\nAge std dev:", grpstd(age))

    gives

    Count: 19944
    Average age: 42.7496991576
    Age std dev: 21.9815913417

.. index::  csv

csv
---

The **csv** function writes values to a csv-file.

    csv(expr1[, expr2, expr3, ...,
        suffix='file_suffix', fname='filename', mode='w'])

The suffix, fname and mode are optional arguments.

  - 'fname' allows defining the exact file names used. You can optionally use
    {entity} and {period} to customize the name.
  - 'suffix' allows to set the name of csv file more easily. If suffix is used,
    the filename will be: "{entity}_{period}_{suffix}.csv"

The default file name (if neither 'fname' nor 'suffix' is used) is  
"{entity}_{period}.csv".

*example* ::

    csv(grpavg(income), suffix='income')

will create one file for each simulated period. Assuming, start_period is
2002 and periods is 2, it will create two files: "person_2002_income.csv" and
"person_2003_income.csv" with the average income of the population for period
2002 and 2003 respectively.

   - 'mode' allows appending (mode='a') to a csv file instead of overwriting it
     (mode='w' by default). This allows you, for example, to store the value of
     some expression for all periods in the same file (instead of one file per
     period by default).

*example* ::

    csv(period, grpavg(income), fname='avg_income.csv', mode='a')

Note that unless you erase/overwrite the file one way or another between two
runs of a simulation, you will append the data of the current simulation to
that of the previous one. One way to do that automatically is to have a
procedure in the init section without mode='a' to overwrite the file.

If you want that file to start empty, you can do so this way: ::
    
    csv(fname='avg_income.csv')

If you want some headers in your file, you could write them at that point: ::
    
    csv('period', 'average income', fname='avg_income.csv')

When you use the csv() function in combination with (at least one) table
expressions (see dump and groupby functions below), the results are appended
below each other.

    csv(table_expr1, 'and here goes another table', table_expr2,
        fname='tables.csv')

Will produce a file with a layout like this: :: 

  | table 1 value at row 1, col 1 | col 2 |   ... | col N |
  |                           ... |   ... |   ... |   ... |
  |                  row N, col 1 | col 2 |   ... | col N |
  | and here goes another table   |       |       |       |
  | table 2 value at row 1, col 1 |   ... | col N |       |
  |                           ... |   ... |   ... |       |
  |                  row N, col 1 |   ... | col N |       |

You can also output several rows with a single command by enclosing values
between brackets: ::

    csv([row1value1, ..., row1valueN],
        ...
        [rowNvalue1, ..., rowNvalueN],
        fname='several_rows.csv')

*example* ::

    csv(['this is', 'a header'],
        ['with', 'several lines'],
        fname='person_age_aggregates.csv')

Will produce a file with a layout like this: :: 

| this is | a header      |
| with    | several lines |

.. index::  dump

dump
----

**dump** produces a table with the expressions given as argument evaluated over
many (possibly all) individuals of the dataset.

*general format*

    dump([expr1, expr2, ...,
          filter=filterexpression, missing=value, header=True])

If no expression is given, *all* fields of the current entity will be dumped
(including temporary variables available at that point), otherwise, each
expression will be evaluated on the objects which satisfy the
filter and produce a table.

The 'filter' argument allows to evaluate the expressions only on the individuals
which satisfy the filter. Defaults to None (evaluate on all individuals).

The 'missing' argument can be used to transform 'nan' values to another value.
Defaults to None (no transformation).

The 'header' argument determine whether column names should be in the dump or
not. Defaults to True.


*example* ::

    show(dump(age, partner.age, gender, filter=id < 10))

gives ::

    id | age | partner.age | gender
     0 |  27 |          -1 |  False
     1 |  86 |          71 |  False
     2 |  16 |          -1 |   True
     3 |  19 |          -1 |  False
     4 |  27 |          21 |  False
     5 |  89 |          92 |   True
     6 |  59 |          61 |   True
     7 |  65 |          29 |  False
     8 |  38 |          35 |   True
     9 |  48 |          52 |   True

.. index::  groupby

groupby
-------

**groupby** (aka *pivot table*): group all individuals by their value for the
given expressions, and optionally compute an expression for each group. If no
expression is given, it will compute the number of individuals in that
group. A filter can be specified to limit the individuals taken into account. 

*general format* ::

    groupby(expr1[, expr2, expr3, ...] [, expr=expression]
            [, filter=filterexpression] [, percent=True])

*example* ::

    show(groupby(age / 10, gender))

gives ::

        gender | False | True |
    (age / 10) |       |      | total
             0 |   818 |  803 |  1621
             1 |   800 |  800 |  1600
             2 |  1199 | 1197 |  2396
             3 |  1598 | 1598 |  3196
             4 |  1697 | 1696 |  3393
             5 |  1496 | 1491 |  2987
             6 |  1191 | 1182 |  2373
             7 |   684 |  671 |  1355
             8 |   369 |  357 |   726
             9 |   150 |  147 |   297
         total | 10002 | 9942 | 19944

*example* ::

    show(groupby(inwork, gender))

gives ::

    gender | False | True |
    inwork |       |      | total
     False |  6170 | 5587 | 11757
      True |  3832 | 4355 |  8187
     total | 10002 | 9942 | 19944

*example* ::

    show(groupby(inwork, gender, percent=True))

gives ::

    gender | False |  True |
    inwork |       |       |  total
     False | 30.94 | 28.01 |  58.95
      True | 19.21 | 21.84 |  41.05
     total | 50.15 | 49.85 | 100.00

*example* ::

    groupby(workstate, gender, expr=grpavg(age))

gives the average age by workstate and gender ::

       gender | False |  True |      
    workstate |       |       | total
            1 | 41.29 | 40.53 | 40.88
            2 | 40.28 | 44.51 | 41.88
            3 |  8.32 |  7.70 |  8.02
            4 | 72.48 | 72.27 | 72.38
            5 | 42.35 | 46.56 | 43.48
        total | 42.67 | 42.38 | 42.53

.. index::  interactive console

Interactive console
===================

LIAM 2 features an interactive console which allows you to interactively explore
the state of the memory either during or after a simulation completed.

You can reach it in two ways. You can either pass "-i" as the last argument when
running the executable, in which case the interactive console will launch after
the whole simulation is over. The alternative is to use breakpoints in your
simulation to interrupt the simulation at a specific point (see below).

Type "help" in the console for the list of available commands. In addition to
those commands, you can type any expression that is allowed in the simulation
file and have the result directly. Show is implicit for all operations.

*examples* ::

    >>> grpavg(age)
    53.7131819615

    >>> groupby(age / 20, gender, expr=grpcount(inwork))

        gender | False | True |
    (age / 20) |       |      | total
             0 |    14 |   18 |    32
             1 |   317 |  496 |   813
             2 |   318 |  258 |   576
             3 |    40 |  102 |   142
             4 |     0 |    0 |     0
             5 |     0 |    0 |     0
         total |   689 |  874 |  1563

.. index::  breakpoint

breakpoint
----------

**breakpoint**: temporarily stops execution of the simulation and launch the
interactive console. There are two additional commands available in the
interactive console when you reach it through a breakpoint: "step" to execute
(only) the next process and "resume" to resume normal execution.

*general format*

    breakpoint([period])

    the "period" argument is optional and if given, will make the breakpoint
    interrupt the simulation only for that period.

*example* ::

    marriage:
        - in_couple: MARRIED or COHAB
        - breakpoint(2002)
        - ...

