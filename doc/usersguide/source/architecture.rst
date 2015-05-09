Code architecture
#################

This document is meant for people who want to know more about the internals of
LIAM2, for example to add or modify some functionality. One should already be
familiar with how the program is used (on the user side). 

Concepts
========

Here is a brief description of the most important concepts to understand the
code of LIAM2, as well as where those concepts are implemented.

Simulation
----------

file: simulation.py

The *Simulation* class takes care of loading the simulation file (it delegates
much of the work to entities), prepares the data source and simulates each
period in turn (runs each process in turn).

Entity
------

file: entities.py

The *Entity* class stores all there is to know about each entity: fields,
links, processes and data. It serves as a glue class between everything
data, processes, ...

Process
-------

file: process.py

The *Process* class stores users processes. The most common kind of process 
is the *Assignment* which computes the value of an expression and
optionally stores the result in a variable.

Another very common process is the *ProcessGroup* (a.k.a procedures) which
runs a list of processes in order.

Expressions
-----------

file: expr.py (and many others)

Expressions are the meat of the code. The *Expr* class is the base class for
all expressions in LIAM2. It defines all the basic operators on expressions
(arithmetic, logical, comparison), but it should not be inherited from
directly.

file: exprbases.py

LIAM2 provides many different bases classes to inherit from when implementing
a new function:

* NumexprFunction: base class for functions which are implemented
  as-is in numexpr. eg. abs, log, exp

* CompoundExpression: base class for expressions which can be expressed in
  terms of other "liam2" expressions. eg. min, max, zeroclip

* EvaluableExpression: base class for all other expressions (those that do not
  exist in numexpr and cannot be expressed in terms of other liam2
  expressions). These expressions need to be pre-evaluated and stored in 
  a (hidden) temporary variable before being fed to numexpr, and this is what
  EvaluableExpression does. One should only inherit from this class directly
  if none of the below subclasses applies.

  a) NumpyFunction: subclass for functions which are implemented
     as is in Numpy. Should not be used directly.
     
     * NumpyCreateArray: subclass for functions which create arrays out of
       nothing (usually random functions).
     * NumpyChangeArray: subclass for functions which take an array as input
       and give another array as output (eg clip, round).
     * NumpyAggregate: subclass for aggregate functions. eg. count, min,
       max, std, median.

  b) FunctionExpr: subclass for functions. eg. trunc, lag, duration, ...

     * FilteredExpression: subclass for functions which have a filter
       argument and need to support contextual filters. eg. align, sum, avg,
       gini.

LIAM2 current expressions are implemented in the following files:

actions.py
    actions are expressions which do not have any result (that can be stored in
    variables), but have side-effects. Examples include: csv(), show(),
    remove(), breakpoint()

alignment.py
    handles align() and align_abs() functions

align_link.py
    the core algorithm (an implementation of Chenard's) for align_abs(link=)

exprmisc.py
    all expressions which are not defined in another file.
    
groupby.py
    handles groupby()

links.py
    contains all link-related code: 

    * the *Link* class stores the definition of links
    * the *LinkGet* class handles ManyToOne links
    * one class for each method of OneToMany links: *Count*, *Sum*, *Avg*,
      *Min* and *Max*.

matching.py
    handles the matching functions: matching() and rank_matching()

regressions.py
    handles all the regression functions: logit_score, logit_regr, cont_regr,
    clip_regr, log_regr

tfunc.py
    handles all time-related functions: value_for_period, lag, duration, tavg
    and tsum

Context
-------

file: context.py

A context is a data structure used to keep track of "contextual" information:
what is the "current" entity, what is the "current" period, what is the
"current" dataset. The context is passed around to the evaluation
functions/methods.

A context must present a simple dictionary interface (key: value). There are
a few keys with special meanings:

period
    should be the period currently being evaluated
__len__ 
    if present, should be an int representing the number of rows in the context 
__entity__
    current entity
__globals__
    if present, should be a dictionary of global tables ('periodic', ...)

The kind of context which is most used is the *EntityContext* which provides
a context interface to an Entity.


Other files
===========

Main code
---------

config.py
    Stores some global configuration variables

console.py
    Handles the interactive console

cpartition.pyx
    Cython source to speed up our partitioning function (group_indices_nd)
    which is used in groupby and alignment. 

cpartition.c
    generated from cpartition.pyx using Cython

cpartition.pyd
    cpartition.c compiled

cutils.pyx
    Cython source to speed up some commonly used utility functions. 

cutils.c
    generated from cutils.pyx using Cython

cutils.pyd
    cutils.c compiled

data.py
    handles loading, indexing, checking, merging, copying or modifying (adding
    or removing fields) tables (or subsets of them). It tries to provide a
    uniform interface from different data sources but it is a work in
    progress. 

exprtools.py
    parsing code for expressions

importer.py
    code to import csv files in our own hdf5 "subformat" by reading an
    "import file" (in yaml).

khash.h
    Generic hash table from Klib, used in cpartition.pyx
    see https://github.com/attractivechaos/klib

main.py 
    The main script. It reads command line arguments and calls the
    corresponding code (run, import, explore) in simulation.py (run/explore)
    or importer.py (import)

partition.py 
    handles partitioning objects depending on the possible values of their
    columns. 

utils.py
    miscellaneous support functions 

standalone scripts
------------------

diff_h5.py
    diff two liam2 files

dropfields_h5.py
    copy a subset of a liam2 file (excluding specified columns) 

filter_h5.py
    copy a subset of a liam2 file (all rows matching specified condition) 

merge_h5.py
    merge two liam2 files

build scripts
-------------

build_exe.py
    generic script to make executables (for standalones scripts)

setup.py
    compile cython extensions to pyd and make an .exe for the main liam2
    executable (using cx_Freeze)