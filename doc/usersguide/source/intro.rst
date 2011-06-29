.. highlight:: YAML

Introduction
############

LIAM 2 is a tool to develop (different kinds of) microsimulation models.

About LIAM2
===========

The goal of the project is to let modellers concentrate on what is strictly
specific to their model without having to worry about the technical details.
This is achieved by providing a generic microsimulation toolbox which is not
tied to a particular model. By making it available for free, our hope is to
greatly reduce the development costs (in terms of both time and money) of
microsimulation models.

The toolbox is made as generic as possible so that it can be used to develop
almost any microsimulation model as long as it use cross-sectional ageing, ie
all individuals are simulated at the same time for one period, then for the next
period, etc.

You can find the latest version of LIAM2 and this documentation at:
http://liam2.plan.be

About this guide
================

This guide will help you develop dynamic microsimulation models using LIAM 2. 
Please note that it describes version |version| of LIAM 2, but both the software
package and this manual are very much work-in-progress, and are therefore
subject to change, including in the syntax described in this manual for defining
models.

Microsimulation
===============

Microsimulation is (as defined by the International Microsimulation
Association), a modelling technique that operates at the level of individual
units such as persons, households, vehicles or firms. Each unit has a set of
associated attributes – e.g. each person in the model has an associated age,
sex, marital and employment status. At each time step, a set of rules (intended
to represent individual preferences and tendencies) are applied to these units
leading to simulated changes in state and possibly behaviour. These rules may be
deterministic (probability = 1), such as ageing, or stochastic 
(probability < 1), such as the chance of dying, marrying, giving birth or moving
within a given time period.

The aim of such simulations is to give insight about both the overall aggregate
change of some characteristics and (importantly) the way these changes are
distributed in the population that is being modelled. 

Credits
=======

LIAM2 is being developed at the Federal Planning Bureau (Belgium), with funding
and testing by CEPS/INSTEAD (Luxembourg) and IGSS (Luxembourg), and funding from
the European Community. It is the spiritual successor of LIAM 1, developed by
Cathal O’Donoghue.

More formally, it is part of the MiDaL project, supported by the European
Community Programme for Employment and Social Solidarity - PROGRESS (2007-2013),
under the Grant VS/2009/0569 Abstract - Project PROGRESS MiDaL deliverable Work
Package A.