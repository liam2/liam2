@echo off

rem if your python version is already in your PATH, you could just do:
rem call setenv64.bat
rem python setup.py build

rem However, this script uses conda (http://conda.pydata.org/miniconda.html)
rem to build LIAM2 using an environment.

rem This script assumes:
rem 1) you have conda installed with an environment called "liam2" with all
rem    LIAM2 dependencies installed
rem 2) you have a script in your PATH called c64.bat which setups all
rem    environment variables so that a 64-bit conda is in your PATH and
rem    everything is ready to compile C extensions using a 64-bit compiler
rem    (see setenv64.bat)
call c64.bat
call activate liam2
python setup.py build