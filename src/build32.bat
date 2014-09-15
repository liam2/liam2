@echo off

rem if your python version is already in your PATH, you could just do:
rem call setenv32.bat
rem python setup.py build

rem However, this script uses conda (http://conda.pydata.org/miniconda.html)
rem to build LIAM2 using an environment.

rem This script assumes:
rem 1) you have conda installed with an environment called "liam2" with all
rem    LIAM2 dependencies installed
rem 2) you have a script in your PATH called c32.bat which setups all
rem    environment variables so that a 32-bit conda is in your PATH and
rem    everything is ready to compile C extensions using a 32-bit compiler
rem    (see setenv32.bat)
call c32.bat
call activate liam2

rem cxfreezing an app which loads .ui files fails
rem https://bitbucket.org/anthony_tuininga/cx_freeze/issue/103
rem cx-freeze 4.3.4 should fix this, but in the meantime:
rem rm -rf %ANACONDA_ENVS%\%CONDA_DEFAULT_ENV%\Lib\site-packages\PyQt4\uic\port_v3\
python setup.py build