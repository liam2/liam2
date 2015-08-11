@echo off
rem This script will compile C extensions in-place. It assumes that all
rem environment variables are set up so that everything is ready to compile C
rem extensions using the compiler of your choice.
rem calling setenv32.bat or setenv64.bat before this script should be enough
rem if the correct version of Python is already in your PATH.
python setup.py build_ext --inplace