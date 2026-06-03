@echo off
rem This script will compile C extensions in-place. It assumes that all
rem environment variables are set up so that everything is ready to compile C
rem extensions using the compiler of your choice.
python setup.py build_ext --inplace