@echo off
call setenv32mingw.bat
%PYTHONROOT%\python.exe setup.py build_ext --inplace --compiler=mingw32