@echo off

@rem setup.py install build --compiler=mingw32

rem make sure we use the correct version of python
set PYTHONROOT=c:\soft\Python27-32b 
set PYTHONPATH=%PYTHONROOT%\Lib\site-packages\;

%PYTHONROOT%\python.exe build_cython.py