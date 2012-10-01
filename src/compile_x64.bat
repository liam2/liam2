@echo off

rem use Microsoft Windows SDK for Windows 7 and .NET Framework 3.5 SP1
rem you can get the "web" installer at: 
rem http://www.microsoft.com/en-us/download/details.aspx?id=3138
rem or, if you prefer, the full "iso" is at:
rem http://www.microsoft.com/en-us/download/details.aspx?id=18950
call "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars64.bat"

rem tell distutils to use the SDK to compile
set DISTUTILS_USE_SDK=1
set MSSdk=1

rem make sure we use the correct version of python
set PYTHONROOT=c:\soft\Python27
set PYTHONPATH=%PYTHONROOT%\Lib\site-packages\;

%PYTHONROOT%\python.exe build_cython.py