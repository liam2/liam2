@echo off

rem for Python 2.6-3.2
rem use Microsoft Windows SDK for Windows 7 and .NET Framework 3.5 SP1
rem you can get the "web" installer at: 
rem http://www.microsoft.com/en-us/download/details.aspx?id=3138
rem or the full "iso" at:
rem http://www.microsoft.com/en-us/download/details.aspx?id=18950
call "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat"

rem for Python 3.3+
rem use Microsoft Windows SDK for Windows 7 and .NET Framework 4
rem you can get the "web" installer at: 
rem https://www.microsoft.com/en-us/download/details.aspx?id=8279
rem or the full "iso" at:
rem https://www.microsoft.com/en-us/download/details.aspx?id=8442

rem tell distutils to use the SDK to compile
set DISTUTILS_USE_SDK=1
set MSSdk=1
