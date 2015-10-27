@echo off

rem if you just want to build LIAM2 for one version of Python and it is already
rem in your PATH, you could just do:
rem call setenv32.bat
rem python setup.py build

rem However, this script uses conda (http://conda.pydata.org/miniconda.html)
rem to build both 32b and 64b versions of LIAM2 at once and using
rem "virtual environments".

rem This script assumes:
rem 1) you have both a 32b and a 64b version of conda installed
rem 2) you have two scripts in your PATH called c32.bat and c64.bat which setup
rem    all environment variables so that the corresponding conda is in your PATH
rem    and everything is ready to compile C extensions using a 32-bit or 64-bit
rem    compiler respectively (see setenv32.bat and setenv64.bat)
rem 3) in both conda installations, you have an environment named "liam2"
rem    with all LIAM2 dependencies installed
rem 4) you have upx in your PATH

rem ======== 32 bits =========
call c32.bat
call activate liam2

rem cxfreezing an app which loads .ui files fails
rem https://bitbucket.org/anthony_tuininga/cx_freeze/issue/103
rem cx-freeze 4.3.4 should fix this, but in the meantime:
rem rd /s /q %ANACONDA_ENVS%\%CONDA_DEFAULT_ENV%\Lib\site-packages\PyQt4\uic\port_v3\
python setup.py build_exe

set BDIR32=build\exe.win32-2.7
upx -9 %BDIR32%\*.exe
upx -9 %BDIR32%\*.pyd
upx -9 %BDIR32%\*.dll
upx -9 %BDIR32%\imageformats\*.dll
rd /s /q %BDIR32%\mpl-data\sample_data

rem ======== 64 bits =========
call c64.bat
call activate liam2

rem rd /s /q %ANACONDA_ENVS%\%CONDA_DEFAULT_ENV%\Lib\site-packages\PyQt4\uic\port_v3\
python setup.py build_exe

set BDIR64=build\exe.win-amd64-2.7
rem upx does not support 64b files
rem upx -9 %BDIR64%\*.exe
rem upx -9 %BDIR64%\*.pyd
rem upx -9 %BDIR64%\*.dll
rem upx -9 %BDIR64%\imageformats\*.dll
rd /s /q %BDIR64%\mpl-data\sample_data
