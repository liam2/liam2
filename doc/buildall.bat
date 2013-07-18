@echo off
rem change directory to the location of this batch file
rem (but remember original directory)
pushd %~dp0
rem pushd reference
rem call buildall.bat
rem popd
pushd usersguide
call buildall.bat
popd
echo
echo Build finished: all documentation built
popd