@echo off
rem change directory to the location of this batch file
rem (but remember original directory)
pushd %~dp0

call c64.bat
call activate liam2

rem pushd reference
rem call buildall.bat
rem popd

pushd usersguide
call buildall.bat
popd

rem we should NOT build the website at this point because we need to build it
rem after the bundles are created so that we know their size
rem pushd website
rem call buildall.bat
rem popd

echo
echo Build finished: all documentation built
popd