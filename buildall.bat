@echo off

pushd %~dp0

pushd doc
call buildall
popd

pushd src
call buildall
popd

popd