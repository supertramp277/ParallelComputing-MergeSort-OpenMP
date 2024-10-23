@echo off
if not exist ..\bin ( mkdir ..\bin )
mingw32-make
..\bin\mergesort-co 200000000
