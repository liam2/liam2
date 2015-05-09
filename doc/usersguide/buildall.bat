@echo off
call make web
call make html

call make htmlhelp
pushd build\htmlhelp
hhc.exe LIAM2UserGuide.hhp > hhc.log
popd

call make latex
pushd build\latex
rem we can also use --run-viewer to open the pdf
texify.exe --clean --pdf --tex-option=-synctex=1 LIAM2UserGuide.tex > texify.log
popd
