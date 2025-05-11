#!/bin/bash
set -eu 

./ipes_to_pdfs.sh

REPORT="report"
REPORT_TEX="${REPORT}.tex"
pdflatex $REPORT_TEX 
bibtex $REPORT
pdflatex $REPORT_TEX 
pdflatex $REPORT_TEX 

