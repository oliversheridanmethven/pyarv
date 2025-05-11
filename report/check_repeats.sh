#!/bin/tcsh
# (use sh2doc.pl to auto generate HTML doc)
#
##BRIEF
# check textfile for repeated words using grep 
#
##AUTHOR
# Ronni Grapenthin
#
##DATE
# version 2011-01-27
#
##DETAILS
# check textfile for repeated words using grep 
# (regexp partly blatantly stolen off the web: http://www.codeproject.com/kb/dotnet/RegexTutorial.aspx?fid=136362&df=90&mpp=25&noise=3&sort=Position&view=Quick&fr=226)
# gives line number and marks repeated words, doesn't care about cases
#
# USAGE:
#
# usage: check_repeats <text-file>
#
##CHANGELOG
# 2010-06-13, ronni: First version.
# 2011-01-27, ronni: added checking over linebreaks, apparently this should be 
#                    possible and easier using sed; couldn't get it to work though
#		     I iterate over the file again, that's inefficient, but the output
#                    might be clearer.

echo "Checking for repeated words in a line of ${1}:"
grep -Ein --color  "\b(\w+)\b\s*\1\b" $1
echo " "

echo "Checking for repeated words over linebreaks of ${1}:"
awk 'BEGIN{getline l;} {combined=l " " $0; printf(" %.5d - %.5d: %s\n",FNR,FNR+1, combined); l=$0;}' $1 | grep -Ei --color  "\b(\w+)\b\s*\1\b"
echo " "
echo "Done."

#thank you very much!
