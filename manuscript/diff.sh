#!/usr/bin/env bash
#
# Make a latexdiff of the current manuscript.
#
# NOTES:
# - Must be run from a directory inside the git repo.
# - Produces a file "diff.pdf" containing the diff.
# - Removes blue highlighting in MNRAS links
#   (which conflicts with blue highlighting from diff).

################################################################################
# EDIT AS REQUIRED
################################################################################

# DIR is the directory in which the manuscript lives.
# This must be specified relative to the git repository root.
DIR=manuscript

# FILENAME is the manuscript filename.
FILENAME=multigrain-paper.tex

# COMMIT is the commit hash of the earlier version.
# Note: get the commit hash of a TAG with:
# $ git rev-parse TAG
COMMIT=e4b704197b882c6f49560ebbea9f0aabbd844e9e

################################################################################
# DO NOT EDIT BELOW
################################################################################

cd $(git rev-parse --show-toplevel)/${DIR} || return
git show ${COMMIT}:${DIR}/${FILENAME} > previous.tex
latexdiff previous.tex ${FILENAME} > diff.tex
rm previous.tex
(( LN=$(grep -n 'documentclass' diff.tex | cut -f1 -d:) + 1 ))
awk -v n=${LN} -v s="\\\hypersetup{colorlinks=false}" 'NR == n {print s} {print}' diff.tex > tmp
mv tmp diff.tex
latexmk diff.tex
latexmk -c diff.tex
rm diff.tex
cd - || return
