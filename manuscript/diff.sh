#!/usr/bin/env bash
#
# Make a latexdiff of the current manuscript

# This is the submitted commit hash
# Edit if you need a diff to another commit
COMMIT=e4b704197b882c6f49560ebbea9f0aabbd844e9e
# Note: get the commit hash of a tag with:
#   git rev-parse TAG

git show ${COMMIT}:manuscript/multigrain-paper.tex > multigrain-paper_previous.tex
latexdiff multigrain-paper_previous.tex multigrain-paper.tex > multigrain-paper_diff.tex
python diff.py
python << END
f = open("multigrain-paper_diff.tex", "r")
contents = f.readlines()
f.close()
contents.insert(44, "\hypersetup{colorlinks=false}")
f = open("multigrain-paper_diff.tex", "w")
contents = "".join(contents)
f.write(contents)
f.close()
END
latexmk multigrain-paper_diff.tex
latexmk -c multigrain-paper_diff.tex
rm multigrain-paper_diff.tex multigrain-paper_previous.tex
