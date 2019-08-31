#
# Makefile to build the manuscript.

.PHONY: manuscript clean

manuscript: multigrain-paper.pdf

multigrain-paper.pdf: multigrain-paper.tex
	latexmk multigrain-paper.tex

clean:
	latexmk -c
