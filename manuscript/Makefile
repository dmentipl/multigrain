#
# Makefile to build the manuscript.

.PHONY: manuscript clean

manuscript: multigrain-paper.pdf

diff:
	@bash diff.sh

multigrain-paper.pdf: multigrain-paper.tex
	@latexmk multigrain-paper.tex

clean:
	@latexmk -c
