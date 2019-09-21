#
# Makefile to run Phantom multigrain tests and perform analysis.

.PHONY: run analysis

run: dustybox-run
analysis: dustybox-analysis

# ----------------------------------------------------------------------
# DUSTYBOX

.PHONY: dustybox-run dustybox-analysis

dustybox-run:
	@python DUSTYBOX/dustybox_setup.py
	@python DUSTYBOX/dustybox_run.py

dustybox-analysis:
	@python DUSTYBOX/dustybox_analysis.py
