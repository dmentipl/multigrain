#
# Makefile to run Phantom multigrain tests and perform analysis.

.PHONY: run analysis

run: dustybox-run
analysis: dustybox-analysis

# ----------------------------------------------------------------------
# dustybox

.PHONY: dustybox-run

dustybox-run:
	@python dustybox/run.py --run_directory ~/runs/multigrain/dustybox
