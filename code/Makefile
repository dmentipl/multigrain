#
# Makefile to run Phantom multigrain tests.

.PHONY: run

run: dustybox dustywave

# ----------------------------------------------------------------------
# dustybox

.PHONY: dustybox dustywave

dustybox:
	@python modules/dustybox/run.py --run_directory ~/runs/multigrain/dustybox

dustywave:
	@python modules/dustywave/run.py --run_directory ~/runs/multigrain/dustywave
