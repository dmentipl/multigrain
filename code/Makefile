#
# Makefile to run Phantom multigrain tests and perform analysis.

.PHONY: run
run: dustybox-run # dustywave-run dustyshock-run

.PHONY: analysis
analysis: dustybox-analysis # dustywave-analysis dustyshock-analysis

# ----------------------------------------------------------------------
# DUSTYBOX

.PHONY: dustybox-run
dustybox-run:
	python DUSTYBOX/dustybox_run.py

.PHONY: dustybox-analysis
dustybox-analysis:
	python DUSTYBOX/dustybox_analysis.py

# ----------------------------------------------------------------------
# DUSTYWAVE

.PHONY: dustywave-run
dustywave-run:
	python DUSTYWAVE/dustywave_run.py

.PHONY: dustywave-analysis
dustywave-analysis:
	python DUSTYWAVE/dustywave_analysis.py

# ----------------------------------------------------------------------
# DUSTYSHOCK

.PHONY: dustyshock-run
dustyshock-run:
	python DUSTYSHOCK/dustyshock_run.py

.PHONY: dustyshock-analysis
dustyshock-analysis:
	python DUSTYSHOCK/dustyshock_analysis.py
