#
# Makefile to setup and run Phantom multigrain tests

.PHONY: run-tests
run-tests: dustybox-run

.PHONY: analyse-tests
analyse-tests: dustybox-analyse

# ----------------------------------------------------------------------
# DUSTYBOX

.PHONY: dustybox-run
dustybox-run:
	python DUSTYBOX/dustybox_run.py

.PHONY: dustybox-analyse
dustybox-analyse:
	python DUSTYBOX/dustybox_analysis.py
