#
# Makefile to run Phantom multigrain tests.

.PHONY: run

run: dustybox_time_evolution dustybox_accuracy dustywave dustyshock

# ----------------------------------------------------------------------

.PHONY: dustybox_time_evolution dustybox_accuracy dustywave dustyshock

dustybox_time_evolution:
	@python scripts/dustybox_time_evolution.py

dustybox_accuracy:
	@python scripts/dustybox_accuracy.py

dustywave_benitez_llambay:
	@python scripts/dustywave_benitez_llambay.py

dustyshock_benitez_llambay:
	@python scripts/dustyshock_benitez_llambay.py
