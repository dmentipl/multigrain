#
# Makefile to run Phantom multigrain tests and perform analysis.

.PHONY: run analysis

run: dustybox-run
analysis: dustybox-analysis

# ----------------------------------------------------------------------
# DUSTYBOX

.PHONY: dustybox-run dustybox-analysis

dustybox-run:
	@python DUSTYBOX/dustybox_setup.py \
		--run_root_dir ~/runs/multigrain/dustybox \
		--parameters DUSTYBOX/*.toml
	@python DUSTYBOX/dustybox_run.py \
		--run_root_dir ~/runs/multigrain/dustybox

dustybox-analysis:
	@python DUSTYBOX/dustybox_analysis.py \
		--run_root_dir ~/runs/multigrain/dustybox