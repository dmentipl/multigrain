#
# Makefile to run Phantom multigrain tests and perform analysis.

.PHONY: run analysis

run: dustybox-run
analysis: dustybox-analysis

# ----------------------------------------------------------------------
# dustybox

.PHONY: dustybox-run dustybox-analysis

dustybox-run:
	@python dustybox/dustybox_setup.py \
		--run_root_dir ~/runs/multigrain/dustybox \
		--parameters dustybox/parameters/*.toml
	@python dustybox/dustybox_run.py \
		--run_root_dir ~/runs/multigrain/dustybox

dustybox-analysis:
	@python dustybox/dustybox_analysis.py \
		--run_root_dir ~/runs/multigrain/dustybox