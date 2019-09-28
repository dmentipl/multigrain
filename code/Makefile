#
# Makefile to run Phantom multigrain tests and perform analysis.

.PHONY: run analysis

run: dustybox-run
analysis: dustybox-analysis

# ----------------------------------------------------------------------
# dustybox

.PHONY: dustybox-run dustybox-analysis

dustybox-run:
	@python dustybox/cli.py setup \
		--run_root_dir ~/runs/multigrain/dustybox
	@python dustybox/cli.py run \
		--run_root_dir ~/runs/multigrain/dustybox

dustybox-analysis:
	@python dustybox/cli.py analysis \
		--run_root_dir ~/runs/multigrain/dustybox