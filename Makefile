#
# Master Makefile to run Makefiles in subdirectories.

.PHONY: help
help:
	@echo "Makefile targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
        awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

.PHONY: manuscript
manuscript: ## Build the PDF of the manuscript.
	make -C manuscript manuscript

.PHONY: run
run: ## Set up and run the tests with Phantom
	make -C code run
