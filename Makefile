.PHONY: install setup clean uninstall help

# Define colors for output
GREEN := \033[0;32m
NC := \033[0m
BLUE := \033[0;34m

help: ## Show this help message
	@echo "$(BLUE)Available make targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Set up Python environment and packages
	@bash install.sh

setup: ## Download datasets (FB15k-237 and WN18RR)
	@bash basic_setup.sh

uninstall: ## Removes virtual environments, logs, and datasets
	@bash uninstall.sh

clean: ## Clean python cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist build *.egg-info
	rm -rf logs/*
	rm -rf checkpoints/*
	rm -rf datasets/*
	rm -rf runs/*
