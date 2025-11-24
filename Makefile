PYTHON := python3
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
REQ := common/python/requirements.txt

.PHONY: generate-test-cases venv install

generate-test-cases: venv install
	@echo "Running test case generator..."
	@$(VENV_BIN)/python common/python/t5_testkit/main.py

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists."; \
	fi

install: venv
	@echo "Installing requirements..."
	@$(VENV_BIN)/pip install -r $(REQ)
