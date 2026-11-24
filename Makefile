PYTHON ?= python3
CONFIG_DIR ?= configs
OUTPUT_DIR ?= outputs

.PHONY: data features experiments phase2 report test lint all

data:
	PYTHONPATH=src $(PYTHON) -m aml_graph data --config-dir $(CONFIG_DIR) --output-dir $(OUTPUT_DIR)

features:
	PYTHONPATH=src $(PYTHON) -m aml_graph features --config-dir $(CONFIG_DIR) --output-dir $(OUTPUT_DIR)

experiments:
	PYTHONPATH=src $(PYTHON) -m aml_graph experiments --config-dir $(CONFIG_DIR) --output-dir $(OUTPUT_DIR)

phase2:
	PYTHONPATH=src $(PYTHON) -m aml_graph phase2 --config-dir $(CONFIG_DIR) --output-dir $(OUTPUT_DIR)

report:
	PYTHONPATH=src $(PYTHON) -m aml_graph report --config-dir $(CONFIG_DIR) --output-dir $(OUTPUT_DIR)

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	ruff check src tests

all: data features experiments report
