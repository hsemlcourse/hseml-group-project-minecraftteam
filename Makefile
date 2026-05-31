PYTHON ?= python

.PHONY: install lint test train predict experiments all

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m flake8 src tests

test:
	$(PYTHON) -m pytest -q

precommit:
	$(PYTHON) -m pre_commit run --all-files

train:
	$(PYTHON) src/repair_cost_cli.py train

predict:
	$(PYTHON) src/repair_cost_cli.py predict

experiments:
	$(PYTHON) src/repair_cost_experiments.py

all: lint test experiments train predict
