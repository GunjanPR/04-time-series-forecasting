# Windows: install make with `choco install make`, or run from Git Bash.
PYTHON ?= python

.PHONY: install lint format test data run report dashboard up down

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,tracking,dashboard]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) -m ts_forecasting.data

run:
	$(PYTHON) -m ts_forecasting.benchmark

report:                     # significance tables + plots, published to docs/results
	$(PYTHON) -m ts_forecasting.report --publish docs/results
	$(PYTHON) -m ts_forecasting.analysis --publish docs/results
	$(PYTHON) -m ts_forecasting.analysis --simple lgbm --challenger comb --publish docs/results

dashboard:
	$(PYTHON) -m streamlit run src/ts_forecasting/dashboard.py

up:
	docker compose up --build

down:
	docker compose down
