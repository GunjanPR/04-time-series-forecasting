# Windows: install make with `choco install make`, or run from Git Bash.
PYTHON ?= python

.PHONY: install lint format test data run run-electricity report dashboard up down

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

report:                     # significance tables + plots -> docs/results/<dataset>/
	for d in m4-monthly electricity-daily; do \
		$(PYTHON) -m ts_forecasting.report --dataset $$d --publish docs/results/$$d && \
		$(PYTHON) -m ts_forecasting.analysis --dataset $$d --publish docs/results/$$d && \
		$(PYTHON) -m ts_forecasting.analysis --dataset $$d --simple lgbm --challenger comb \
			--publish docs/results/$$d; \
	done

run-electricity:
	$(PYTHON) -m ts_forecasting.benchmark --dataset electricity-daily

dashboard:
	$(PYTHON) -m streamlit run src/ts_forecasting/dashboard.py

up:
	docker compose up --build

down:
	docker compose down
