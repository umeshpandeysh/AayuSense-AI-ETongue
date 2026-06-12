# AayuSense-AI-ETongue — Project Makefile
# Usage: make <target>

.PHONY: help install test lint train-rf train-xgb dashboard clean

PYTHON ?= python
DATA    ?= data/raw/sample_sensor_data.csv
MODEL   ?= rf

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:       ## Install all dependencies
	pip install -r requirements.txt

test:          ## Run the full test suite
	pytest tests/ -v --tb=short

lint:          ## Run flake8 linter
	flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503

train-rf:      ## Train Random Forest on sample data
	$(PYTHON) scripts/train.py --data $(DATA) --model rf --output models/ --report artifacts/

train-xgb:     ## Train XGBoost on sample data
	$(PYTHON) scripts/train.py --data $(DATA) --model xgb --output models/ --report artifacts/

dashboard:     ## Launch the Streamlit development dashboard
	streamlit run dashboard/app.py

clean:         ## Remove generated artifacts
	rm -rf artifacts/*.json models/*.joblib logs/*.log
	@echo "Cleaned generated files."
