# Makefile for Malware Classification Project

.PHONY: help install install-dev clean test lint format run run-synthetic setup-dirs

# Default target
help:
	@echo "Available commands:"
	@echo "  install       - Install project dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  setup-dirs    - Create necessary directories"
	@echo "  clean         - Clean up generated files"
	@echo "  test          - Run unit tests"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black"
	@echo "  run           - Run full pipeline with real data"
	@echo "  run-synthetic - Run full pipeline with synthetic data"
	@echo "  run-fast      - Run pipeline with reduced samples"
	@echo "  download      - Download malware samples only"
	@echo "  process       - Process data only"
	@echo "  train         - Train models only"
	@echo "  visualize     - Generate visualizations only"
	@echo "  notebook      - Start Jupyter notebook server"

# Installation targets
install:
	pip install -r requirements/requirements.txt

install-dev:
	pip install -r requirements/requirements.txt
	pip install -r requirements/requirements-dev.txt

install-gpu:
	pip install -r requirements/requirements.txt
	pip install -r requirements/requirements-gpu.txt

# Setup
setup-dirs:
	mkdir -p data/raw data/processed data/external
	mkdir -p models/random_forest models/decision_tree models/svm models/ensemble
	mkdir -p results/benchmarks results/reports results/visualizations/plots results/predictions
	mkdir -p logs
	@echo "Directory structure created successfully!"

# Pipeline execution targets
run:
	python scripts/run_full_pipeline.py --max-samples 1000

run-synthetic:
	python scripts/run_full_pipeline.py --use-synthetic --max-samples 1000

run-fast:
	python scripts/run_full_pipeline.py --use-synthetic --max-samples 500

run-tune:
	python scripts/run_full_pipeline.py --use-synthetic --tune-hyperparameters --max-samples 500

# Individual step targets
download:
	python scripts/run_full_pipeline.py --steps download --max-samples 1000

process:
	python scripts/run_full_pipeline.py --steps process

train:
	python scripts/run_full_pipeline.py --steps train

visualize:
	python scripts/run_full_pipeline.py --steps visualize

# Individual script targets
download-data:
	python scripts/download_data.py

preprocess-data:
	python scripts/preprocess_data.py

train-models:
	python scripts/train_models.py

evaluate-models:
	python scripts/evaluate_models.py

generate-viz:
	python scripts/generate_visualizations.py

# Development targets
test:
	python -m pytest tests/ -v --cov=src --cov-report=html

test-quick:
	python -m pytest tests/ -x -v

lint:
	flake8 src/ scripts/ tests/
	mypy src/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# Jupyter notebook
notebook:
	jupyter notebook notebooks/

# Cleaning targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "Cleaned up generated files!"

clean-data:
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/external/*
	@echo "Cleaned up data files!"

clean-models:
	rm -rf models/*/
	@echo "Cleaned up model files!"

clean-results:
	rm -rf results/*/
	@echo "Cleaned up result files!"

clean-all: clean clean-data clean-models clean-results
	@echo "Complete cleanup performed!"

# Docker targets (if using Docker)
docker-build:
	docker build -t malware-classifier .

docker-run:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/results:/app/results malware-classifier

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "API documentation available in docs/api_reference.md"
	@echo "Usage guide available in docs/usage_guide.md"

# Validation
validate-setup:
	@echo "Validating project setup..."
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('Core libraries: OK')"
	@python -c "import pefile; print('PE analysis library: OK')"
	@python -c "import requests; print('HTTP library: OK')"
	@echo "Setup validation complete!"

# Performance monitoring
monitor:
	@echo "System resource monitoring..."
	@python -c "import psutil; print(f'CPU cores: {psutil.cpu_count()}')"
	@python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total // (1024**3)} GB')"
	@python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available // (1024**3)} GB')"

# Project status
status:
	@echo "Project Status:"
	@echo "==============="
	@echo "Data files:"
	@ls -la data/raw/ 2>/dev/null || echo "  No raw data files"
	@ls -la data/processed/ 2>/dev/null || echo "  No processed data files"
	@echo ""
	@echo "Trained models:"
	@ls -la models/*/*.pkl 2>/dev/null || echo "  No trained models"
	@echo ""
	@echo "Results:"
	@ls -la results/benchmarks/ 2>/dev/null || echo "  No benchmark results"
	@ls -la results/visualizations/plots/ 2>/dev/null || echo "  No visualization plots"

# Quick start for new users
quickstart: setup-dirs install run-synthetic
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo ""
	@echo "Your malware classification project is ready!"
	@echo "Check the following directories:"
	@echo "  - results/visualizations/plots/ for charts"
	@echo "  - results/reports/ for detailed reports"
	@echo "  - models/ for trained models"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make run' to use real malware data"
	@echo "  - Run 'make notebook' to explore with Jupyter"
	@echo "  - Check 'make help' for more commands"

# Benchmark run
benchmark:
	@echo "Running benchmark with multiple configurations..."
	python scripts/run_full_pipeline.py --use-synthetic --max-samples 500
	python scripts/run_full_pipeline.py --use-synthetic --max-samples 1000 --tune-hyperparameters
	@echo "Benchmark complete! Check results/benchmarks/ for comparison"