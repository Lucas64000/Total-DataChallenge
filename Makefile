# PhotoTrap Wildlife Analysis System - Makefile

.PHONY: install install-dev test lint format clean docker-build docker-run dev help

# Default target
help:
	@echo "PhotoTrap Wildlife Analysis System"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run tests with coverage"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (black + isort)"
	@echo "  make clean        Remove cache and build files"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo "  make dev          Start development server"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=app --cov=models --cov=pipeline --cov=database --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

# Linting & Formatting
lint:
	ruff check .
	mypy app models pipeline database --ignore-missing-imports

format:
	black .
	isort .
	ruff check --fix .

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Docker
docker-build:
	docker build -t phototrap -f docker/Dockerfile .

docker-run:
	docker run -p 8000:8000 --gpus all phototrap

docker-compose-up:
	docker-compose -f docker/docker-compose.yml up

docker-compose-down:
	docker-compose -f docker/docker-compose.yml down

# Development
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Inference commands
inference:
	python -m pipeline.inference --image $(IMAGE)

batch:
	python -m pipeline.batch_processor --input-dir $(INPUT) --output $(OUTPUT)

# Database commands
build-index:
	python -m database.species_db build --images data/labelized/images/

# ONNX export
export-onnx:
	python -m models.onnx_export --output models/onnx/
