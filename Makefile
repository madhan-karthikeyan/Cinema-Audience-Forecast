.PHONY: help install install-dev lint type test test-ci build run run-detached stop benchmark seed-history export-models clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:                        ## Install production dependencies
	pip install -r requirements.txt

install-dev: install            ## Install dev dependencies
	pip install -r requirements-dev.txt

lint:                           ## Run ruff linter
	ruff check app/ tests/

type:                           ## Run mypy type checker
	mypy app/

test:                           ## Run all tests
	python -m pytest tests/ -v --cov=app --cov-report=term-missing

test-ci:                        ## Run tests with CI coverage threshold
	python -m pytest tests/ -v --cov=app --cov-report=xml --cov-fail-under=80

build:                          ## Build Docker image
	docker build -f deployment/Dockerfile -t cinema-forecast-api:latest .

run:                            ## Run full stack with docker-compose
	docker compose -f deployment/docker-compose.yml up --build

run-detached:                   ## Run full stack in background
	docker compose -f deployment/docker-compose.yml up --build -d

stop:                           ## Stop all containers
	docker compose -f deployment/docker-compose.yml down

api:                            ## Run API server directly (no Docker)
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

benchmark:                      ## Run load tests (requires k6)
	k6 run k6/batch_forecast.js

seed-history:                   ## Initialize history store from training data
	python scripts/seed_history.py

export-models:                  ## Export and register trained models
	python scripts/export_models.py

clean:                          ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
