.PHONY: format lint typecheck all install

format:
	black . --exclude venv
	isort . --skip venv

lint:
	ruff check . --fix --exclude venv

typecheck:
	mypy --explicit-package-bases .

all: format lint typecheck

install:
	pip install -r requirements.txt -r requirements-dev.txt
