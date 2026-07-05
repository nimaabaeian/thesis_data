.PHONY: build execute check clean

NOTEBOOK := analysis/orexigenic_analysis.ipynb
BUILDER := analysis/build_notebook.py

build:
	python $(BUILDER) $(NOTEBOOK)

execute: build
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --inplace --ExecutePreprocessor.timeout=900

check:
	python -m py_compile $(BUILDER)
	python analysis/check_notebook.py $(NOTEBOOK)

clean:
	rm -rf .vscode .claude
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) -delete
