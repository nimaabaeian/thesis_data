.PHONY: all build execute check test manifest verify-manifest check-constants repro clean clean-outputs

NOTEBOOK := analysis/orexigenic_analysis.ipynb
BUILDER  := analysis/build_notebook.py
PY       := python

## all: full clean regeneration — the only supported way to produce the reported numbers.
all: clean-outputs manifest execute check test repro
	@echo
	@echo "Regenerated from a clean state. Every number in analysis/outputs/ and the README"
	@echo "came from this run. See analysis/outputs/reproducibility_report.md."

build:
	$(PY) $(BUILDER) $(NOTEBOOK)

## execute: run every cell top-to-bottom, in place. Outputs are regenerated from scratch.
execute: build
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --inplace \
		--ExecutePreprocessor.timeout=1800

## check: notebook compiled, fully executed, error-free, and leaks no participant identity.
check:
	$(PY) -m py_compile $(BUILDER)
	$(PY) analysis/check_notebook.py $(NOTEBOOK)

## test: regression tests pinning each corrected defect.
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest analysis/tests/ -q

## manifest: hash the raw data + private maps so a holder of the data can confirm they match.
manifest:
	$(PY) analysis/make_manifest.py

verify-manifest:
	$(PY) analysis/make_manifest.py --verify

## check-constants: verify CONST against the pinned controller source (SKIPPED if unavailable).
check-constants:
	$(PY) analysis/check_constants.py

## repro: input hashes, software versions, execution status, output hashes.
repro:
	$(PY) analysis/repro_report.py

## clean-outputs: delete every generated artifact, so `execute` cannot inherit a stale number.
clean-outputs:
	rm -rf analysis/outputs analysis/figures analysis/cache
	mkdir -p analysis/outputs analysis/figures analysis/cache

clean:
	rm -rf .vscode .claude
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints -o -name .pytest_cache \) \
		-prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) -delete
