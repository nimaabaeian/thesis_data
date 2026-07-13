.PHONY: all build execute check test manifest verify-manifest check-constants repro \
        clean clean-outputs determinism ci

NOTEBOOK := analysis/orexigenic_analysis.ipynb
BUILDER  := analysis/build_notebook.py
PY       := python
PYTEST   := PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest

# Path to a checkout of the controller repo. Constants are verified against the PINNED
# deployment commits inside it. Override on the command line if it lives elsewhere:
#     make all CONTROLLER_SRC=/path/to/social-robot-embodied-behaviour-architecture
CONTROLLER_SRC ?= $(HOME)/Desktop/social-robot-embodied-behaviour-architecture

## all: the acceptance build. Order matters and is enforced.
##
##   clean-outputs -> verify-manifest -> check-constants -> execute -> check -> test -> repro
##
## verify-manifest runs BEFORE execute and does NOT regenerate the manifest: a build that
## quietly re-hashes whatever inputs happen to be on disk and then declares them correct has
## verified nothing. If the inputs changed, the build fails and you re-issue `make manifest`
## deliberately.
##
## repro runs LAST, after every other artifact exists, so the hashes it records are the hashes
## of this build.
all: clean-outputs verify-manifest check-constants execute check test repro
	@echo
	@echo "ACCEPTANCE BUILD PASSED."
	@echo "Every number in analysis/outputs/ and README.md came from this run."
	@echo "See analysis/outputs/reproducibility_report.md."

build:
	$(PY) -m py_compile $(BUILDER) analysis/statistical_helpers.py
	$(PY) $(BUILDER) $(NOTEBOOK)

## execute: run the notebook top-to-bottom into genuinely empty output dirs.
execute: clean-outputs build
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --inplace \
		--ExecutePreprocessor.timeout=1800

## check: notebook fully executed, error-free, and leaks no participant identity.
check:
	$(PY) analysis/check_notebook.py $(NOTEBOOK)

## test: regression tests. They import the SAME functions the notebook runs.
test:
	$(PYTEST) analysis/tests/ -q

## determinism: execute twice from clean and compare every generated artifact.
determinism:
	$(PYTEST) analysis/tests/test_determinism.py -q -m slow

## manifest: (re)hash the raw data. Deliberate — never run as part of `all`.
manifest:
	$(PY) analysis/make_manifest.py

## verify-manifest: fail if the inputs differ from the recorded manifest.
verify-manifest:
	$(PY) analysis/make_manifest.py --verify

## check-constants: every constant, key by key, at every pinned deployment commit. Fails hard.
check-constants:
	$(PY) analysis/check_constants.py --source "$(CONTROLLER_SRC)" \
		--json analysis/outputs/constants_check.json

## repro: input hashes, versions, execution status, output hashes — and verify they match.
repro:
	$(PY) analysis/repro_report.py --verify

## clean-outputs: actually delete every generated artifact, so nothing stale can survive.
clean-outputs:
	rm -rf analysis/outputs analysis/figures analysis/cache
	mkdir -p analysis/outputs analysis/figures analysis/cache

## ci: what the GitHub workflow runs (no raw data available there).
ci:
	$(PY) -m py_compile $(BUILDER) analysis/statistical_helpers.py
	$(PYTEST) analysis/tests/ -q
	$(PY) analysis/check_constants.py --source "$(CONTROLLER_SRC)"

clean:
	rm -rf .vscode .claude
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints -o -name .pytest_cache \) \
		-prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) -delete
