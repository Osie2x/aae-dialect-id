.PHONY: install test run verify clean

install:
	pip install -r requirements.txt && pip install -e .

test:
	python -m pytest -q

run:
	python -m aaeid.run --config configs/default.yaml

# Rerun into a scratch folder and confirm the results are byte-identical.
verify:
	python -m aaeid.run --config configs/default.yaml --out results/_verify
	diff results/default/results.json results/_verify/results.json && \
	diff results/default/folds.csv results/_verify/folds.csv && echo "REPRODUCED: identical results"

clean:
	rm -rf results/_verify .pytest_cache
