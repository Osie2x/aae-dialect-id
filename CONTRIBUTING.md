# Contributing

## Local setup

From the repository root, create and activate a virtual environment, then run:

```bash
make install
make test
```

Tests use synthetic fixtures and do not require restricted research inputs. See [the README](README.md#setup-and-reproduction) for complete setup instructions.

## Where changes belong

- Package code: `src/aaeid/`
- Experiment settings: `configs/`
- Synthetic tests: `tests/`
- Research methodology and data guidance: `docs/`
- Saved analysis artifacts: `results/`

## Review checklist

1. Run `make test` for code changes.
2. Document changes to features, preprocessing, folds, thresholds, or label handling.
3. Preserve the distinction between human-reviewed and model-labeled evidence.
4. Keep raw data, annotator personal information, credentials, and local environments out of commits.
5. If replacing saved results, record input hashes, configuration, and dependency versions, and explain the differences.

The MIT license covers code; it does not grant rights to redistribute the underlying datasets. See [the data guide](docs/DATA.md).
