# Contributing to CellFlow

Contributions are welcome, whether bug reports, feature requests, documentation,
or code.

## Reporting issues

Please open an issue on the [GitHub issue tracker](https://github.com/samanseifi/cellflow/issues).
For bugs, include a minimal configuration that reproduces the problem, the
expected and observed behaviour, and your platform / Python version.

## Development setup

```bash
git clone https://github.com/samanseifi/cellflow
cd cellflow
pip install -e ".[test]"
pytest
```

## Pull requests

1. Fork the repository and create a feature branch.
2. Keep changes focused; match the style and structure of the surrounding code.
3. Add or update tests for any new physics or behaviour change. The existing
   suite includes analytic solver benchmarks, convergence checks, and
   reproducibility tests — new numerical features should come with comparable
   verification.
4. Ensure `pytest` passes locally before opening the PR.
5. Update the technical and/or user manual (`docs/`) when behaviour or
   configuration options change.

## Getting help

For questions about the model or numerics, open a discussion or issue. The
technical manual (`docs/technical_manual.pdf`) documents the governing equations
and numerical methods.
