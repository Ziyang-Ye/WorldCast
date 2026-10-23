# Contributing

WorldCast reproduces the paper's latents bit for bit on the reference stack and plays them live. A change on the
generation path keeps the tests green and the fingerprints equal ([docs/inference.md](docs/inference.md), "Numerics").

```bash
pip install -e ".[dev]"                          # tests, black / isort / flake8 at CI's versions, pre-commit
python -m pytest tests -q                        # CPU; tests that need a GPU, the weights or the data skip and say why
black --check worldcast tools tests && isort --check-only worldcast tools tests
flake8 worldcast tools tests
```

- Code style, vocabulary and the repo map are in [docs/development.md](docs/development.md): Python 3.10 types,
  Google docstrings, black and isort at 100 columns, a comment only for a non-obvious reason.
- A new module is named in the repo map of docs/development.md and has its test in the mirrored folder of `tests/`
  (`tests/docs` checks the map, the docs' commands, settings and links against the code).
- Settings are config keys that default to the paper's values and are checked when the config is built; a setting
  that changes the latents is documented as not bit-equal.
- The data builders are `tools/prepare_data.py` steps ([docs/data.md](docs/data.md)); the live demo is
  [docs/demo.md](docs/demo.md).
