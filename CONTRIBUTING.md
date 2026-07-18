# Contributing to etree

Thanks for your interest! Bug reports, fixes, and focused improvements are
welcome. This is a header-only C++17 library (Eigen is the only dependency) with
optional Python bindings.

## Build and test

```sh
cmake -S . -B build && cmake --build build -j $(nproc)
ctest --test-dir build --output-on-failure
```

Tests use [doctest](https://github.com/doctest/doctest) (vendored) and live in
`tests/`. Please add a test for any behavior you change or add.

### Sanitizers

CI runs ASan+UBSan and TSan. To reproduce locally:

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j 2      # sanitizer builds are memory-hungry; keep -j small
ctest --test-dir build-asan --output-on-failure
```

### Python bindings

```sh
cmake -S . -B build -DETREE_BUILD_PYTHON=ON
cmake --build build --target etree_python
PYTHONPATH=build/bindings python3 -m pytest bindings/tests
```

Points are rows at the Python boundary (`(n, d)` arrays, scipy-style).

## Documentation

- The example pages under `docs/examples/` are **generated** from the programs
  in `examples/` by `python3 docs/generate_examples.py`; CI checks that the
  committed Markdown is current, so regenerate after changing an example.
- The Python notebook `examples/python_batch_picking.ipynb` is **re-executed**
  in CI (`pytest --nbmake`), so its code must run against the current bindings.
- Public API prose lives in the headers as Doxygen `///` comments and is
  published to GitHub Pages.

## Pull requests

- Branch from `main` and keep each PR focused.
- All CI must be green: Linux builds under g++ and clang++, the sanitizers, the
  Python bindings, and the example/doc freshness checks.
- Match the surrounding code style (naming, comment density, idiom).

## Versioning

The version is single-sourced in `include/etree/etree.hpp` (the
`ETREE_VERSION_*` macros). `CMakeLists.txt` parses it and a CI check keeps
`pyproject.toml` and `CITATION.cff` in sync — bump all three together.
