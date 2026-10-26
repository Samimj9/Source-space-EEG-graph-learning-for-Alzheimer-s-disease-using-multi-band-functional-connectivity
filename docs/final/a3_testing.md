# Amendment A3 — reproducing the test run

Run on a **fixed, clean commit** and do not commit or switch branches while the tests run (several tests record
`git rev-parse HEAD` in checkpoint identities and pin comparisons, so a moving `HEAD` makes them fail spuriously).

```
git status --short                       # must print nothing
git rev-parse HEAD                       # the commit under test
# Interpreter: a NON-venv Python 3.13 (Colab's kernel is one). Inside a venv the A2 test
# tests/test_exp8_amendment.py::test_ensure_interpreter_uses_reuses_builds_or_refuses fails, because it symlinks
# sys.executable and a symlinked venv interpreter loses its pyvenv.cfg (an artifact of the venv, not of the code).
SITE=$(mktemp -d); uv pip install --python /usr/bin/python3.13 --target $SITE -r requirements.txt
PYTHONPATH=$SITE:$PWD/src /usr/bin/python3.13 -m pytest tests -q -p no:cacheprovider --show-capture=no
PYTHONPATH=$SITE:$PWD/src /usr/bin/python3.13 scripts/a3.py smoke             # synthetic end-to-end, hard links
PYTHONPATH=$SITE:$PWD/src /usr/bin/python3.13 scripts/a3.py smoke --no-link   # same on a file system without hard links
git status --short                       # still nothing
```

In Colab: `pip install -r requirements.txt` then `pytest tests -q` (notebook 00 runs the suite once per commit).
A3 tests: `tests/test_a3.py` (freeze, gates, holdout, reports, statistics, notebooks; about 15 minutes, dominated by
one module-level synthetic development-CV fixture), `tests/test_a3_write_once.py` (write-once, seconds).
