# Amendment A3 — reproducing the test run

Run on a **fixed, clean commit** and do not commit or switch branches while the tests run (several tests record
`git rev-parse HEAD` in checkpoint identities and pin comparisons, so a moving `HEAD` makes them fail spuriously).

```
git status --short                       # must print nothing
git rev-parse HEAD                       # the commit under test
# Interpreter: a NON-venv Python 3.13 whose site-packages (or user site) already hold requirements.txt, so that the
# subprocesses some tests spawn (with their own PYTHONPATH=src) can import them too. Colab's kernel is such an interpreter.
# Inside a venv the A2 test tests/test_exp8_amendment.py::test_ensure_interpreter_uses_reuses_builds_or_refuses fails,
# because it symlinks sys.executable and a symlinked venv interpreter loses its pyvenv.cfg (a venv artifact, not a
# code defect). Example with a system Python and no root access:
/usr/bin/python3.13 -m pip install --user -r requirements.txt
PYTHONPATH=$PWD/src /usr/bin/python3.13 -m pytest tests -q -p no:cacheprovider --show-capture=no
PYTHONPATH=$PWD/src /usr/bin/python3.13 scripts/a3.py smoke             # synthetic end-to-end, hard links
PYTHONPATH=$PWD/src /usr/bin/python3.13 scripts/a3.py smoke --no-link   # same on a file system without hard links
git status --short                       # still nothing
```

In Colab: `pip install -r requirements.txt` then `pytest tests -q` (notebook 00 runs the suite once per commit).
A3 tests: `tests/test_a3.py` (freeze, gates, holdout, reports, statistics, notebooks; about 15 minutes, dominated by
one module-level synthetic development-CV fixture), `tests/test_a3_write_once.py` (write-once, seconds).
