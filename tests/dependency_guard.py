"""pytest plugin: a test skipped because a dependency is missing fails the session.

The real-run environment is pinned (requirements.txt) and the suite must pass with zero
dependency-related skips before the real-data smoke test; ``pytest.importorskip`` would
otherwise turn a missing package (e.g. edfio, torch) into a silent, green skip.
"""

DEPENDENCY_SKIPS: list[str] = []


def _dependency_skip(report) -> bool:
    return report.skipped and "could not import" in str(report.longrepr)


def pytest_collectreport(report):
    if _dependency_skip(report):
        DEPENDENCY_SKIPS.append(f"{report.nodeid}: {report.longrepr}")


def pytest_runtest_logreport(report):
    if _dependency_skip(report):
        DEPENDENCY_SKIPS.append(f"{report.nodeid}: {report.longrepr}")


def pytest_sessionfinish(session, exitstatus):
    if DEPENDENCY_SKIPS and exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    if DEPENDENCY_SKIPS:
        terminalreporter.section("dependency-related skips (session FAILED)", red=True)
        for line in DEPENDENCY_SKIPS:
            terminalreporter.write_line(line)
        terminalreporter.write_line("install the pinned environment: pip install -r requirements.txt")
