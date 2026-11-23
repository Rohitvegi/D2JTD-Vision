"""Single command-line entry point:  rgbthermal <command> [options]

  rgbthermal app                       start the desktop application
  rgbthermal doctor                    check that this installation is ready
  rgbthermal calibrate find|verify|apply|selftest|benchmark ...   RGB->thermal calibration (src.calibration)
  rgbthermal process [--only ...]      run the measurement pipeline on the original dataset
  rgbthermal joint-view --rec ID       draw RGB joints with temperatures + thermal ROIs
  rgbthermal test                      run the automated tests
  rgbthermal version

Works from any folder: it switches to the installation folder (RGBT_HOME, default: where this code lives).
All runs are logged to logs/rgbthermal.log.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

HOME = Path(os.environ.get("RGBT_HOME", Path(__file__).resolve().parents[1]))


def setup(verbose: bool = False) -> logging.Logger:
    os.chdir(HOME)
    if str(HOME) not in sys.path:
        sys.path.insert(0, str(HOME))
    (HOME / "logs").mkdir(exist_ok=True)
    root = logging.getLogger()
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler(HOME / "logs" / "rgbthermal.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
        root.setLevel(logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger("rgbthermal")

    def hook(t, v, tb):                               # uncaught errors end up in the log, not only on screen
        log.critical("uncaught error", exc_info=(t, v, tb))
        sys.__excepthook__(t, v, tb)
    sys.excepthook = hook
    return log


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    log = setup("--verbose" in rest)
    rest = [a for a in rest if a != "--verbose"]
    from src.__version__ import __version__
    log.info("rgbthermal %s: %s %s", __version__, cmd, " ".join(rest))
    if cmd == "version":
        print(__version__)
        return 0
    if cmd == "app":
        from ui.app import main as app_main
        return app_main()
    if cmd == "doctor":
        from src.doctor import main as m
        return m()
    if cmd == "calibrate":
        from src.calibration import main as m
        return m(rest)
    if cmd == "process":
        from src.run_pipeline import main as m
        return m(rest)
    if cmd == "joint-view":
        from src.joint_view import main as m
        return m(rest)
    if cmd == "test":
        import pytest
        return pytest.main(["-q", str(HOME / "tests"), *rest])
    print(f"unknown command '{cmd}'\n{__doc__}")
    return 2


def app_entry() -> int:
    """GUI entry point (no console window when installed as a gui-script)."""
    setup()
    from ui.app import main as app_main
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
