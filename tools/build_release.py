"""Build the deployable release: dist/rgb_thermal_hand-<version>/ and a .zip of it.

  .venv312\\Scripts\\python.exe tools\\build_release.py

Contains program code, models (with SHA-256 checksums), configuration (no personal paths), tests, installer,
launchers, documentation and the final report/slides. NOT included: datasets, processed data, virtual
environments, caches, logs (datasets are private and large; the app imports new data itself).
"""
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.__version__ import __version__  # noqa: E402

NAME = f"rgb_thermal_hand-{__version__}"
DIST = ROOT / "dist"
OUT = DIST / NAME
CODE_DIRS = ["src", "ui", "tests", "tools"]
FILES = ["config.yaml", "pyproject.toml", "requirements.lock.txt", "requirements-docs.txt", "install.ps1",
         "rgbthermal.bat", "RGB-Thermal App.bat", "README.md", "DEPLOYMENT.md"]
DOCS = ["docs/annotation_guide.md", "docs/annotation_example_start.png", "docs/annotation_example_half.png",
        "docs/annotation_example_done.png", "docs/stage2_literature_review.md", "docs/stage3_learning_objective.md",
        "docs/v4_audit_and_plan.md"]
DELIVERABLES = ["RGB_Thermal_Hand_Complete_Project_Report.pdf", "RGB_Thermal_Hand_Arthritis_Technical_Research_Report.pdf", "RGB_Thermal_Hand_Arthritis_Project_Presentation.pptx",
                "RGB_Thermal_Hand_Project_Audit_and_Viva_Guide.pdf", "RGB_Thermal_Hand_Viva_Presentation.pptx", "Demo_Script.md", "README.md"]
REPORTS = ["reports/final_report.md", "reports/v5_v7_update.md", "reports/v6_v8_update.md", "reports/v9_calibration/README.md",
           "reports/v9_calibration/fig_calibration_model.png", "reports/v9_calibration/method_comparison.csv",
           "reports/v9_calibration/selftest_known_distortions.csv", "reports/v9_calibration/benchmark_find_calibration.json"]
SKIP = re.compile(r"(__pycache__|\.pyc$|\.pytest_cache)")


def copy_tree(src: Path, dst: Path):
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if SKIP.search(str(rel)) or p.is_dir():
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for d in CODE_DIRS:
        copy_tree(ROOT / d, OUT / d)
    copy_tree(ROOT / "models", OUT / "models")
    for f in FILES + DOCS + REPORTS:
        if (ROOT / f).is_file():
            (OUT / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, OUT / f)
        elif f not in ("DEPLOYMENT.md",):
            print("  (skipped, not found)", f)
    for f in DELIVERABLES:
        if (ROOT / "deliverables" / f).is_file():
            (OUT / "deliverables").mkdir(exist_ok=True)
            shutil.copy2(ROOT / "deliverables" / f, OUT / "deliverables" / f)
    # configuration without personal paths; the dataset location is set per machine
    cfg = (OUT / "config.yaml").read_text(encoding="utf-8")
    cfg = re.sub(r'^dataset_root:.*$', 'dataset_root: ""            # original FLIROne dataset folder (optional); '
                 'or set the environment variable RGBT_DATASET_ROOT', cfg, count=1, flags=re.M)
    (OUT / "config.yaml").write_text(cfg, encoding="utf-8")
    for d in ("data", "results", "logs"):
        (OUT / d).mkdir(exist_ok=True)
        (OUT / d / ".keep").write_text("")
    (OUT / "VERSION").write_text(__version__ + "\n")
    # personal-path check: nothing in the release may contain this machine's user folder
    leaks = [str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and p.suffix in (".py", ".yaml", ".json", ".md", ".txt", ".ps1", ".bat", ".toml")
             and re.search(r"C:[/\\]Users[/\\][^/\\]+", p.read_text(encoding="utf-8", errors="ignore"))]
    if leaks:
        print("WARNING: files with a personal path:", leaks)
    lines = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}")
    (OUT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    z = DIST / f"{NAME}.zip"
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, f"{NAME}/{p.relative_to(OUT).as_posix()}")
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"release: {OUT} ({n} files)\nzip:     {z} ({z.stat().st_size / 2**20:.1f} MB)")
    return 0 if not leaks else 1


if __name__ == "__main__":
    raise SystemExit(main())
