"""RGB-Thermal Hand-Joint Screening Support — desktop application (PySide6).

Start (from the project folder):   .venv312\\Scripts\\python.exe -m ui.app

Every action calls the tested modules in src/; long tasks run in a background thread; errors are
shown as messages instead of crashing. Research / screening-support tool, not a diagnostic device.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,  # noqa: E402
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from src import dataset_manager as DM  # noqa: E402
from src.io_utils import load_config  # noqa: E402

CFG = load_config(ROOT / "config.yaml")
REGISTRY = ROOT / "models" / "registry"
STATUS_COLOURS = {"Valid": "#0ca30c", "Valid with warnings": "#c98500", "Invalid": "#d03b3b",
                  "Requires calibration": "#6250d6", "Requires manual file matching": "#eb6834",
                  "Normal measurement": "#0ca30c", "Potential anomaly": "#d03b3b", "Low-confidence measurement": "#c98500",
                  "Invalid measurement": "#7a7973", "Insufficient data": "#7a7973", "Outside model experience": "#6250d6",
                  "Not applicable": "#b4b2a9"}
STYLE = """
QMainWindow, QWidget { background: #fcfcfb; color: #0b0b0b; font-size: 10pt; }
QListWidget#nav { background: #1f2a37; color: #f0efec; border: none; font-size: 11pt; }
QListWidget#nav::item { padding: 10px 14px; }
QListWidget#nav::item:selected { background: #2a78d6; }
QPushButton { background: #2a78d6; color: white; border: none; padding: 6px 12px; border-radius: 4px; }
QPushButton:disabled { background: #b4b2a9; }
QPushButton[secondary="true"] { background: #e4e3df; color: #0b0b0b; }
QLabel#h1 { font-size: 16pt; font-weight: bold; }
QLabel#note { color: #52514e; }
QTableWidget { gridline-color: #e4e3df; background: white; }
QPlainTextEdit { background: white; font-family: Consolas, monospace; font-size: 9pt; }
"""


# --------------------------------------------------------------------------- #
# background work
# --------------------------------------------------------------------------- #
class _Sig(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)


class Job(QRunnable):
    def __init__(self, fn, *a, **k):
        super().__init__()
        self.fn, self.a, self.k, self.sig = fn, a, k, _Sig()

    def run(self):
        try:
            self.sig.done.emit(self.fn(*self.a, **self.k))
        except Exception as e:  # shown to the user, with details in the log box
            self.sig.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}")


def run_job(owner, fn, on_done, *a, progress_cb=None, **k):
    job = Job(fn, *a, **k)
    if progress_cb is not None:
        k_progress = lambda n, t, r: job.sig.progress.emit(n, t, str(r.get("rec_id", "")) if isinstance(r, dict) else "")  # noqa: E731
        job.k["progress"] = k_progress
        job.sig.progress.connect(progress_cb)
    job.sig.done.connect(on_done)
    job.sig.failed.connect(lambda msg: error(owner, msg))
    owner._jobs = getattr(owner, "_jobs", []) + [job]
    QThreadPool.globalInstance().start(job)


def error(parent, msg):
    QMessageBox.warning(parent, "Something went wrong", msg.split("\n\n")[0] + "\n\nDetails are in the log panel.")
    log = parent.window().findChild(QPlainTextEdit, "log")
    if log:
        log.appendPlainText(msg)


def h1(text):
    lab = QLabel(text)
    lab.setObjectName("h1")
    return lab


def note(text):
    lab = QLabel(text)
    lab.setObjectName("note")
    lab.setWordWrap(True)
    return lab


def secondary(btn):
    btn.setProperty("secondary", True)
    return btn


def fill_table(t: QTableWidget, df: pd.DataFrame, colour_col: str | None = None):
    t.clear()
    t.setRowCount(len(df))
    t.setColumnCount(len(df.columns))
    t.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for i, (_, row) in enumerate(df.iterrows()):
        for j, c in enumerate(df.columns):
            v = row[c]
            txt = f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
            it = QTableWidgetItem(txt)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            if colour_col and c == colour_col and txt in STATUS_COLOURS:
                it.setForeground(QColor(STATUS_COLOURS[txt]))
            t.setItem(i, j, it)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


def default_calibration() -> str:
    """The calibration profile the pipeline uses by default (config: registration.calibration_model)."""
    p = CFG["pipeline"]["registration"].get("calibration_model")
    names = DM.list_calibrations()
    stem = Path(p).stem if p else ""
    return stem if stem in names else (names[0] if names else "")


def rel_to(path: str, root: str) -> str:
    """Show a file relative to its dataset folder (no personal folders in messages or screenshots)."""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return Path(path).name


def datasets():
    return [d["dataset_id"] for d in DM.list_datasets()]


def models():
    return sorted((p.name for p in REGISTRY.glob("*") if (p / "metadata.json").is_file()), reverse=True)


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Dashboard"))
        lay.addWidget(note("Research and screening-support tool for joint-wise hand temperature. It measures and compares "
                           "against a healthy reference; it does not diagnose arthritis."))
        self.text = QPlainTextEdit(readOnly=True)
        lay.addWidget(self.text)
        b = secondary(QPushButton("Refresh"))
        b.clicked.connect(self.refresh)
        lay.addWidget(b, alignment=Qt.AlignLeft)
        self.refresh()

    def refresh(self):
        lines = [f"Datasets imported: {len(datasets())}", f"Saved models: {len(models())}"]
        tests = sorted((ROOT / "results" / "independent_tests").glob("*"))
        lines.append(f"Independent test runs: {len(tests)}")
        for t in tests[-5:]:
            lines.append(f"   {t.name}")
        s = ROOT / "baselines" / "v3_baseline" / "reports" / "results" / "summary.json"
        if s.is_file():
            j = json.load(open(s))
            lines += ["", "v3 baseline (frozen):", f"   hand-mask Dice {j['RQ2']['dice_heldout_median']:.3f}",
                      f"   joint error {j['RQ2'].get('landmark_err_corrected_loso_median_px', float('nan')):.1f} px (corrected, LOSO)",
                      f"   10-joint D2JTD MDC95 {j['RQ4']['mean10_mdc95']:.2f} degC"]
        v7 = ROOT / "reports" / "v7_models" / "summary.json"
        if v7.is_file():
            lines += ["", f"v7 selected anomaly model: {json.load(open(v7))['selected_model']}"]
        self.text.setPlainText("\n".join(lines))


class SingleUpload(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload one recording")
        f = QFormLayout(self)
        self.paths = {}
        for s, lab in (("rgb", "RGB video *"), ("thermal_video", "Thermal video (optional)"), ("radiometric", "Radiometric .npy *")):
            row = QHBoxLayout()
            e = QLineEdit()
            b = secondary(QPushButton("Browse"))
            b.clicked.connect(lambda _=False, e=e: e.setText(QFileDialog.getOpenFileName(self, "Choose file")[0]))
            row.addWidget(e)
            row.addWidget(b)
            f.addRow(lab, row)
            self.paths[s] = e
        self.subject, self.hand, self.task = QLineEdit("P01"), QComboBox(), QComboBox()
        self.hand.addItems(["left", "right"])
        self.task.addItems(["static", "open_close"])
        f.addRow("Subject ID", self.subject)
        f.addRow("Hand", self.hand)
        f.addRow("Recording type", self.task)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)


class DatasetPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Dataset upload and validation"))
        lay.addWidget(note("Import a ZIP (extracted safely), a folder (used in place, nothing copied) or one recording. "
                           "Files are matched automatically by name; fix mismatches with 'Assign file'."))
        row = QHBoxLayout()
        for txt, fn in (("Import ZIP...", self.import_zip), ("Import folder...", self.import_folder),
                        ("Upload one recording...", self.import_single)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        row2 = QHBoxLayout()
        self.ds = QComboBox()
        self.ds.currentTextChanged.connect(self.show_dataset)
        self.cal = QComboBox()
        row2.addWidget(QLabel("Dataset"))
        row2.addWidget(self.ds, 2)
        row2.addWidget(QLabel("Calibration"))
        row2.addWidget(self.cal, 1)
        self.assign_btn = secondary(QPushButton("Assign file..."))
        self.assign_btn.clicked.connect(self.assign)
        self.val_btn = QPushButton("Validate")
        self.val_btn.clicked.connect(self.validate)
        row2.addWidget(self.assign_btn)
        row2.addWidget(self.val_btn)
        lay.addLayout(row2)
        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget()
        self.table.itemSelectionChanged.connect(self.show_checks)
        self.checks = QPlainTextEdit(readOnly=True)
        split.addWidget(self.table)
        split.addWidget(self.checks)
        lay.addWidget(split, 1)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.validation = {}
        self.refresh()

    def refresh(self):
        cur = self.ds.currentText()
        self.ds.blockSignals(True)
        self.ds.clear()
        self.ds.addItems(datasets())
        if cur:
            self.ds.setCurrentText(cur)
        self.ds.blockSignals(False)
        cur_cal = self.cal.currentText()
        self.cal.clear()
        self.cal.addItems(DM.list_calibrations())
        self.cal.setCurrentText(cur_cal or default_calibration())
        self.show_dataset(self.ds.currentText())

    def show_dataset(self, did):
        if not did:
            self.table.setRowCount(0)
            return
        m = DM.load_manifest(did)
        vfile = DM.DATASETS / did / "validation.json"
        self.validation = {v["rec_id"]: v for v in json.load(open(vfile))["results"]} if vfile.is_file() else {}
        rows = [dict(rec_id=r["rec_id"], subject=r["subject"], hand=r["hand"], task=r["task"],
                     rgb=Path(r["files"].get("rgb", "")).name, thermal=Path(r["files"].get("thermal_video", "")).name,
                     radiometric=Path(r["files"].get("radiometric", "")).name, matching=r["match_status"],
                     validation=self.validation.get(r["rec_id"], {}).get("status", "not validated")) for r in m["recordings"]]
        fill_table(self.table, pd.DataFrame(rows), colour_col="validation")
        self.checks.setPlainText(f"{len(m['recordings'])} recordings; {len(m['unmatched_files'])} unmatched files:\n" +
                                 "\n".join(rel_to(f, m.get("root", "")) for f in m["unmatched_files"][:50]))

    def _after_import(self, m):
        self.refresh()
        self.ds.setCurrentText(m["dataset_id"])
        QMessageBox.information(self, "Imported", f"{m['dataset_id']}: {len(m['recordings'])} recordings, "
                                                  f"{len(m['unmatched_files'])} unmatched files.")

    def import_zip(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose dataset ZIP", "", "ZIP archives (*.zip)")
        if p:
            run_job(self, DM.import_zip, self._after_import, Path(p))

    def import_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Choose dataset folder")
        if p:
            run_job(self, DM.import_folder, self._after_import, Path(p))

    def import_single(self):
        d = SingleUpload(self)
        if d.exec():
            files = {k: e.text() for k, e in d.paths.items() if e.text()}
            try:
                self._after_import(DM.import_single(files, d.subject.text(), d.hand.currentText(), d.task.currentText()))
            except Exception as e:
                error(self, f"{type(e).__name__}: {e}")

    def assign(self):
        r = self.table.currentRow()
        if r < 0:
            return QMessageBox.information(self, "Assign file", "Select a recording first.")
        rec_id = self.table.item(r, 0).text()
        stream, ok = QInputStream.get(self)
        if not ok:
            return
        p, _ = QFileDialog.getOpenFileName(self, f"File for {rec_id} ({stream})")
        if p:
            m = DM.set_file(DM.load_manifest(self.ds.currentText()), rec_id, stream, p)
            DM.save_manifest(m)
            self.show_dataset(m["dataset_id"])

    def validate(self):
        did, cal = self.ds.currentText(), self.cal.currentText()
        if not did or not cal:
            return QMessageBox.information(self, "Validate", "Choose a dataset and a calibration profile.")
        self.val_btn.setEnabled(False)
        run_job(self, DM.validate_manifest, self._validated, DM.load_manifest(did), DM.load_calibration(cal), CFG,
                progress_cb=lambda n, t, _: (self.progress.setMaximum(t), self.progress.setValue(n)))

    def _validated(self, res):
        self.val_btn.setEnabled(True)
        self.show_dataset(self.ds.currentText())
        counts = pd.Series([r["status"] for r in res]).value_counts().to_dict()
        QMessageBox.information(self, "Validation finished", "\n".join(f"{k}: {v}" for k, v in counts.items()))

    def show_checks(self):
        r = self.table.currentRow()
        if r < 0:
            return
        v = self.validation.get(self.table.item(r, 0).text())
        if not v:
            return self.checks.setPlainText("Not validated yet.")
        self.checks.setPlainText(f"{v['rec_id']}: {v['status']}\n" + "\n".join(
            f"  [{c['level'].upper():5s}] {c['check']}: {c['message']}" for c in v["checks"]))


class QInputStream:
    @staticmethod
    def get(parent):
        d = QDialog(parent)
        d.setWindowTitle("Which file?")
        f = QFormLayout(d)
        c = QComboBox()
        c.addItems(list(DM.STREAMS))
        f.addRow("Stream", c)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        return c.currentText(), bool(d.exec())


class CalibrationPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Calibration profiles"))
        lay.addWidget(note("A profile stores the camera geometry and the RGB-to-thermal mapping (image area, scale, rotation "
                           "and a per-frame shift learned from the hand). Default: flirone_one_pro_v9_found, found from the "
                           "68 FLIROne recordings. For a new camera or export (validation says 'Requires calibration'), "
                           "choose the dataset below and press Auto-calibrate: the mapping is found from its own recordings, "
                           "checked on frames not used for fitting, and saved as a new profile (grade good / acceptable / poor)."))
        row = QHBoxLayout()
        self.list = QListWidget()
        self.list.currentTextChanged.connect(self.show)
        self.text = QPlainTextEdit(readOnly=True)
        row.addWidget(self.list, 1)
        row.addWidget(self.text, 3)
        lay.addLayout(row, 1)
        b = secondary(QPushButton("Create profile from current config (v3 mapping)"))
        b.clicked.connect(self.create)
        auto = QHBoxLayout()
        self.ds = QComboBox()
        self.ds.addItems(datasets())
        self.ds.setToolTip("Dataset whose own recordings are used to find the calibration")
        ab = QPushButton("Auto-calibrate from this dataset")
        ab.setToolTip("Finds the image area, scale and RGB-to-thermal shift model from the dataset itself (learned "
                      "calibration, v9), checks it on frames not used for fitting and saves a new profile.")
        ab.clicked.connect(self.auto)
        self.status = QLabel("")
        for w in (b, QLabel("   Dataset"), self.ds, ab, self.status):
            auto.addWidget(w)
        auto.addStretch(1)
        lay.addLayout(auto)
        self.refresh()

    def auto(self):
        did = self.ds.currentText()
        if not did:
            return QMessageBox.information(self, "Auto-calibrate", "Import a dataset first.")
        self.status.setText("calibrating… (about 1–2 minutes)")
        run_job(self, DM.auto_calibrate_dataset, self._auto_done, DM.load_manifest(did), CFG)

    def _auto_done(self, prof):
        sc = prof["self_check"]
        self.status.setText(f"saved '{prof['name']}': scale {prof['scale']:.2f}, held-out Dice {sc['dice_heldout_median']:.3f} "
                            f"({sc['frames_test']} test frames)")
        self.refresh()

    def refresh(self):
        self.list.clear()
        self.list.addItems(DM.list_calibrations())

    def show(self, name):
        if name:
            self.text.setPlainText(json.dumps(DM.load_calibration(name), indent=2))

    def create(self):
        try:
            DM.save_calibration(DM.calibration_from_config(CFG, f"flirone_one_pro_v3_{pd.Timestamp.now():%Y%m%d_%H%M%S}"))
            self.refresh()
        except Exception as e:
            error(self, str(e))


class ProcessingPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Data processing"))
        lay.addWidget(note("Runs hand detection, joint localisation, RGB-thermal mapping, joint ROIs and feature extraction. "
                           "Only recordings validated as Valid / Valid with warnings are selected; including others "
                           "needs your explicit confirmation."))
        row = QHBoxLayout()
        self.ds, self.cal = QComboBox(), QComboBox()
        self.workers = QSpinBox(minimum=1, maximum=16, value=4)
        for w, lab in ((self.ds, "Dataset"), (self.cal, "Calibration"), (self.workers, "Parallel workers")):
            row.addWidget(QLabel(lab))
            row.addWidget(w)
        self.ds.currentTextChanged.connect(self.show)
        self.start = QPushButton("Start processing")
        self.start.clicked.connect(self.run)
        row.addWidget(self.start)
        lay.addLayout(row)
        self.table = QTableWidget()
        lay.addWidget(self.table, 1)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.refresh()

    def refresh(self):
        for c, items in ((self.ds, datasets()), (self.cal, DM.list_calibrations())):
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            c.addItems(items)
            if cur:
                c.setCurrentText(cur)
            elif c is self.cal:
                c.setCurrentText(default_calibration())
            c.blockSignals(False)
        self.show(self.ds.currentText())

    def show(self, did):
        if not did:
            return
        m = DM.load_manifest(did)
        vf = DM.DATASETS / did / "validation.json"
        val = {v["rec_id"]: v["status"] for v in json.load(open(vf))["results"]} if vf.is_file() else {}
        df = pd.DataFrame([dict(rec_id=r["rec_id"], status=val.get(r["rec_id"], "not validated")) for r in m["recordings"]])
        fill_table(self.table, df, colour_col="status")
        self.table.setSelectionMode(QTableWidget.MultiSelection)
        for i, s in enumerate(df.status):
            if s in ("Valid", "Valid with warnings"):
                self.table.selectRow(i)

    def run(self):
        did, cal = self.ds.currentText(), self.cal.currentText()
        if not did or not cal:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        recs = [self.table.item(r, 0).text() for r in rows]
        bad = [self.table.item(r, 0).text() for r in rows if self.table.item(r, 1).text() not in ("Valid", "Valid with warnings")]
        if bad and QMessageBox.question(self, "Include unvalidated or invalid recordings?",
                                        f"{len(bad)} selected recording(s) are not validated as usable:\n{', '.join(bad[:8])}\n\nProcess them anyway?") != QMessageBox.Yes:
            return
        if not recs:
            return QMessageBox.information(self, "Processing", "Select at least one recording.")
        self.start.setEnabled(False)
        self.bar.setMaximum(len(recs))
        self.bar.setValue(0)
        run_job(self, DM.process_dataset, self.done, DM.load_manifest(did), CFG, DM.load_calibration(cal), recs,
                self.workers.value(), progress_cb=lambda n, t, _: self.bar.setValue(n))

    def done(self, res):
        self.start.setEnabled(True)
        reg, F = res
        QMessageBox.information(self, "Processing finished", f"{len(F)} recordings processed; features saved.")


class ModelsPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Models"))
        lay.addWidget(note("Saved models are versioned and never overwritten. Healthy-reference models need healthy "
                           "recordings only. A supervised arthritis classifier can only be validated when the labels "
                           "contain both healthy and arthritis subjects."))
        split = QSplitter()
        self.list = QListWidget()
        self.list.currentTextChanged.connect(self.show)
        self.meta = QPlainTextEdit(readOnly=True)
        split.addWidget(self.list)
        split.addWidget(self.meta)
        lay.addWidget(split, 1)
        f = QFormLayout()
        self.ds, self.model, self.fset = QComboBox(), QComboBox(), QComboBox()
        from src.anomaly_models import MODELS
        from src.feature_extraction import FEATURE_SETS
        self.model.addItems(list(MODELS))
        self.model.setCurrentText("mahalanobis_ledoit_wolf")
        self.fset.addItems(list(FEATURE_SETS))
        self.labels = QLineEdit()
        lb = secondary(QPushButton("Browse"))
        lb.clicked.connect(lambda: self.labels.setText(QFileDialog.getOpenFileName(self, "Labels", "", "CSV/Excel (*.csv *.xlsx)")[0]))
        lrow = QHBoxLayout()
        lrow.addWidget(self.labels)
        lrow.addWidget(lb)
        f.addRow("Processed dataset", self.ds)
        f.addRow("Model", self.model)
        f.addRow("Feature set", self.fset)
        f.addRow("Labels (optional)", lrow)
        lay.addLayout(f)
        row = QHBoxLayout()
        self.train = QPushButton("Train healthy-reference model")
        self.train.clicked.connect(self.run_train)
        sup = secondary(QPushButton("Supervised cross-validation (needs labels)"))
        sup.clicked.connect(self.run_supervised)
        row.addWidget(self.train)
        row.addWidget(sup)
        row.addStretch()
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        self.list.clear()
        self.list.addItems(models())
        self.ds.clear()
        self.ds.addItems(["FLIROne original (reports/v6_features)"] + [d for d in datasets() if (DM.DATASETS / d / "processed" / "features.csv").is_file()])

    def show(self, mid):
        if mid:
            self.meta.setPlainText(json.dumps(json.load(open(REGISTRY / mid / "metadata.json")), indent=2)[:20000])

    def _features(self):
        d = self.ds.currentText()
        p = ROOT / "reports/v6_features/features.csv" if d.startswith("FLIROne") else DM.DATASETS / d / "processed" / "features.csv"
        return pd.read_csv(p), p, d

    def run_train(self):
        from src.evaluation import load_labels, match_labels, train_reference_model
        from src.feature_extraction import FEATURE_SETS
        F, p, d = self._features()
        labels = None
        if self.labels.text():
            recs = [dict(rec_id=r.rec_id, subject=r.subject, hand=r.hand, task=r.task) for r in F.itertuples()]
            labels, _ = match_labels(load_labels(self.labels.text()), recs)
        self.train.setEnabled(False)
        run_job(self, train_reference_model, self.trained, F, FEATURE_SETS[self.fset.currentText()],
                self.model.currentText(), f"{d} ({self.fset.currentText()})", p, labels)

    def trained(self, res):
        self.train.setEnabled(True)
        d, s = res
        self.refresh()
        QMessageBox.information(self, "Model saved", f"{Path(d).name}\nspecificity (nested LOSO) {s['specificity']:.2f}")

    def run_supervised(self):
        from src.evaluation import load_labels, match_labels, supervised_cv
        from src.feature_extraction import FEATURE_SETS
        if not self.labels.text():
            return QMessageBox.information(self, "Supervised", "Choose a label file first.")
        F, _, _ = self._features()
        recs = [dict(rec_id=r.rec_id, subject=r.subject, hand=r.hand, task=r.task) for r in F.itertuples()]
        labels, _ = match_labels(load_labels(self.labels.text()), recs)
        run_job(self, supervised_cv, lambda r: self.meta.setPlainText(json.dumps(r, indent=2, default=float)),
                F, labels, FEATURE_SETS[self.fset.currentText()])


class TestingPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Independent model testing"))
        lay.addWidget(note("Applies a saved model to a processed dataset without any retraining or threshold change. "
                           "Each run is locked (inputs fingerprinted) and stored separately. Metrics are computed only "
                           "for labels that support them."))
        f = QFormLayout()
        self.ds, self.model = QComboBox(), QComboBox()
        self.labels = QLineEdit()
        lb = secondary(QPushButton("Browse"))
        lb.clicked.connect(lambda: self.labels.setText(QFileDialog.getOpenFileName(self, "Labels", "", "CSV/Excel (*.csv *.xlsx)")[0]))
        lrow = QHBoxLayout()
        lrow.addWidget(self.labels)
        lrow.addWidget(lb)
        f.addRow("Processed dataset", self.ds)
        f.addRow("Saved model", self.model)
        f.addRow("Ground-truth labels (optional)", lrow)
        lay.addLayout(f)
        row = QHBoxLayout()
        self.go = QPushButton("Run independent test")
        self.go.clicked.connect(self.run)
        self.rep = secondary(QPushButton("Create PDF / Excel report"))
        self.rep.clicked.connect(self.report)
        row.addWidget(self.go)
        row.addWidget(self.rep)
        row.addStretch()
        lay.addLayout(row)
        self.summary = QPlainTextEdit(readOnly=True)
        self.table = QTableWidget()
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.summary)
        split.addWidget(self.table)
        lay.addWidget(split, 1)
        self.last = None
        self.refresh()

    def refresh(self):
        self.ds.clear()
        self.ds.addItems([d for d in datasets() if (DM.DATASETS / d / "processed" / "features.csv").is_file()])
        self.model.clear()
        self.model.addItems(models())

    def run(self):
        from src.evaluation import run_independent_test
        if not self.ds.currentText() or not self.model.currentText():
            return QMessageBox.information(self, "Test", "Process a dataset and save a model first.")
        self.go.setEnabled(False)
        run_job(self, run_independent_test, self.done, self.ds.currentText(), REGISTRY / self.model.currentText(),
                self.labels.text() or None)

    def done(self, out):
        self.go.setEnabled(True)
        self.last = Path(out)
        res = json.load(open(self.last / "results.json"))
        res_short = {k: v for k, v in res.items() if k not in ("lock",)}
        if "metrics_flag_vs_label" in res_short:
            res_short["metrics_flag_vs_label"] = {k: v for k, v in res_short["metrics_flag_vs_label"].items()
                                                  if k not in ("roc_curve", "pr_curve")}
        self.summary.setPlainText(f"Run folder: {self.last}\n\n" + json.dumps(res_short, indent=2, default=float))
        P = pd.read_csv(self.last / "predictions.csv")
        fill_table(self.table, P[[c for c in ["rec_id", "task", "category", "score", "threshold", "label", "top_features"] if c in P]],
                   colour_col="category")

    def report(self):
        from src.report_generator import test_report
        if not self.last:
            return QMessageBox.information(self, "Report", "Run a test first.")
        self.rep.setEnabled(False)

        def done(p):
            self.rep.setEnabled(True)
            self.window().show_report(p["pdf"])        # jump to the Reports page and display it
        run_job(self, test_report, done, self.last)


class ExplainPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Explainability"))
        lay.addWidget(note("Per-joint deviation from the healthy reference (z: negative D2JTD z means the joint is warmer "
                           "than usual relative to the back of the hand), measurement quality, and the RGB/thermal overlay "
                           "used for the measurement."))
        row = QHBoxLayout()
        self.run = QComboBox()
        self.run.currentTextChanged.connect(self.load_run)
        self.rec = QComboBox()
        self.rec.currentTextChanged.connect(self.show)
        row.addWidget(QLabel("Test run"))
        row.addWidget(self.run, 2)
        row.addWidget(QLabel("Recording"))
        row.addWidget(self.rec, 1)
        lay.addLayout(row)
        split = QSplitter()
        self.table = QTableWidget()
        self.img = QLabel(alignment=Qt.AlignCenter)
        sc = QScrollArea()
        sc.setWidget(self.img)
        sc.setWidgetResizable(True)
        split.addWidget(self.table)
        split.addWidget(sc)
        lay.addWidget(split, 1)
        self.P = None
        self.refresh()

    def refresh(self):
        self.run.clear()
        self.run.addItems(sorted((p.name for p in (ROOT / "results" / "independent_tests").glob("*")), reverse=True))

    def load_run(self, name):
        if not name:
            return
        self.P = pd.read_csv(ROOT / "results" / "independent_tests" / name / "predictions.csv")
        self.lock = json.load(open(ROOT / "results" / "independent_tests" / name / "lock.json"))
        self.rec.clear()
        self.rec.addItems(self.P.rec_id.tolist())

    def show(self, rec):
        if self.P is None or not rec:
            return
        r = self.P[self.P.rec_id == rec].iloc[0]
        zc = [c for c in self.P.columns if c.startswith("z_")]
        df = pd.DataFrame(dict(feature=[c[2:] for c in zc], z=[r[c] for c in zc]))
        df["direction"] = df.z.map(lambda z: "warmer than reference" if z < -2 else ("cooler than reference" if z > 2 else "within reference"))
        df = pd.concat([pd.DataFrame([dict(feature="CATEGORY", z=float("nan"), direction=r.category)]), df])
        fill_table(self.table, df)
        qc = DM.DATASETS / self.lock["dataset_id"] / "processed" / "qc" / f"{rec}.png"
        if qc.is_file():
            self.img.setPixmap(QPixmap(str(qc)).scaledToWidth(560, Qt.SmoothTransformation))
        else:
            self.img.setText("No overlay image for this recording.")


class JointViewPage(QWidget):
    ORIG_DIR = "processed_v8" if (ROOT / "data" / "processed_v8" / "registration.csv").is_file() else "processed_v6"
    ORIGINAL = f"FLIROne original (data/{ORIG_DIR})"

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Joint view"))
        lay.addWidget(note("RGB photo with the 14 detected joints and the temperature measured at each (°C), next to the "
                           "thermal image showing the region each temperature was read from. Dot colour: red = joint "
                           "warmer than the back of the hand, blue = cooler. Joints not visible in the photo (e.g. "
                           "folded into a fist) are not shown and get no temperature; 'n/a' = visible but no reliable "
                           "measurement."))
        row = QHBoxLayout()
        self.src, self.rec = QComboBox(), QComboBox()
        self.src.currentTextChanged.connect(self.load_recs)
        self.frame = QSpinBox(minimum=-1, maximum=100000, value=-1)
        self.frame.setSpecialValueText("middle")
        self.btn = QPushButton("Show")
        self.btn.clicked.connect(self.show_view)
        for w, lab in ((self.src, "Data"), (self.rec, "Recording"), (self.frame, "Frame")):
            row.addWidget(QLabel(lab))
            row.addWidget(w, 2 if w is not self.frame else 0)
        row.addWidget(self.btn)
        lay.addLayout(row)
        split = QSplitter()
        self.img = QLabel(alignment=Qt.AlignCenter)
        sc = QScrollArea()
        sc.setWidget(self.img)
        sc.setWidgetResizable(True)
        self.table = QTableWidget()
        split.addWidget(sc)
        split.addWidget(self.table)
        split.setSizes([1100, 360])
        self.sc = sc
        lay.addWidget(split, 1)
        self.refresh()

    def _proc(self):
        s = self.src.currentText()
        return (ROOT / "data" / self.ORIG_DIR, None) if s == self.ORIGINAL else (DM.DATASETS / s / "processed", s)

    def refresh(self):
        cur = self.src.currentText()
        self.src.blockSignals(True)
        self.src.clear()
        items = ([self.ORIGINAL] if (ROOT / "data" / self.ORIG_DIR / "frames").is_dir() else []) + \
                [d for d in datasets() if (DM.DATASETS / d / "processed" / "frames").is_dir()]
        self.src.addItems(items)
        if cur in items:
            self.src.setCurrentText(cur)
        self.src.blockSignals(False)
        self.load_recs()

    def load_recs(self, *_):
        proc, _ = self._proc()
        self.rec.clear()
        if (proc / "frames").is_dir():
            self.rec.addItems(sorted(p.stem for p in (proc / "frames").glob("*.csv")))

    def show_view(self):
        from src.joint_view import render
        if not self.rec.currentText():
            return
        proc, did = self._proc()
        fr = None if self.frame.value() < 0 else self.frame.value()
        out = ROOT / "reports" / "joint_views" / f"{self.rec.currentText()}_gui.png"
        self.btn.setEnabled(False)
        run_job(self, render, self.shown, self.rec.currentText(), proc, CFG, fr, did, out)

    def shown(self, res):
        self.btn.setEnabled(True)
        png, J = res
        width = self.sc.viewport().width() if hasattr(self, "sc") else 1000
        self.img.setPixmap(QPixmap(str(png)).scaledToWidth(max(600, width - 8), Qt.SmoothTransformation))
        fill_table(self.table, J[["joint", "visible", "temperature_C", "d2jtd_C", "on_hand_pct"]].round(2))


class LivePage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Live mode"))
        lay.addWidget(note("FLIR ONE Pro live streaming needs FLIR's mobile SDK, which is not available on this computer. "
                           "SIMULATION mode replays a recorded recording on two independent clocks (RGB and 8.7 Hz thermal) "
                           "and runs the live measurement path with timestamp pairing. Results are labelled as simulated."))
        row = QHBoxLayout()
        self.rec = QComboBox()
        from src.io_utils import load_recordings
        try:
            self.rec.addItems([i.rec_id for i in load_recordings(CFG) if i.rec_id.endswith("static")])
        except Exception:
            pass
        self.secs = QSpinBox(minimum=5, maximum=120, value=15)
        b = QPushButton("Start simulation")
        b.clicked.connect(self.start)
        cam = secondary(QPushButton("Connect FLIR camera"))
        cam.clicked.connect(self.camera)
        row.addWidget(QLabel("Recording"))
        row.addWidget(self.rec, 2)
        row.addWidget(QLabel("Seconds"))
        row.addWidget(self.secs)
        row.addWidget(b)
        row.addWidget(cam)
        lay.addLayout(row)
        self.out = QPlainTextEdit(readOnly=True)
        lay.addWidget(self.out, 1)

    def start(self):
        from src.live_acquisition import run_simulation
        self.out.setPlainText("SIMULATION running...")
        run_job(self, run_simulation, lambda r: self.out.setPlainText(json.dumps(r, indent=2, default=float)),
                self.rec.currentText(), float(self.secs.value()))

    def camera(self):
        from src.live_acquisition import FlirOneSource
        try:
            FlirOneSource()
        except NotImplementedError as e:
            QMessageBox.information(self, "Camera not available", str(e))


def all_reports() -> list[Path]:
    pdfs = list((ROOT / "results" / "independent_tests").glob("*/*.pdf")) + list(DM.DATASETS.glob("*/reports/*.pdf"))
    return sorted(pdfs, key=lambda p: p.stat().st_mtime, reverse=True)


def report_title(p: Path) -> str:
    if p.parent.parent.name == "independent_tests":
        ds, model, when = (p.parent.name.split("__") + ["", ""])[:3]
        return f"Test: {ds}  ·  {model}  ·  {when}"
    return f"Dataset: {p.parent.parent.name}"


class ReportsPage(QWidget):
    def __init__(self):
        super().__init__()
        from PySide6.QtPdf import QPdfDocument
        from PySide6.QtPdfWidgets import QPdfView
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Reports"))
        lay.addWidget(note("All reports created by the application, newest first. Select one to read it here. "
                           "Test reports are created on the Independent test page; dataset reports below."))
        row = QHBoxLayout()
        self.ds = QComboBox()
        self.mk = QPushButton("Create dataset report")
        self.mk.clicked.connect(self.dataset)
        row.addWidget(QLabel("Dataset"))
        row.addWidget(self.ds, 2)
        row.addWidget(self.mk)
        row.addStretch()
        lay.addLayout(row)
        split = QSplitter()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.show_selected)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.title = QLabel("No report selected")
        self.title.setObjectName("note")
        bar.addWidget(self.title, 1)
        for txt, fn in (("Zoom −", lambda: self.zoom(0.8)), ("Fit width", self.fit), ("Zoom +", lambda: self.zoom(1.25)),
                        ("Open in PDF app", self.open_pdf), ("Open Excel", self.open_excel), ("Open folder", self.open_folder)):
            b = secondary(QPushButton(txt))
            b.clicked.connect(fn)
            bar.addWidget(b)
        rl.addLayout(bar)
        self.doc = QPdfDocument(self)
        self.view = QPdfView()
        self.view.setDocument(self.doc)
        self.view.setPageMode(QPdfView.PageMode.MultiPage)
        self.view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        rl.addWidget(self.view, 1)
        split.addWidget(self.list)
        split.addWidget(right)
        split.setSizes([460, 1000])
        self.list.setWordWrap(True)
        lay.addWidget(split, 1)
        self.paths: list[Path] = []
        self.current: Path | None = None
        self.refresh()

    def refresh(self):
        cur = self.ds.currentText()
        self.ds.clear()
        self.ds.addItems(datasets())
        if cur:
            self.ds.setCurrentText(cur)
        keep = self.current
        self.paths = all_reports()
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.paths:
            self.list.addItem(report_title(p))
            self.list.item(self.list.count() - 1).setToolTip(report_title(p) + "\n" + str(p))
        self.list.blockSignals(False)
        if keep in self.paths:
            self.list.setCurrentRow(self.paths.index(keep))
        elif self.paths and self.current is None:
            self.list.setCurrentRow(0)

    def open_report(self, path):
        self.current = Path(path)
        self.refresh()
        if self.current in self.paths:
            self.list.setCurrentRow(self.paths.index(self.current))

    def show_selected(self, row):
        if 0 <= row < len(self.paths):
            self.current = self.paths[row]
            self.doc.close()
            self.doc.load(str(self.current))
            self.title.setText(f"{report_title(self.current)}  ({self.doc.pageCount()} pages)")
            self.fit()

    def fit(self):
        from PySide6.QtPdfWidgets import QPdfView
        self.view.setZoomMode(QPdfView.ZoomMode.FitToWidth)

    def zoom(self, f):
        from PySide6.QtPdfWidgets import QPdfView
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(max(0.2, min(5.0, self.view.zoomFactor() * f)))

    def _open(self, p: Path | None):
        if p and p.exists():
            os.startfile(str(p))                      # Windows: opens with the default application
        else:
            QMessageBox.information(self, "Not found", "No file for this report.")

    def open_pdf(self):
        self._open(self.current)

    def open_excel(self):
        if self.current:
            xl = sorted(self.current.parent.glob("*.xlsx"))
            self._open(xl[0] if xl else None)

    def open_folder(self):
        if self.current:
            self._open(self.current.parent)

    def dataset(self):
        from src.report_generator import dataset_report
        if not self.ds.currentText():
            return
        self.mk.setEnabled(False)
        self.title.setText("Creating report (renders a joint view per recording)...")

        def done(p):
            self.mk.setEnabled(True)
            self.open_report(p["pdf"])
        run_job(self, dataset_report, done, self.ds.currentText())


# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    PAGES = [("Dashboard", Dashboard), ("Dataset upload", DatasetPage), ("Calibration", CalibrationPage),
             ("Processing", ProcessingPage), ("Models", ModelsPage), ("Independent test", TestingPage),
             ("Joint view", JointViewPage), ("Explainability", ExplainPage), ("Live (simulation)", LivePage),
             ("Reports", ReportsPage)]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RGB-Thermal Hand-Joint Screening Support (research tool)")
        self.resize(1280, 820)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(200)
        self.stack = QStackedWidget()
        self.pages = []
        for name, cls in self.PAGES:
            self.nav.addItem(name)
            p = cls()
            self.pages.append(p)
            self.stack.addWidget(p)
        self.nav.currentRowChanged.connect(self.switch)
        right = QVBoxLayout()
        right.addWidget(self.stack, 1)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setObjectName("log")
        self.log.setMaximumHeight(90)
        self.log.setPlaceholderText("Messages and error details appear here.")
        right.addWidget(self.log)
        lay.addWidget(self.nav)
        w = QWidget()
        w.setLayout(right)
        lay.addWidget(w, 1)
        self.setCentralWidget(central)
        self.nav.setCurrentRow(0)
        if not DM.list_calibrations():
            try:
                DM.save_calibration(DM.calibration_from_config(CFG))
            except Exception:
                pass

    def show_report(self, pdf_path):
        i = [n for n, _ in self.PAGES].index("Reports")
        self.nav.setCurrentRow(i)
        self.pages[i].open_report(pdf_path)

    def switch(self, i):
        self.stack.setCurrentIndex(i)
        p = self.pages[i]
        if hasattr(p, "refresh"):
            try:
                p.refresh()
            except Exception as e:
                error(self, f"{type(e).__name__}: {e}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
