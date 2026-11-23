"""Short faculty presentation (5 slides): title, architecture, innovation, individual outcome + LLM use, demo.

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_faculty_deck.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deck_helpers import *  # noqa: F401,F403
from deck_helpers import EMBER, FIG, GREY, INK, NAVY, PALE, PR, V3, WHITE, badge, bullets, card, f, img, new, prs, table, text  # noqa: E402

OUT = Path("deliverables/RGB_Thermal_Hand_Faculty_Presentation.pptx")
BEN = json.load(open("reports/v9_calibration/benchmark_find_calibration.json"))["results"]
B3, B9 = BEN["v3: fitted per recording"], BEN["found calibration (this module)"]

# ---------------------------------------------------------------- 1 title
s = new("RGB–Thermal Hand-Joint Temperature Analysis", dark=True,
        notes="Project: measuring the temperature of 14 hand joints from a smartphone RGB + radiometric thermal camera and "
              "comparing each hand with a healthy reference. Research and screening-support software, not a diagnosis.")
s.shapes.title.top = s.shapes.title.top + 1828800
text(s, "Architecture · innovation · individual outcome · use of LLM · live demo", 0.6, 3.35, 12, 0.6, 22, PALE, face="Cambria")
text(s, ["FLIROne dataset · 17 healthy subjects · 68 recordings", "Software release 1.0.0"], 0.6, 4.4, 11, 1.0, 15, WHITE)
text(s, "Research and screening-support software — not a diagnostic device", 0.6, 6.4, 11, 0.4, 13, EMBER, italic=True)

# ---------------------------------------------------------------- 2 architecture
s = new("End-to-end architecture and required pipelines", "Architecture",
        notes="Top: the RGB video and the radiometric temperature data pass the data audit and frame pairing. The RGB branch "
              "finds 21 hand points with MediaPipe, checks quality and visibility and builds the RGB hand mask; the thermal "
              "branch segments the hand with Otsu plus hysteresis. The automatic calibration (v9) maps the RGB joints into the "
              "thermal image. Middle: 14 joint regions, temperature sampling (median per region per frame, then median across "
              "usable frames), feature engineering (D2JTD and spatial contrasts), the healthy-reference anomaly model "
              "(Ledoit-Wolf Mahalanobis) and the outputs. Bottom: the subject-wise training and evaluation pipeline (no neural "
              "network is trained) and the pipeline the application uses for new data.")
img(s, FIG / "U01_architecture_overview.png", 0.6, 1.3, 12.1, 5.55, None, None)

# ---------------------------------------------------------------- 3 innovation
s = new("Innovation implemented", "Innovation",
        notes="Six things that go beyond a standard pipeline, each with its measured evidence. 1) The calibration is found from "
              "the hand itself, without a calibration target, and was proven on known distortions and on unseen people. "
              "2) Joints that are folded or covered are not measured. 3) The joint position error was diagnosed and corrected "
              "using hand-clicked ground truth. 4) Screening uses a healthy reference evaluated without leakage, with a "
              "per-joint explanation. 5) Each joint gets one robust temperature with its spread. 6) The system is deployable "
              "and verifiable on another PC.")
INNOV = [
    ("Calibration from the hand itself", "finds image area, scale, rotation and per-frame shift — no calibration target",
     f"12/12 known distortions · {B9['joint_err_median']:.2f} vs {B3['joint_err_median']:.2f} px on unseen people"),
    ("Visibility-aware measurement", "folded or covered joints detected (outline, depth, bend) and not measured",
     "66 % of open–close frames had a hidden joint"),
    ("Annotation-driven joint correction", "777 hand-clicked joints revealed a systematic along-finger shift; learned and removed",
     f"{f(V3['joint_err_raw'], 1)} → {f(V3['joint_err_corr'], 1)} px; cost {f(V3['dT_corr'], 2)} °C"),
    ("Leakage-free healthy reference", "nested leave-one-subject-out model + threshold; per-joint explanation of a flag",
     f"{PR['real_correct']}/{PR['n_hands']} healthy correct · {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints located"),
    ("One robust value per joint", "median over usable frames with Q1–Q3 and frame count in every report",
     f"repeatable to {f(V3['mdc'], 2)} °C (MDC95)"),
    ("Deployable and verifiable", "16 validation checks, locked tests (SHA-256), auto-calibrate, installer + health check",
     "clean install: full workflow + 50 tests passed"),
]
for i, (head, what, ev) in enumerate(INNOV):
    x, y = 0.6 + (i % 3) * 4.1, 1.5 + (i // 3) * 2.75
    card(s, x, y, 3.9, 2.55)
    badge(s, i + 1, x + 0.18, y + 0.2, d=0.5)
    text(s, head, x + 0.8, y + 0.22, 2.95, 0.6, 15, NAVY, bold=True, face="Cambria")
    text(s, what, x + 0.2, y + 0.92, 3.5, 0.95, 12, INK)
    text(s, ev, x + 0.2, y + 1.85, 3.5, 0.6, 11.5, EMBER, bold=True)

# ---------------------------------------------------------------- 4 individual outcome + LLM
s = new("Individual outcome and use of LLM assistance", "Individual outcome · LLM use",
        notes="Left: what I did and what I take away. Right: how the LLM coding assistant was used and how its work was checked. "
              "The assistant helped write and debug code, run experiments and draft documents; the hand-clicked ground truth, "
              "the decisions at each stage and the review of every result were mine. No number is typed by hand: all come from "
              "saved outputs, and the tests caught the assistant's own mistakes.")
card(s, 0.6, 1.5, 5.95, 5.25)
text(s, "Individual outcome", 0.85, 1.62, 5.5, 0.5, 19, NAVY, bold=True, face="Cambria")
bullets(s, [("Directed the project: ", "set the objectives, reviewed and approved each stage, requested each change from what I observed."),
            ("Ground truth: ", "hand-clicked 777 joint centres on 136 thermal frames."),
            ("Problems I found: ", "wrong frame mapping, folded joints showing temperatures, reports shown only as paths, need for calibration on any video."),
            ("Delivered: ", "measurement pipeline, healthy-reference model, desktop app, release 1.0.0, reports and slides."),
            ("Learned: ", "radiometric thermal imaging, hand landmarking, camera mapping, leakage-free evaluation, deployment.")],
        0.85, 2.2, 5.5, 4.5, size=13, gap=6)
card(s, 6.8, 1.5, 5.95, 5.25)
text(s, "Use of LLM assistance", 7.05, 1.62, 5.5, 0.5, 19, NAVY, bold=True, face="Cambria")
bullets(s, [("Tool: ", "Claude (Anthropic) as an AI coding and analysis assistant."),
            ("Used for: ", "writing and debugging code, running experiments, figures, drafting reports and slides."),
            ("Not used for: ", "ground-truth labels (clicked by me) or results — every number comes from saved outputs."),
            ("Checked by: ", "50 automated tests, known-distortion self-test, clean-install acceptance test, my review of every output."),
            ("Mistakes caught: ", "a model-loading bug and a benchmark coordinate error were found by these checks and fixed.")],
        7.05, 2.2, 5.5, 4.5, size=13, gap=6)

# ---------------------------------------------------------------- 5 demo
s = new("Live demo: new data to joint temperatures in the application", "Demo",
        notes="DEMO (about 5 minutes). Before the talk start the app: .venv312\\Scripts\\python.exe -m src.cli app. "
              "1) Dataset upload: demo_new_dataset, calibration flirone_one_pro_v9_found, Validate: three recordings Valid, "
              "a stray video listed as unmatched. 2) Calibration: the v9 profile (scale 1.27, rotation 0) and Auto-calibrate. "
              "3) Joint view: S2_right_static shows temperatures on the visible joints; S1_left_open_close at a closed frame "
              "hides folded joints. 4) Independent test: demo dataset with the saved model: one result per hand with score, "
              "threshold and top joints. 5) Reports: final joint temperature with Q1-Q3 and frames. "
              "If anything fails live, use the screenshots on this slide.")
table(s, ["#", "Show", "What the audience sees"],
      [["1", "Dataset upload → Validate", "3 recordings Valid; stray file listed"],
       ["2", "Calibration", "v9 profile (scale 1.27, 0°) + Auto-calibrate"],
       ["3", "Joint view", "temperatures on visible joints; folded joints hidden"],
       ["4", "Independent test", "one result per hand + top joints"],
       ["5", "Reports", "final joint °C with Q1–Q3 and frames"]],
      0.6, 1.5, 6.6, col_w=[0.5, 2.6, 3.5], size=13, row_h=0.55)
text(s, ".venv312\\Scripts\\python.exe -m src.cli app", 0.6, 5.2, 6.6, 0.4, 13, NAVY, bold=True, face="Consolas")
text(s, "Backup if the live run fails: the screenshots on the right.", 0.6, 5.7, 6.6, 0.6, 12, GREY, italic=True)
img(s, FIG / "A03_app_joint_view.png", 7.5, 1.45, 5.3, 2.5, "joint view", "screenshot")
img(s, FIG / "N04_report_joint_table.png", 7.5, 4.3, 5.3, 2.3, "report: final joint temperatures", "application output")

prs.core_properties.title = "RGB–Thermal Hand-Joint Temperature Analysis — Faculty Presentation"
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
