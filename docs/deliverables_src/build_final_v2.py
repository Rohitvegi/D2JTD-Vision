"""Add a metrics slide and an intermediate-outputs slide to the user's deck, speaker notes on every slide,
and write the slide-by-slide presentation script (.docx).

Input : C:/Users/rohit/Downloads/AI_PROJECT_RGB_Thermal_Final.pptx (the user's edited 12-slide deck; not modified)
Output: deliverables/AI_PROJECT_RGB_Thermal_Final_v2.pptx and deliverables/AI_PROJECT_Presentation_Script.docx

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_final_v2.py
"""
import copy
import io
import json
from pathlib import Path

from docx import Document
from docx.shared import Pt as DPt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image

SRC = Path(r"C:/Users/rohit/Downloads/AI_PROJECT_RGB_Thermal_Final.pptx")
OUT = Path("deliverables/AI_PROJECT_RGB_Thermal_Final_v2.pptx")
OUT_DOC = Path("deliverables/AI_PROJECT_Presentation_Script.docx")
STEPS = Path("deliverables/figures/steps")
M = json.load(open("deliverables/supporting/metrics_verified.json"))
V3 = M["v3"]
PR = json.load(open("deliverables/supporting/prediction_examples_summary.json"))
BEN = json.load(open("reports/v9_calibration/benchmark_find_calibration.json"))["results"]
B3, B9 = BEN["v3: fitted per recording"], BEN["found calibration (this module)"]
S3 = next(r for r in M["v6"]["ablation"] if r["feature_set"].startswith("S3") and r["model"] == "mahalanobis_ledoit_wolf")

FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)
ORANGE = RGBColor(0xC0, 0x50, 0x10)
TEAL = RGBColor(0x2E, 0x6E, 0x80)

prs = Presentation(str(SRC))
S = list(prs.slides)
s_title, s_agenda, s_arch, s_u1, s_u2, s_f1, s_f2, s_innov, s_llm, s_demo, s_refs, s_end = S
LAYOUT = {l.name: l for l in prs.slide_layouts}
LOGO = next(sh for sh in s_innov.shapes if sh.shape_type == 13).image.blob
BOX = next(sh for sh in s_innov.shapes if sh.shape_type == 1)._element


def remove(shape):
    shape._element.getparent().remove(shape._element)


def textbox(slide, x, y, w, h, lines, size=13, bold=False, color=BLACK, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, ln in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = align
        parts = ln if isinstance(ln, tuple) else (None, ln)
        for k, t in enumerate(parts):
            if t is None:
                continue
            r = p.add_run()
            r.text = t
            r.font.size, r.font.name, r.font.color.rgb = Pt(size), FONT, color
            r.font.bold = bold or (isinstance(ln, tuple) and k == 0)
        p.space_after = Pt(3)
    return tb


def box(slide, x, y, w, h, text, size, color=BLACK, bold=False):
    el = copy.deepcopy(BOX)
    slide.shapes._spTree.append(el)
    b = slide.shapes[-1]
    b.left, b.top, b.width, b.height = Inches(x), Inches(y), Inches(w), Inches(h)
    p = b.text_frame.paragraphs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    r = p.runs[0] if p.runs else p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, FONT
    p.alignment = PP_ALIGN.CENTER
    return b


def new_slide(title):
    s = prs.slides.add_slide(LAYOUT["Title and two content"])
    for ph in [p for p in s.placeholders if p.placeholder_format.idx != 0]:
        remove(ph)
    s.shapes.title.text = title
    t, ref = s.shapes.title, s_innov.shapes.title
    t.left, t.top, t.width, t.height = ref.left, ref.top, ref.width, ref.height
    s.shapes.add_picture(io.BytesIO(LOGO), Inches(12.26), Inches(5.96), Inches(0.84), Inches(1.3))
    return s


# ------------------------------------------------------------------ fixes to two values in the user's copy
for cell in (c for row in s_u1.shapes[1].table.rows for c in row.cells):
    for p in cell.text_frame.paragraphs:
        if p.text.startswith("8.3 px; 0.05"):
            rs = p.runs
            rs[0].text = f"{V3['joint_err_corr']:.1f} px"
            for r in rs[1:3]:
                r.text = ""
for sh in s_innov.shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            if p.runs and p.text.strip().startswith("healthy correct"):
                p.runs[0].text = f"{PR['real_correct']}/{PR['n_hands']} " + p.runs[0].text.lstrip()

# ------------------------------------------------------------------ agenda
tf = next(sh for sh in s_agenda.placeholders if sh.placeholder_format.idx != 0).text_frame
items = ["Final architecture", "Intermediate outputs", "Sequence of updates", "Formulas by module", "Metrics & results",
         "Innovation implemented", "Individual outcome & LLM usage", "Demo"]
first = tf.paragraphs[0]._p
for p in list(tf.paragraphs)[1:]:
    p._p.getparent().remove(p._p)
for i in range(1, len(items)):
    first.getparent().append(copy.deepcopy(first))
for p, t in zip(tf.paragraphs, items):
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    p.runs[0].text = t

# ------------------------------------------------------------------ intermediate outputs
s_mid = new_slide("Intermediate outputs")
TILES = [("t_1_landmarks.png", "1  RGB frame", "21 hand landmarks (MediaPipe)"),
         ("t_2_thermal.png", "2  Thermal frame", "radiometric °C per pixel"),
         ("t_3_mask.png", "3  Thermal hand mask", "Otsu + hysteresis keeps cold fingers"),
         ("t_4_overlay.png", "4  Calibration check", f"RGB hand on thermal; white = overlap (Dice {V3['dice_median']:.2f})"),
         ("t_5_rois.png", "5  Joint regions", "14 joint ROIs + back of hand (dorsum)"),
         ("t_6_jointview.png", "6  Joint temperatures", "°C per visible joint, shown on RGB")]
W, GAP, X0, Y0, HMAX = 1.68, 0.29, 0.55, 2.05, 2.75
for i, (fn, head, sub) in enumerate(TILES):
    x = X0 + i * (W + GAP)
    iw, ih = Image.open(STEPS / fn).size
    w = min(W, HMAX * iw / ih)
    h = w * ih / iw
    s_mid.shapes.add_picture(str(STEPS / fn), Inches(x + (W - w) / 2), Inches(Y0 + (HMAX - h)), Inches(w), Inches(h))
    textbox(s_mid, x - 0.05, Y0 + HMAX + 0.08, W + 0.1, 0.9, [head, sub], size=11.5, align=PP_ALIGN.CENTER)
    s_mid.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
    if i < len(TILES) - 1:
        textbox(s_mid, x + W - 0.04, Y0 + HMAX / 2 - 0.3, 0.42, 0.5, "➜", size=20, bold=True, color=TEAL)
        s_mid.shapes[-1].text_frame.margin_left = s_mid.shapes[-1].text_frame.margin_right = 0
textbox(s_mid, 0.55, 5.95, 11.4, 0.45,
        [("Then: ", "repeat for every frame → median over usable frames = one °C per joint → D2JTD features → "
                    "score vs healthy → result and report.")], size=12.5, color=TEAL)

# ------------------------------------------------------------------ metrics
s_met = new_slide("Metrics & results")
acc_n = PR["real_correct"] + PR["warmed_correct"]
CARDS = [
    [(f"{V3['hand_detection'] * 100:.2f} %", "Hand detection", "frames where the hand was found"),
     (f"{V3['dice_median']:.3f}", "Dice score", f"RGB ↔ thermal hand overlap ({B9['dice_median']:.3f} on unseen people)"),
     (f"{V3['joint_err_raw']:.1f} → {V3['joint_err_corr']:.1f} px", "Joint position error",
      f"vs 777 hand-clicked joints; {B9['joint_err_median']:.2f} px with v9 calibration"),
     (f"{V3['roi_on_skin'] * 100:.0f} %", "ROIs on skin", "joint regions inside the hand"),
     (f"{V3['ms_frame']:.1f} ms", "Speed per frame", f"≈ {V3['fps']:.0f} frames / s on a laptop CPU")],
    [(f"{acc_n / (2 * PR['n_hands']) * 100:.0f} %", "Accuracy", f"{acc_n}/{2 * PR['n_hands']} hands: healthy + simulated warm joint"),
     (f"{S3['specificity']:.2f}", "Specificity", f"{PR['real_correct']}/{PR['n_hands']} healthy hands called normal"),
     (f"{PR['warmed_correct']}/{PR['n_hands']}", "Sensitivity (simulated)",
      f"+2 °C index MCP flagged and located; any single joint {S3['sens_single_joint_2C']:.2f}"),
     (f"{V3['icc']:.3f}", "ICC (repeatability)", "two recordings of the same hand"),
     (f"{V3['mdc']:.2f} °C", "MDC95", "smallest real change detectable")],
]
for row, cards in enumerate(CARDS):
    y = 1.95 + row * 1.95
    for i, (val, name, what) in enumerate(cards):
        x = 0.55 + i * 2.35
        box(s_met, x, y, 2.15, 0.72, val, 20, color=ORANGE, bold=True)
        textbox(s_met, x - 0.03, y + 0.76, 2.21, 1.0, [name, what], size=11, align=PP_ALIGN.CENTER)
        s_met.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
textbox(s_met, 0.55, 5.95, 11.4, 0.5,
        ["Evaluation: nested leave-one-subject-out on 17 healthy volunteers (34 hands). Warm joints are simulated (+2 °C); "
         "no arthritis patients were measured, so no diagnostic accuracy is claimed."], size=11.5, color=TEAL)

# ------------------------------------------------------------------ order
lst = prs.slides._sldIdLst
ids = list(lst)
by_slide = dict(zip([id(s) for s in prs.slides], ids))
order = [s_title, s_agenda, s_arch, s_mid, s_u1, s_u2, s_f1, s_f2, s_met, s_innov, s_llm, s_demo, s_refs, s_end]
els = [by_slide[id(s)] for s in order]
for e in ids:
    lst.remove(e)
for e in els:
    lst.append(e)

# ------------------------------------------------------------------ script (speaker notes + document)
SCRIPT = [
    ("Title", "0:30",
     ["Good morning. Our project is arthritis screening support using the dorsal-to-joint temperature difference of the hand, "
      "under the guidance of Prof. Nagarajan Ganapathy.",
      "Idea: inflamed joints are warmer. We record the hand with a smartphone RGB + thermal camera (FLIR One), find the 14 finger "
      "joints in the RGB image, read their temperature from the thermal image and compare the hand with healthy hands.",
      "Data: public FLIROne dataset, 17 healthy volunteers, 68 recordings. It is screening support, not a diagnosis."]),
    ("Agenda", "0:20", ["Architecture first, then what each stage outputs, how the system evolved, the formulas, the metrics, "
                        "our innovations, my individual contribution and LLM usage, and a live demo."]),
    ("Final architecture", "1:30",
     ["Top row: the RGB video and the radiometric data are audited and paired frame by frame.",
      "RGB branch: MediaPipe finds 21 hand points; quality and visibility checks; RGB hand mask. Thermal branch: Otsu + hysteresis hand mask.",
      "Automatic calibration (v9) maps RGB joints into the thermal image: scale, rotation and a per-frame shift.",
      "Middle row: 14 joint regions → temperature sampling → features (D2JTD) → healthy-reference model → output.",
      "Bottom: training with leave-one-subject-out, and the pipeline the app uses on new data. No neural network is trained."]),
    ("Intermediate outputs", "1:30",
     ["This is what each stage produces on one real frame.",
      "1 – MediaPipe gives 21 landmarks on the RGB frame. 2 – the thermal frame is a temperature in °C for every pixel.",
      "3 – the thermal hand mask; hysteresis keeps the cold fingertips that plain Otsu loses.",
      f"4 – calibration check: the RGB hand outline mapped onto the thermal mask; white is overlap, median Dice {V3['dice_median']:.2f}.",
      "5 – 14 circular joint regions and the dorsum polygon; each region's temperature is the median of its skin pixels.",
      "6 – the temperatures shown back on the RGB photo. Repeated for every frame, then one value per joint is taken as the median over frames."]),
    ("Sequence of updates (1/2)", "1:15",
     ["Each row is a problem we observed, what we changed and the measured result.",
      "The audit fixed geometry: letterbox crop, k = 4/3 scaling and a swapped recording.",
      "v1 hysteresis kept cold fingers; v2 capped the ROI radius so 98 % of ROIs are on skin.",
      f"v3: joint error was {V3['joint_err_raw']:.1f} px; we hand-clicked 777 joints and learned an along-finger correction: "
      f"{V3['joint_err_corr']:.1f} px, only {V3['dT_corr']:.2f} °C effect on temperature.",
      "v7: thresholds were chosen on the same hands, so we moved to nested leave-one-subject-out: specificity 0.94 honestly."]),
    ("Sequence of updates (2/2)", "1:00",
     ["v6–v7: Ledoit–Wolf Mahalanobis with 32 D2JTD + spatial features was the most stable model.",
      "v8: folded joints were getting temperatures; the visibility rule hides them.",
      f"v9: automatic calibration for any video; on 12 unseen people {B9['joint_err_median']:.2f} px vs {B3['joint_err_median']:.2f} px, "
      "and it recovered 12/12 known distortions.",
      "1.0.0: installer, health check and command line; clean install passed."]),
    ("Formulas by module (1/2)", "1:00",
     ["Main formula: p_th = s·Rot(α)(k·p − c) + c + t — maps an RGB joint into the thermal image (s = 1.27).",
      "Dice measures mask overlap; Otsu and the hysteresis threshold build the thermal mask.",
      "Openness, bend angle and depth decide whether a joint is visible.",
      "ROI radius r is capped by the distance to the finger edge."]),
    ("Formulas by module (2/2)", "1:00",
     ["Per frame: joint °C = median of ROI skin pixels. Final: median over usable frames with Q1–Q3.",
      "D2JTD = dorsum − joint temperature; healthy mean is about 0.39 °C.",
      "Model: Mahalanobis distance with Ledoit–Wolf shrinkage; flag if above the 95th-percentile threshold.",
      "Per-joint z explains which joint drives a flag. ICC and MDC95 measure repeatability."]),
    ("Metrics & results", "1:30",
     [f"Measurement quality: hand found in {V3['hand_detection'] * 100:.2f} % of frames; Dice {V3['dice_median']:.3f}; "
      f"joint error {V3['joint_err_raw']:.1f} → {V3['joint_err_corr']:.1f} px; {V3['roi_on_skin'] * 100:.0f} % ROIs on skin; "
      f"{V3['ms_frame']:.1f} ms per frame.",
      f"Screening: specificity {S3['specificity']:.2f} — {PR['real_correct']} of {PR['n_hands']} healthy hands called normal; "
      f"all {PR['warmed_correct']} hands with a simulated +2 °C index MCP were flagged and the warm joint was located. "
      f"Combined accuracy {acc_n}/{2 * PR['n_hands']}.",
      f"Repeatability: ICC {V3['icc']:.3f}, MDC95 {V3['mdc']:.2f} °C.",
      "Important: all volunteers are healthy and the warm joints are simulated, so this is not a diagnostic accuracy."]),
    ("Innovation implemented", "1:00",
     ["Six innovations, each with evidence: calibration found from the hand itself; folded joints not measured; "
      "joint correction learned from clicked ground truth; leakage-free healthy reference with per-joint explanation; "
      "one robust value per joint; deployable and verifiable software."]),
    ("Individual outcome & LLM usage", "1:00",
     ["Left: what I did — set the objectives, ran the audit, clicked the 777 ground-truth joints, found the key problems, "
      "verified every output and delivered the system.",
      "Right: how I used the LLM — reference, error verification and implementation support under my direction; "
      "never for labels or results, every number comes from saved outputs."]),
    ("Demo", "5:00",
     ["Switch to the application (started before the talk with: .venv312\\Scripts\\python.exe -m src.cli app).",
      "1. Dataset upload → choose demo_new_dataset and calibration flirone_one_pro_v9_found → Validate: "
      "three recordings Valid, a stray video listed as unmatched.",
      "2. Calibration page → flirone_one_pro_v9_found: scale 1.27, rotation 0, per-frame shift. Auto-calibrate is for a new camera.",
      "3. Joint view → S2_right_static → Show: temperature on each visible joint. Then S1_left_open_close at a closed-hand frame: "
      "folded joints are hidden.",
      "4. Independent test → demo dataset + saved model v7_mahalanobis_ledoit_wolf → Run: one result per hand with score, "
      "threshold and top joints.",
      "5. Reports → open the demo report → joint table: final temperature per joint with Q1–Q3 and number of frames.",
      "If anything fails: say “I'll show the recorded output” and use the screenshot on this slide."]),
    ("References", "0:15", ["Key sources: Qiao et al. 2026 for D2JTD from RGB-T images; Borojevic 2011 and Ahalya 2023 for "
                            "hand thermography in arthritis."]),
    ("Thank you", "—", ["Thank you. Questions are welcome."]),
]
for s, (_, _, lines) in zip(order, SCRIPT):
    s.notes_slide.notes_text_frame.text = "\n".join(lines)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")

QA = [("Is this a diagnosis?", "No. It measures distance from 17 healthy volunteers; no patients were measured yet."),
      ("What is accuracy here?", f"{acc_n}/{2 * PR['n_hands']} hands correct: {PR['real_correct']}/{PR['n_hands']} healthy called normal "
                                 f"and {PR['warmed_correct']}/{PR['n_hands']} simulated warm hands flagged."),
      ("What is Dice?", f"Overlap of the RGB hand (after calibration) and the thermal hand: 2|A∩B|/(|A|+|B|); median {V3['dice_median']:.3f}."),
      ("How do you know the calibration is right?", f"12/12 known distortions recovered; on 12 unseen people {B9['joint_err_median']:.2f} vs "
                                                    f"{B3['joint_err_median']:.2f} px against clicked joints."),
      ("Why the median over frames?", f"Robust to bad frames; repeats within {V3['mdc']:.2f} °C (MDC95)."),
      ("Why no deep learning model?", "17 healthy subjects are too few to train a classifier; a healthy-reference distance needs no patient labels.")]
d = Document()
st = d.styles["Normal"]
st.font.name, st.font.size = "Calibri", DPt(11)
d.add_heading("AI Project DT5010 — Presentation and demo script", 0)
d.add_paragraph("Arthritis detection using the dorsal-to-joint temperature difference of the hand. "
                f"{len(order)} slides, about 15 minutes plus demo. The same text is in each slide's speaker notes.")
d.add_heading("Before you start", 1)
for t in ["Open PowerShell in the project folder and run: .venv312\\Scripts\\python.exe -m src.cli app",
          "Optional check: .venv312\\Scripts\\python.exe -m src.cli doctor (must say READY).",
          "Leave the app on the Dashboard and open the slides in Slide Show."]:
    d.add_paragraph(t, style="List Number")
for n, ((title, tm, lines), _) in enumerate(zip(SCRIPT, order), 1):
    d.add_heading(f"Slide {n} — {title}  ({tm})", 1)
    for ln in lines:
        d.add_paragraph(ln, style="List Bullet")
d.add_heading("Likely questions", 1)
for q, a in QA:
    p = d.add_paragraph()
    p.add_run(q + " ").bold = True
    p.add_run(a)
d.save(OUT_DOC)
print("saved", OUT_DOC)
