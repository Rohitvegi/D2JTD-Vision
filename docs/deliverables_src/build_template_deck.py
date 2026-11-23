"""Faculty presentation in the format of the user's template (AI PROJECT.pptx, IIT Hyderabad style).

Keeps the template's slides, layouts, fonts and logo; fills it with: architecture, sequence of updates (problems ->
changes -> results), formulas by module, innovation, individual outcome + LLM usage, demo.

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_template_deck.py
"""
import copy
import io
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt
from PIL import Image

TEMPLATE = Path(r"C:/Users/rohit/Downloads/AI PROJECT.pptx")
OUT = Path("deliverables/AI_PROJECT_RGB_Thermal_IITH.pptx")
FIG = Path("deliverables/figures")
M = json.load(open("deliverables/supporting/metrics_verified.json"))
V3, V5 = M["v3"], M["v5"]
PR = json.load(open("deliverables/supporting/prediction_examples_summary.json"))
BEN = json.load(open("reports/v9_calibration/benchmark_find_calibration.json"))["results"]
B3, B9 = BEN["v3: fitted per recording"], BEN["found calibration (this module)"]
PROD = json.load(open("models/calibration/flirone_one_pro_v9_found.json"))
FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)
ORANGE = RGBColor(0xC0, 0x50, 0x10)
TEAL = RGBColor(0x2E, 0x6E, 0x80)

prs = Presentation(str(TEMPLATE))
S = list(prs.slides)
s_title, s_agenda, s_pipe, s_bigimg, s_outcome, s_refs, s_end = S
LOGO = next(sh for sh in s_outcome.shapes if sh.shape_type == 13).image.blob
LAYOUT = {l.name: l for l in prs.slide_layouts}


def f(x, d=2):
    return f"{x:.{d}f}"


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def remove(shape):
    shape._element.getparent().remove(shape._element)


def logo(slide):
    slide.shapes.add_picture(io.BytesIO(LOGO), Inches(12.26), Inches(5.96), Inches(0.84), Inches(1.3))


def set_lines(tf, lines, size=None, bold_first=False, color=None):
    """Replace a text frame's paragraphs, keeping the first paragraph's formatting for every line."""
    first = tf.paragraphs[0]._p
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    proto = copy.deepcopy(first)
    for i, ln in enumerate(lines):
        p_el = first if i == 0 else copy.deepcopy(proto)
        if i:
            first.getparent().append(p_el)
    for p, ln in zip(tf.paragraphs, lines):
        runs = p.runs
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
        r = runs[0] if runs else p.add_run()
        if isinstance(ln, tuple):
            r.text = ln[0]
            r.font.bold = True
            r2 = p.add_run()
            r2.text = ln[1]
            r2.font.bold = False
            for rr in (r, r2):
                if size:
                    rr.font.size = Pt(size)
                if color:
                    rr.font.color.rgb = color
        else:
            r.text = ln
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = color


def textbox(slide, x, y, w, h, lines, size=13, bold=False, color=BLACK, font=FONT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, ln in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(ln, tuple):
            r = p.add_run()
            r.text = ln[0]
            r.font.bold = True
            r.font.size, r.font.name, r.font.color.rgb = Pt(size), font, color
            r = p.add_run()
            r.text = ln[1]
        else:
            r = p.add_run()
            r.text = ln
            r.font.bold = bold
        r.font.size, r.font.name, r.font.color.rgb = Pt(size), font, color
        p.space_after = Pt(4)
    return tb


def table_slide(title, header, rows, widths, size=10.5, note=""):
    s = prs.slides.add_slide(LAYOUT["Table"])
    s.shapes.title.text = title
    ph = next(p for p in s.placeholders if p.placeholder_format.type == 12)
    x, y, w = ph.left, ph.top, ph.width
    remove(ph)
    shape = s.shapes.add_table(len(rows) + 1, len(header), x, y, w, Inches(0.3) * (len(rows) + 1))
    t = shape.table
    for i, cw in enumerate(widths):
        t.columns[i].width = Inches(cw)
    for i, h in enumerate(header):
        c = t.cell(0, i)
        c.text = h
        for r in c.text_frame.paragraphs[0].runs:
            r.font.size, r.font.bold, r.font.name = Pt(size + 0.5), True, FONT
    for k, row in enumerate(rows):
        for i, v in enumerate(row):
            c = t.cell(k + 1, i)
            c.text = str(v)
            c.margin_top = c.margin_bottom = Inches(0.02)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.name = Pt(size), FONT
    logo(s)
    if note:
        notes(s, note)
    return s


# ============================================================ 1 title: unchanged
notes(s_title, "Project: arthritis screening research using the dorsal-to-joint temperature difference measured with a "
               "smartphone RGB + thermal camera.")

# ============================================================ 2 agenda
ph = next(p for p in s_agenda.placeholders if p.placeholder_format.idx == 1)
set_lines(ph.text_frame, ["Final architecture", "Key updates", "Key formulas", "How the output is obtained",
                          "Innovation implemented", "Individual outcome & LLM usage", "Demo", "References"])
ph.top, ph.height = Inches(2.5), Inches(3.6)

# ============================================================ 3 architecture (template 'Project pipeline' slide)
for sh in list(s_pipe.shapes):
    if sh.shape_type == 17 or (sh.shape_type == 13 and sh.width > Inches(2)):
        remove(sh)
s_pipe.shapes.title.text = "Final architecture"
t_ = s_pipe.shapes.title
t_.left, t_.width = Inches(0.12), Inches(2.5)
for p_ in t_.text_frame.paragraphs:
    for r_ in p_.runs:
        r_.font.size = Pt(15)
im = Image.open(FIG / "U01_architecture_overview.png")
w_in = 9.6
s_pipe.shapes.add_picture(str(FIG / "U01_architecture_overview.png"), Inches(2.62), Inches(0.75), Inches(w_in), Inches(w_in * im.height / im.width))
notes(s_pipe, "Top: RGB video and radiometric data pass the data audit and frame pairing. RGB branch: MediaPipe 21 hand points, "
              "quality and visibility checks, RGB hand mask. Thermal branch: Otsu + hysteresis, largest region, eroded skin core. "
              "The automatic calibration (v9) maps the RGB joints into the thermal image. Then 14 joint ROIs, temperature "
              "sampling (median per ROI per frame, then median across usable frames), features (D2JTD + spatial), the "
              "healthy-reference model and the outputs. Bottom: subject-wise training/evaluation and the new-data pipeline.")
remove_ids = [s_bigimg]                                   # old big pipeline picture: replaced by the architecture slide

# ============================================================ 4 key updates (one slide)
UPD = [
    ["Audit", "RGB and thermal frames mismatched; one recording's files swapped", "decode frames in order; automatic swap detection", "frames matched (sync 32/34 verified)"],
    ["v1", "Cold fingers lost from the thermal mask", "Otsu + hysteresis segmentation", "whole hand kept"],
    ["v2", "Joint regions fell off thin fingers", "radius capped at the finger width", "48 % → 98 % on skin"],
    ["v3", f"Joint position error {f(V3['joint_err_raw'], 1)} px", "correction learned from 777 clicked joints", f"{f(V3['joint_err_corr'], 1)} px (effect {f(V3['dT_corr'], 2)} °C)"],
    ["v7", "Thresholds tuned on the test hands", "nested leave-one-subject-out", "specificity 0.94 on unseen people"],
    ["v8", "Folded joints were given temperatures", "visibility rule (outline, depth, bend)", "hidden joints not measured"],
    ["v9", "Calibration needed for any video", "automatic calibration: scale, rotation, per-frame shift", f"{B9['joint_err_median']:.2f} vs {B3['joint_err_median']:.2f} px; 12/12 distortions"],
    ["1.0.0", "Use on another PC", "installer, health check, command line", "clean install passed"],
]
s_upd = table_slide("Key updates", ["Version", "Problem faced", "How we changed it", "Result"], UPD, [1.0, 3.9, 3.9, 2.7], size=13,
                    note="The most important changes, in order. Each answered a problem observed in the previous version and was kept "
                         "only if it improved a measured result.")

# ============================================================ 5 key formulas (one slide)
FORM = [
    ["calibration", "p_th = s·Rot(α)·(k·p − c) + c + t", f"RGB joint → thermal pixel (s = {PROD['scale']:.2f})"],
    ["rois", "T_j,f = median{ T(x,y) ∈ ROI ∩ skin }", "joint temperature in one frame"],
    ["feature_extraction", "T_j = median over usable frames of T_j,f", "final joint temperature (+ Q1–Q3)"],
    ["feature_extraction", "D2JTD = T_dorsum − T_joint", f"main feature (healthy mean {f(V3['mean10'], 2)} °C)"],
    ["anomaly_models", "D_M = √( (x − μ)ᵀ Σ̂⁻¹ (x − μ) )", "distance from healthy (Ledoit–Wolf)"],
    ["anomaly_models", "flag if D_M > θ,  θ = 95th percentile", f"specificity {0.94:.2f} on unseen people"],
    ["segmentation", "Dice = 2|A ∩ B| / (|A| + |B|)", f"mapping check: {f(V3['dice_median'], 3)}"],
    ["analysis", "MDC95 = 1.96 · SD(x₂ − x₁)", f"repeatability: {f(V3['mdc'], 2)} °C"],
]
s_form = table_slide("Key formulas", ["Module", "Formula", "Meaning / value"], FORM, [2.4, 5.2, 3.9], size=13,
                     note="The eight formulas that define the measurement and the decision. All others are listed in the complete "
                          "project report (formula compendium).")

# ============================================================ 8 innovation (template 'Expected outcome' slide, same box style)
box_proto = copy.deepcopy(next(sh for sh in s_outcome.shapes if sh.shape_type == 1)._element)
arrow_proto = copy.deepcopy(next(sh for sh in s_outcome.shapes if sh.shape_type == 9)._element)
for sh in list(s_outcome.shapes):
    if sh.shape_type in (1, 9, 17):
        remove(sh)
s_outcome.shapes.title.text = "Innovation implemented"
s_outcome.shapes.title.width = Inches(10.5)
INNOV = [
    ("Calibration from the hand itself", "Finds image area, scale, rotation and per-frame shift — no calibration target.",
     f"12/12 known distortions; {B9['joint_err_median']:.2f} vs {B3['joint_err_median']:.2f} px on unseen people"),
    ("Visibility-aware measurement", "Folded or covered joints are detected (outline, depth, bend) and not measured.",
     "66 % of open–close frames had a hidden joint"),
    ("Annotation-driven correction", "777 hand-clicked joints revealed a systematic along-finger shift; learned and removed.",
     f"{f(V3['joint_err_raw'], 1)} → {f(V3['joint_err_corr'], 1)} px; {f(V3['dT_corr'], 2)} °C effect"),
    ("Leakage-free healthy reference", "Nested leave-one-subject-out model and threshold; per-joint explanation.",
     f"{PR['real_correct']}/{PR['n_hands']} healthy correct; {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints"),
    ("One robust value per joint", "Median over usable frames, with Q1–Q3 and frame count in every report.",
     f"repeatable to {f(V3['mdc'], 2)} °C (MDC95)"),
    ("Deployable and verifiable", "16 validation checks, locked tests, auto-calibrate, installer and health check.",
     "clean install: workflow + 50 tests passed"),
]
for i, (head, what, ev) in enumerate(INNOV):
    x, y = 0.7 + (i % 3) * 4.05, 2.2 + (i // 3) * 2.45
    el = copy.deepcopy(box_proto)
    s_outcome.shapes._spTree.append(el)
    box = s_outcome.shapes[-1]
    box.left, box.top, box.width, box.height = Inches(x), Inches(y), Inches(3.5), Inches(0.62)
    set_lines(box.text_frame, [head], size=14)
    textbox(s_outcome, x, y + 0.66, 3.5, 1.7, [what, ""], size=12)
    textbox(s_outcome, x, y + 1.45, 3.5, 0.6, ev, size=11.5, bold=True, color=ORANGE)
notes(s_outcome, "Six innovations, each with its measured evidence: calibration from the hand itself; visibility-aware "
                 "measurement; annotation-driven joint correction; leakage-free healthy reference with explanation; one robust "
                 "value per joint; deployable and verifiable software.")

# ============================================================ how the output is obtained
s_out = prs.slides.add_slide(LAYOUT["Title and two content"])
for ph in [p for p in s_out.placeholders if p.placeholder_format.type == 7]:
    remove(ph)
s_out.shapes.title.text = "How the output is obtained"
t_ = s_out.shapes.title
t_.left, t_.top, t_.height, t_.width = s_outcome.shapes.title.left, s_outcome.shapes.title.top, s_outcome.shapes.title.height, Inches(10.5)
STEPS = [("Measure every frame", "Joints found in RGB are calibrated into the thermal image; each joint's °C is the median of its skin pixels. Hidden joints get no value."),
         ("One value per joint", "Median over the usable frames, with Q1–Q3 and the number of frames."),
         ("Feature vector", "D2JTD (back of hand − joint) and spatial contrasts: 32 numbers per hand."),
         ("Score vs healthy", "Distance from the healthy reference (Ledoit–Wolf Mahalanobis) compared with the threshold θ."),
         ("Result + report", "Normal measurement or potential anomaly, the joints that drive it; PDF, Excel and joint view.")]
for i, (head, what) in enumerate(STEPS):
    x = 0.6 + i * 2.5
    el = copy.deepcopy(box_proto)
    s_out.shapes._spTree.append(el)
    b = s_out.shapes[-1]
    b.left, b.top, b.width, b.height = Inches(x), Inches(2.35), Inches(2.15), Inches(0.75)
    set_lines(b.text_frame, [head], size=14)
    textbox(s_out, x, 3.25, 2.15, 2.4, what, size=12.5)
    if i < len(STEPS) - 1:
        a = copy.deepcopy(arrow_proto)
        s_out.shapes._spTree.append(a)
        ar = s_out.shapes[-1]
        ar.left, ar.top, ar.width, ar.height = Inches(x + 2.15), Inches(2.725), Inches(0.35), 0
textbox(s_out, 0.6, 5.85, 11.3, 0.5, [("Categories: ", "Normal measurement · Potential anomaly · Outside model experience · "
                                       "Low-confidence · Invalid measurement · Insufficient data")], size=12.5, color=TEAL)
logo(s_out)
notes(s_out, "How one result is produced from a recording: every frame is measured; each joint becomes one value (median "
             "over usable frames); the values form a feature vector; the model measures its distance from healthy hands and "
             "compares it with the threshold; the application writes the category, the joints responsible and the reports.")

# ============================================================ 9 individual outcome + LLM usage
s_llm = prs.slides.add_slide(LAYOUT["Title and two content"])
s_llm.shapes.title.text = "Individual outcome & LLM usage"
t_ = s_llm.shapes.title
t_.left, t_.top, t_.height = s_outcome.shapes.title.left, s_outcome.shapes.title.top, s_outcome.shapes.title.height
t_.width = Inches(11.0)
for ph in [p for p in s_llm.placeholders if p.placeholder_format.type == 7]:
    remove(ph)
textbox(s_llm, 0.7, 2.0, 5.9, 0.4, "Individual outcome (done by me)", size=16, bold=True, color=TEAL)
textbox(s_llm, 0.7, 2.45, 5.9, 4.3, [
    ("Project direction: ", "set the objectives and research questions, decided every design step and approved each version."),
    ("Data and ground truth: ", "ran the dataset audit and hand-clicked 777 joint centres on 136 thermal frames."),
    ("Problems found by me: ", "wrong frame mapping, folded joints showing temperatures, reports shown only as paths, need for calibration on any video, one value per joint."),
    ("Verification: ", "checked every output, figure and number; ran the application, tests and demo."),
    ("Delivered: ", "measurement pipeline, calibration, healthy-reference model, desktop app (release 1.0.0), reports."),
    ("Learned: ", "radiometric thermal imaging, hand landmarking, camera mapping, leakage-free evaluation, deployment."),
], size=12)
textbox(s_llm, 6.85, 2.0, 5.3, 0.4, "LLM usage (Claude, Anthropic)", size=16, bold=True, color=TEAL)
textbox(s_llm, 6.85, 2.45, 5.3, 4.3, [
    ("Reference: ", "methods, libraries and literature leads (e.g. MediaPipe, Otsu, Ledoit–Wolf, ICC)."),
    ("Error verification: ", "debugging, automated tests (50), known-distortion self-test, clean-install check; caught a model-loading bug and a benchmark coordinate error."),
    ("Implementation support: ", "code and document drafts written under my direction and reviewed by me."),
    ("Not used for: ", "ground-truth labels (clicked by me) or results — every number comes from saved outputs."),
], size=12)
logo(s_llm)
notes(s_llm, "Left: what I did. Right: how the LLM assistant was used — for reference, for error verification, and for "
             "implementation support under my direction; labels and results are not produced by the LLM.")

# ============================================================ 10 demo
s_demo = prs.slides.add_slide(LAYOUT["Title and content with image right"])
s_demo.shapes.title.text = "Demo"
for ph in [p for p in s_demo.placeholders if p.placeholder_format.type in (7, 18)]:
    remove(ph)
textbox(s_demo, 0.7, 2.0, 6.4, 4.6, [
    ("1. Dataset upload → Validate: ", "3 recordings Valid; stray file listed."),
    ("2. Calibration: ", f"v9 profile (scale {PROD['scale']:.2f}, {PROD['angle_deg']:.0f}°) + Auto-calibrate."),
    ("3. Joint view: ", "temperatures on visible joints; folded joints hidden."),
    ("4. Independent test: ", "saved model → one result per hand + top joints."),
    ("5. Reports: ", "final joint °C with Q1–Q3 and number of frames."),
    "",
    ("Start: ", ".venv312\\Scripts\\python.exe -m src.cli app"),
], size=14)
img = FIG / "A03_app_joint_view.png"
iw, ih = Image.open(img).size
ww = 5.0
s_demo.shapes.add_picture(str(img), Inches(7.6), Inches(1.9), Inches(ww), Inches(ww * ih / iw))
logo(s_demo)
notes(s_demo, "Demo script: start the app before the talk. 1) Dataset upload: demo_new_dataset, calibration "
              "flirone_one_pro_v9_found, Validate. 2) Calibration page. 3) Joint view: S2_right_static, then "
              "S1_left_open_close at a closed frame. 4) Independent test with the saved model. 5) Reports: the joint table. "
              "Fallback: the screenshot on this slide.")

# ============================================================ references (genuine sources used by the project)
for sh in list(s_refs.shapes):
    if sh.shape_type == 17:
        remove(sh)
textbox(s_refs, 1.0, 1.95, 11.0, 4.0, [
    "1.  Y. Qiao, P. Hamann, C. Robson, D. Cosker, “Dorsal-to-Joint Temperature Differences (D2JTD) extraction from RGB-T images for remote rheumatoid arthritis monitoring,” Biomed. Signal Process. Control, vol. 112, 108482, 2026. doi:10.1016/j.bspc.2025.108482",
    "2.  F. Zhang et al., “MediaPipe Hands: on-device real-time hand tracking,” arXiv:2006.10214, 2020.",
    "3.  G. Schiavon et al., “Infrared thermography for the evaluation of inflammatory and degenerative joint diseases: a systematic review,” Cartilage, vol. 13, 2021. doi:10.1177/19476035211063862",
    "4.  N. Borojevic et al., “Thermography hand temperature distribution in rheumatoid arthritis and osteoarthritis,” Periodicum Biologorum, vol. 113, pp. 445–448, 2011.",
    "5.  R. K. Ahalya et al., “RANet: a custom CNN model and quanvolutional neural network for the automated detection of rheumatoid arthritis in hand thermal images,” Sci. Rep., vol. 13, 15638, 2023. doi:10.1038/s41598-023-42111-3",
    "6.  N. Otsu, “A threshold selection method from gray-level histograms,” IEEE Trans. Syst., Man, Cybern., vol. 9, no. 1, pp. 62–66, 1979.",
    "7.  O. Ledoit, M. Wolf, “A well-conditioned estimator for large-dimensional covariance matrices,” J. Multivariate Anal., vol. 88, no. 2, pp. 365–411, 2004.",
    "8.  P. E. Shrout, J. L. Fleiss, “Intraclass correlations: uses in assessing rater reliability,” Psychol. Bull., vol. 86, no. 2, pp. 420–428, 1979.",
    "9.  F. Pedregosa et al., “Scikit-learn: machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
], size=12)

# ============================================================ order + remove old slides
order = [s_title, s_agenda, s_pipe, s_upd, s_form, s_out, s_outcome, s_llm, s_demo, s_refs, s_end]
sld = prs.slides._sldIdLst
order_ids = [str(s.slide_id) for s in order]
remove_ids_str = [str(s.slide_id) for s in remove_ids]
for sid in remove_ids_str:
    el = next(e for e in sld if e.get("id") == sid)
    prs.part.drop_rel(el.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
    sld.remove(el)
els = {e.get("id"): e for e in sld}
for e in list(sld):
    sld.remove(e)
for sid in order_ids:
    sld.append(els[sid])
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
raise SystemExit(0)
for s in remove_ids:
    el = next(e for e in sld if e.get("id") == str(s.slide_id))
    prs.part.drop_rel(el.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
    sld.remove(el)
els = {e.get("id"): e for e in sld}
for e in list(sld):
    sld.remove(e)
for s in order:
    sld.append(els[str(s.slide_id)])
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
