"""Build the project presentation (.pptx) with python-pptx. Numbers come from metrics_verified.json.

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_deck.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))

M = json.load(open("deliverables/supporting/metrics_verified.json"))
D, V3, V5, V6, V7, V8 = (M[k] for k in ("dataset", "v3", "v5", "v6", "v7", "v8"))
C = {r["model"]: r for r in V7["comparison"]}
PR = json.load(open("deliverables/supporting/prediction_examples_summary.json"))
AB = {(r["feature_set"], r["model"]): r for r in V6["ablation"]}
ST = {r["group"]: r for r in V6["stability"]}
FIG = Path("deliverables/figures")
OUT = Path("deliverables/RGB_Thermal_Hand_Arthritis_Project_Presentation.pptx")

NAVY = RGBColor(0x1B, 0x2A, 0x41)
EMBER = RGBColor(0xE4, 0x57, 0x2E)
COOL = RGBColor(0x2A, 0x78, 0xD6)
SAND = RGBColor(0xF6, 0xF3, 0xEE)
INK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x5F, 0x5E, 0x5A)
LINE = RGBColor(0xD9, 0xD5, 0xCE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD3, 0xE0)
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5


def f(x, d=2):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
LAY = prs.slide_layouts[5]          # Title Only
N = [0]


def _font(run, size, color=INK, bold=False, italic=False, face=BODY):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = face


def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def new(title, kicker=None, dark=False, notes=""):
    s = prs.slides.add_slide(LAY)
    N[0] += 1
    bg(s, NAVY if dark else SAND)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.55 if kicker else 0.35), Inches(12.1), Inches(0.9)
    tf = t.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    _font(r, 28 if len(title) <= 50 else 24, WHITE if dark else NAVY, bold=True, face=HEAD)
    if kicker:
        text(s, kicker.upper(), 0.6, 0.25, 9, 0.35, 11, EMBER, bold=True)
    if not dark:
        foot = text(s, f"RGB–thermal hand-joint temperature analysis  ·  {N[0]}", 0.6, 7.05, 8, 0.3, 9, GREY)
        foot.name = "footer"
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(s, t, x, y, w, h, size=16, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT, face=BODY,
         anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    lines = t if isinstance(t, list) else [t]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        _font(r, size, color, bold, italic, face)
    return tb


def bullets(s, items, x, y, w, h, size=17, color=INK, gap=8):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        bu_clr = pPr.makeelement(qn("a:buClr"), {})
        srgb = bu_clr.makeelement(qn("a:srgbClr"), {"val": "E4572E"})
        bu_clr.append(srgb)
        pPr.append(bu_clr)
        pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
        if isinstance(it, tuple):
            r = p.add_run()
            r.text = it[0]
            _font(r, size, color, bold=True)
            r = p.add_run()
            r.text = it[1]
            _font(r, size, color)
        else:
            r = p.add_run()
            r.text = it
            _font(r, size, color)
    return tb


TRIM = Path(tempfile.gettempdir()) / "deck_trimmed"


def trimmed(path):
    """Copy of the figure with near-white margins removed (content unchanged)."""
    TRIM.mkdir(exist_ok=True)
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).min(axis=2) < 245
    ys, xs = np.where(a)
    if len(xs) == 0:
        return path
    pad = 10
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(im.width, xs.max() + pad + 1), min(im.height, ys.max() + pad + 1))
    out = TRIM / Path(path).name
    im.crop(box).save(out)
    return out


def img(s, path, x, y, w, h, caption=None, tag=None):
    stem = Path(path).stem
    path = trimmed(path)
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    pic = s.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    pic.line.color.rgb = LINE
    pic.line.width = Pt(0.75)
    pic.name = stem
    el = pic._element.find(qn("p:nvPicPr")).find(qn("p:cNvPr"))
    el.set("descr", caption or stem)
    if caption:
        lab = (f"{tag.upper()} · " if tag else "") + caption
        text(s, lab, x, py + ph + 0.05, w, 0.35, 10.5, GREY, italic=True, align=PP_ALIGN.CENTER)
    return pic


def card(s, x, y, w, h, fill=WHITE, line=LINE):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def stat(s, value, label, x, y, w=2.9, h=1.55, color=EMBER, dark=False):
    card(s, x, y, w, h, fill=RGBColor(0x24, 0x36, 0x52) if dark else WHITE, line=RGBColor(0x33, 0x46, 0x63) if dark else LINE)
    text(s, value, x + 0.15, y + 0.12, w - 0.3, 0.75, 34, color, bold=True, face=HEAD)
    text(s, label, x + 0.15, y + 0.85, w - 0.3, h - 0.9, 12.5, PALE if dark else GREY)


def badge(s, n, x, y, d=0.55):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = EMBER
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    _font(r, 15, WHITE, bold=True)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def stage_slide(n, title, io_in, algo, io_out, figure, caption, tag, notes, extra=None):
    s = new(title, f"Pipeline stage {n} of 12", notes=notes)
    badge(s, n, 0.6, 1.55)
    rows = [("Input  ", io_in), ("Method  ", algo), ("Output  ", io_out)]
    bullets(s, rows, 1.35, 1.5, 4.9, 4.9, size=15, gap=10)
    if extra:
        extra(s)
    img(s, figure, 6.55, 1.45, 6.2, 5.0, caption, tag)
    return s


def table(s, header, rows, x, y, w, col_w=None, size=12, row_h=0.36):
    shp = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(row_h * (len(rows) + 1)))
    t = shp.table
    if col_w:
        for i, cw in enumerate(col_w):
            t.columns[i].width = Inches(cw)
    for i, htxt in enumerate(header):
        c = t.cell(0, i)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.text = ""
        r = c.text_frame.paragraphs[0].add_run()
        r.text = str(htxt)
        _font(r, size, WHITE, bold=True)
    for k, row in enumerate(rows):
        for i, v in enumerate(row):
            c = t.cell(k + 1, i)
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if k % 2 == 0 else RGBColor(0xEE, 0xF2, 0xF7)
            c.text = ""
            r = c.text_frame.paragraphs[0].add_run()
            r.text = str(v)
            bold = isinstance(v, str) and v in ("Pass", "Fail")
            _font(r, size, EMBER if v == "Fail" else INK, bold=bold)
            c.margin_top = c.margin_bottom = Inches(0.03)
    return t


def chart(s, kind, cats, series, x, y, w, h, title=None, colors=(COOL, EMBER, NAVY), pct_axis=False,
          labels=True, legend=None, num_fmt='0.00', vmax=None, gap=60, overlap=None):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.size = Pt(11)
    ch.font.name = BODY
    ch.font.color.rgb = GREY
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title
        for p in ch.chart_title.text_frame.paragraphs:
            for r in p.runs:
                _font(r, 13, NAVY, bold=True)
    ch.has_legend = legend is not None
    if legend is not None:
        ch.legend.position = legend
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(11)
    plot = ch.plots[0]
    plot.gap_width = gap
    if overlap is not None:
        plot.overlap = overlap
    for i, ser in enumerate(plot.series):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = colors[i % len(colors)]
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = num_fmt
        dl.number_format_is_linked = False
        dl.font.size = Pt(10)
        dl.font.color.rgb = INK
        if kind not in (XL_CHART_TYPE.BAR_STACKED, XL_CHART_TYPE.COLUMN_STACKED):
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
        else:
            dl.position = XL_LABEL_POSITION.CENTER
            dl.font.color.rgb = WHITE
            dl.font.bold = True
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(10)
    if pct_axis:
        va.tick_labels.number_format = '0%'
        va.tick_labels.number_format_is_linked = False
    if vmax is not None:
        va.maximum_scale = vmax
        va.minimum_scale = 0
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(10.5)
    ca.format.line.color.rgb = LINE
    return ch


