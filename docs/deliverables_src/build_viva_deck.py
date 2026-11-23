"""Viva presentation (.pptx) built from defence_content.SLIDES; speaker notes = script + likely questions.

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_viva_deck.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deck_helpers import (EMBER, FIG, NAVY, PALE, WHITE, GREY, PP_ALIGN, bullets, img, new, prs, table, text)  # noqa: E402
from defence_content import SLIDES  # noqa: E402

OUT = Path("deliverables/RGB_Thermal_Hand_Viva_Presentation.pptx")


def notes(s):
    out = "SAY:\n" + s["s"]
    if s.get("q"):
        out += "\n\nLIKELY QUESTIONS:"
        for q, a in s["q"]:
            out += f"\nQ: {q}\nA: {a}"
    return out


for n, s in enumerate(SLIDES, 1):
    lay = s["layout"]
    if lay == "title":
        sl = new(s["t"], dark=True, notes=notes(s))
        if n == 1:
            sl.shapes.title.top = sl.shapes.title.top + 1828800          # 2 in lower
            text(sl, "Project audit, architecture and technical defence", 0.6, 3.35, 11, 0.6, 22, PALE, face="Cambria")
            text(sl, ["FLIROne dataset · 17 healthy subjects · 68 recordings · 55,239 frames",
                      "Research and screening-support software — not a diagnostic device"], 0.6, 4.4, 11, 1.0, 15, WHITE)
        else:
            bullets(sl, s["b"], 0.6, 1.6, 12.1, 5.0, size=20, color=WHITE, gap=14)
        continue
    sl = new(s["t"], s["k"], notes=notes(s))
    if lay == "split":
        bullets(sl, s["b"], 0.6, 1.55, 5.6, 5.2, size=17, gap=12)
        img(sl, FIG / s["fig"], 6.5, 1.45, 6.3, 5.0, s.get("cap"), None)
    elif lay == "wide":
        img(sl, FIG / s["fig"], 0.6, 1.35, 12.1, 5.25, s.get("cap"), None)
    elif lay == "table":
        hdr, rows = s["tab"]
        ncol = len(hdr)
        widths = {2: [3.6, 8.5], 3: [2.6, 5.9, 3.6], 5: [3.6, 2.9, 1.9, 1.9, 1.8]}.get(ncol)
        table(sl, hdr, rows, 0.6, 1.55, 12.1, col_w=widths, size=14 if ncol < 5 else 13, row_h=0.58)
        if s.get("note"):
            text(sl, s["note"], 0.6, 6.55, 12.1, 0.4, 12, GREY, italic=True)

prs.core_properties.title = "RGB–Thermal Hand-Joint Temperature Analysis — Viva Presentation"
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
