"""python-docx helpers: headings, paragraphs, bullets, tables, figures with captions, equation images, TOC."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_ORIENT  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

EQ_DIR = Path("deliverables/figures/equations")
NAVY = RGBColor(0x1B, 0x2A, 0x41)
GREY = RGBColor(0x52, 0x51, 0x4E)


class Doc:
    def __init__(self):
        self.d = Document()
        sec = self.d.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.3))
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        st = self.d.styles
        st["Normal"].font.name = "Calibri"
        st["Normal"].font.size = Pt(11)
        st["Normal"].paragraph_format.space_after = Pt(6)
        st["Normal"].paragraph_format.line_spacing = 1.15
        for lvl, size in ((1, 18), (2, 14), (3, 12)):
            h = st[f"Heading {lvl}"]
            h.font.name = "Cambria"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = NAVY
            h.paragraph_format.space_before = Pt(16 if lvl == 1 else 10)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
            rpr = h.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                fonts.set(qn(a), "Cambria")
        cap = st["Caption"]
        cap.font.name = "Calibri"
        cap.font.size = Pt(9.5)
        cap.font.italic = True
        cap.font.color.rgb = GREY
        self.fig_n = 0
        self.tab_n = 0
        self.eq_n = 0
        self.figs = {}
        self._footer()

    def _footer(self):
        p = self.d.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("RGB–Thermal Hand-Joint Temperature Analysis — Technical Research Report · page ")
        r.font.size = Pt(8.5)
        r.font.color.rgb = GREY
        self._field(p, "PAGE", 8.5)

    def _field(self, p, code, size=None):
        r = p.add_run()
        if size:
            r.font.size = Pt(size)
        for tag, text in (("begin", None), (None, code), ("separate", None), (None, "1"), ("end", None)):
            if tag:
                e = OxmlElement("w:fldChar")
                e.set(qn("w:fldCharType"), tag)
                r._r.append(e)
            elif text == code:
                e = OxmlElement("w:instrText")
                e.set(qn("xml:space"), "preserve")
                e.text = f" {code} "
                r._r.append(e)
            else:
                e = OxmlElement("w:t")
                e.text = text
                r._r.append(e)

    # ---------------------------------------------------------------- content
    def title_page(self, title, subtitle, lines):
        for _ in range(6):
            self.d.add_paragraph()
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "Cambria", Pt(26), True, NAVY
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(subtitle)
        r.font.size, r.font.color.rgb = Pt(14), GREY
        for _ in range(4):
            self.d.add_paragraph()
        for ln in lines:
            p = self.d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(ln).font.size = Pt(11)
        self.page_break()

    def toc(self):
        self.d.add_heading("Contents", 1)
        p = self.d.add_paragraph()
        self._field(p, 'TOC \\o "1-2" \\h \\z \\u')
        self.page_break()

    def h(self, text, lvl=1):
        return self.d.add_heading(text, lvl)

    def p(self, text, bold_lead=None, italic=False):
        para = self.d.add_paragraph()
        if bold_lead:
            para.add_run(bold_lead).bold = True
        r = para.add_run(text)
        r.italic = italic
        return para

    def bullets(self, items, style="List Bullet"):
        for it in items:
            para = self.d.add_paragraph(style=style)
            if isinstance(it, tuple):
                para.add_run(it[0]).bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)

    def numbered(self, items):
        """Numbered list that always restarts at 1 (explicit numbers; Word list numbering would continue across lists)."""
        for n, it in enumerate(items, 1):
            para = self.d.add_paragraph()
            para.paragraph_format.left_indent = Cm(0.75)
            para.paragraph_format.first_line_indent = Cm(-0.6)
            para.paragraph_format.space_after = Pt(3)
            para.add_run(f"{n}.\t")
            if isinstance(it, tuple):
                para.add_run(it[0]).bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)

    def note(self, text):
        para = self.d.add_paragraph()
        r = para.add_run(text)
        r.italic = True
        r.font.color.rgb = GREY
        r.font.size = Pt(10)

    def table(self, header, rows, caption, widths=None, size=9.5):
        self.tab_n += 1
        cp = self.d.add_paragraph(f"Table {self.tab_n}. {caption}", style="Caption")
        cp.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            run = c.paragraphs[0].add_run(str(htxt))
            run.bold = True
            run.font.size = Pt(size)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "1B2A41")
            c._tc.get_or_add_tcPr().append(shd)
        for k, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v))
                run.font.size = Pt(size)
                if k % 2:
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "EEF2F7")
                    cells[i]._tc.get_or_add_tcPr().append(shd)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        trpr = t.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader")
        hdr.set(qn("w:val"), "true")
        trpr.append(hdr)
        self.d.add_paragraph()
        return self.tab_n

    def figure(self, path, caption, width_cm=15.5):
        if not Path(path).is_file():
            self.note(f"[Missing figure: {path}]")
            return None
        self.fig_n += 1
        self.figs[Path(path).stem] = self.fig_n
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        self.d.add_paragraph(f"Figure {self.fig_n}. {caption}", style="Caption")
        return self.fig_n

    def eq(self, latex, explanation=None, size=17):
        EQ_DIR.mkdir(parents=True, exist_ok=True)
        self.eq_n += 1
        f = EQ_DIR / f"eq{self.eq_n:02d}.png"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=size)
        fig.savefig(f, dpi=220, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
        plt.close(fig)
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = bool(explanation)
        run = para.add_run()
        from PIL import Image
        w, h = Image.open(f).size
        h_cm = min(1.6, h / 220 * 2.54)
        max_w = 14.6                                         # leave room for the equation number
        if w / h * h_cm > max_w:                             # too wide for the page: scale down to fit
            run.add_picture(str(f), width=Cm(max_w))
        else:
            run.add_picture(str(f), height=Cm(h_cm))
        para.add_run(f"\t({self.eq_n})").font.size = Pt(10)
        if explanation:
            self.p(explanation)
        return self.eq_n

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        self.d.core_properties.title = "RGB–Thermal Hand-Joint Temperature Analysis — Technical Research Report"
        self.d.core_properties.subject = "Research and screening-support system; not a diagnostic device"
        self.d.save(path)
