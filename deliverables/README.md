# Deliverables: RGB–thermal hand-joint temperature analysis

| File | Contents |
|---|---|
| `RGB_Thermal_Hand_Complete_Project_Report.docx` / `.pdf` | **63-page complete report.** Chapters 1–13 are the same as the technical report. New: **14 Development history** (26 steps: problem → observation → change → effect), **15 Formula compendium** (31 formulas with variables and the values used or obtained), **16 Results compendium**, **17 Folded joints and how the model was trained**, **18 Open items**, then the conclusion. It contains 54 figures, 28 tables and 49 equations |
| `RGB_Thermal_Hand_Arthritis_Technical_Research_Report.docx` / `.pdf` | 55-page technical report (v9 / release 1.0.0) covering implemented components only. It has 14 chapters: pipeline stages, maths and metrics, results including correct predictions, **8: from frames to one temperature per joint**, **9: automatic calibration (v9) with both proofs**, upload/testing, training/validation/testing, system outputs, **13: deployment and how to use the deployed system**, and the conclusion. It contains 54 figures, 20 tables, 18 equations and 31 IEEE references |
| `RGB_Thermal_Hand_Arthritis_Project_Presentation.pptx` | 49 slides covering implemented components only. Slide 9 is the end-to-end architecture overview. Slides 46–48 follow the faculty format: innovation implemented, individual outcome and use of LLM assistance, and the live demo. They include three correct-prediction slides, two on frames → one temperature per joint, three on automatic calibration and three on deployment and how to use it (16:9) with speaker notes on every slide; native charts for repeatability, model comparison, ablation, visibility and compute |
| `RGB_Thermal_Hand_Project_Audit_and_Viva_Guide.docx` / `.pdf` | 44-page defence guide (v9 / release 1.0.0) in 15 parts: implementation audit (with status), dataset, workflow with the reason for each step, frames → one temperature per joint, architecture diagrams A–E, libraries, alignment maths with a worked example and the automatic calibration with both proofs, training, evaluation, new-data validation, robustness, deployment and how to use it, slide-by-slide script, 69 viva Q&A and a severity-ranked critical review |
| `RGB_Thermal_Hand_Viva_Presentation.pptx` | 28-slide viva deck (v9 / release 1.0.0). Each slide's notes hold what to say plus the likely questions and their answers |
| `Demo_Script.md` | Live-demo run sheet: what to click, what to say, the fallback if it fails, and likely questions |
| `figures/` | All figure assets. F = method diagrams and real-frame illustrations, G = graphs from measured or simulated data, R = result figures from the analysis runs, O = pipeline outputs, A = application screenshots, `equations/` = rendered equations |
| `supporting/metrics_verified.json` | Every number used, recomputed from the stored results |
| `supporting/Metric_Calculation_Sheet.xlsx` | The same values with source file, sample size and formula; per-joint and per-model sheets |
| `supporting/References.md` | Reference list in report order |
| `supporting/Figure_and_Table_Index.md` | All figure and table captions with their measured / simulated / diagram / screenshot tags |
| `supporting/Limitations_and_Future_Work.md` | Limitation-versus-improvement table with implementation status, and what must not be claimed |

Rebuild everything from the project root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv312\Scripts\python.exe docs\deliverables_src\collect_metrics.py
.venv312\Scripts\python.exe docs\deliverables_src\make_figures.py
.venv312\Scripts\python.exe docs\deliverables_src\make_prediction_figures.py
.venv312\Scripts\python.exe docs\deliverables_src\build_report.py
.venv312\Scripts\python.exe docs\deliverables_src\build_deck.py
```

After `build_report.py`, open the .docx in Word and update the table of contents (F9). Then export the PDF
(File → Export); the shipped PDF was produced this way through Word.

The scope of all results: 17 healthy volunteers. No arthritis patients were measured, so no diagnostic accuracy is reported.
Sensitivities are from simulated warming of healthy hands and are labelled as such.
