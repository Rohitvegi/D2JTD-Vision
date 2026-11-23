# Figure and table index (technical report)

Figure files are in `deliverables/figures/`.

## Figures

- Figure 1. Dataset folder structure and naming convention (S<subject>_<hand>_<task>).
- Figure 2. Sample input frames from the audit: RGB, thermal MP4 and radiometric .npy of the same frame (measured data).
- Figure 3. Audit: frame timing per recording (measured). Header frame rates were unreliable; measured rates are used.
- Figure 4. System architecture as implemented.
- Figure 5. Application module architecture: Python modules and the GUI pages that use them.
- Figure 6. End-to-end pipeline: data flow from the three input files to scores and reports.
- Figure 7. Coordinate systems and the RGB→thermal transform (diagram).
- Figure 8. Registration on a real frame: RGB-derived hand mask warped onto the thermal mask (measured).
- Figure 9. MediaPipe's 21 landmarks on a real RGB frame, with the 14 analysed joints labelled (measured).
- Figure 10. Joint localisation error against 777 hand-clicked joints, raw vs corrected, static vs open–close (measured).
- Figure 11. Thermal hand segmentation on a real frame: Otsu only vs Otsu + hysteresis, and the eroded core used for sampling (measured).
- Figure 12. Handling of accessories, occlusion and genuine hot spots (diagram of the implemented rules).
- Figure 13. The 14 joint ROIs and the dorsum region on a real thermal frame, with the measured temperatures and D2JTD (measured).
- Figure 14. Quality-control overlay produced by the pipeline for one frame (measured).
- Figure 15. Joint view of a static recording: visible joints with ROI temperature on the RGB photo (colour = D2JTD) and the ROIs on the thermal image (measured).
- Figure 16. Joint view during hand closing: folded joints are not shown and get no temperature (measured).
- Figure 17. Held-out Dice per recording, static vs open–close (measured, 68 recordings).
- Figure 18. Error budget from the three-cornered hat: random scatter of each joint estimator (measured).
- Figure 19. Annotation QC contact sheet: flagged and worst frames with clicks and pipeline joints (measured).
- Figure 20. MDC95 per joint against the 0.4 °C criterion (measured).
- Figure 21. Bland–Altman plot [30] of the 10-joint mean D2JTD, first vs second half (measured).
- Figure 22. Healthy D2JTD per joint and subject (measured, 17 subjects).
- Figure 23. Static vs open–close: absolute temperatures and D2JTD per hand (measured).
- Figure 24. Share of frames with hidden joints by joint type and task (v8 visibility rule, 68 recordings).
- Figure 25. Sensitivity versus simulated warming per model and pattern (simulation on healthy data).
- Figure 26. Model comparison chart from the v7 evaluation (simulation on healthy data).
- Figure 27. Feature-set ablation (simulation): adding spatial contrast helps; adding all 122 features hurts.
- Figure 28. Learning curves: specificity and simulated sensitivity versus number of training subjects.
- Figure 29. Processing time per frame by step (laptop CPU, measured).
- Figure 30. Real-time demo output: per-frame joint status from a saved model (measured on the laptop).
- Figure 31. Correct outputs on six unseen healthy volunteers: all 14 joints measured, temperature at each joint, score below the threshold (measured).
- Figure 32. One unseen hand: the real recording passes; the same hand with a simulated +2 °C index MCP is flagged and the explanation locates the warm joint.
- Figure 33. All 34 hands: score relative to the threshold for the real recording (green) and with the simulated warm joint (red triangles).
- Figure 34. Dataset upload and validation workflow.
- Figure 35. Application: dataset validation page with per-recording status (screenshot of the real application).
- Figure 36. Nested leave-one-subject-out validation: outer loop for evaluation, inner loop for the threshold.
- Figure 37. Locked independent-test workflow.
- Figure 38. Application: independent-test page with locked run, metrics and warnings (screenshot of the real application).
- Figure 39. Application dashboard: datasets, models and recent runs (screenshot of the real application).
- Figure 40. Joint-view page: RGB with visible joints and temperatures, thermal with ROIs, joint table (screenshot).
- Figure 41. Joint-view page on a closing hand: hidden joints are omitted and explained in the table (screenshot).
- Figure 42. Models page: saved model bundles with their validation metadata (screenshot).
- Figure 43. Reports page with the built-in PDF viewer showing a generated dataset report (screenshot).

## Tables

- Table 1. Implemented components.
- Table 2. Research questions and pre-registered pass criteria.
- Table 3. Dataset summary (measured by the audit).
- Table 4. Software layers (Python 3.12; OpenCV, MediaPipe, NumPy, pandas, scikit-learn, PySide6).
- Table 5. Joint localisation accuracy (pixels on the 480×640 .npy grid; one sensor pixel = 4 px).
- Table 6. Implemented feature groups (129 per recording) and their split-half stability (34 static hands; median over the group).
- Table 7. Implemented healthy-reference models (input: D2JTD, or D2JTD + spatial, standardised).
- Table 8. Summary of results against the pre-registered criteria. *RQ3 partly by construction.
- Table 9. Split-half repeatability of D2JTD per joint (34 static hands, measured).
- Table 10. Healthy D2JTD reference (static recordings, measured).
- Table 11. Model comparison, nested leave-one-subject-out (34 healthy hands; sensitivities are SIMULATED).
- Table 12. Feature-set ablation, nested LOSO (sensitivities SIMULATED).
- Table 13. The 24 system outputs.
