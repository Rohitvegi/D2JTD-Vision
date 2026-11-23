# Live demo script (about 5 minutes)

## Before the presentation (10 minutes earlier)
1. Open PowerShell in the project folder and start the application:
   ```powershell
   cd C:\Users\rohit\Downloads\rgb_thermal_hand_screening_stage1\rgb_thermal_hand_screening
   .venv312\Scripts\python.exe -m src.cli app
   ```
2. Optional health check (it must say READY):
   ```powershell
   .venv312\Scripts\python.exe -m src.cli doctor
   ```
3. Leave the app open on the **Dashboard** and switch back to the slides.

## During the demo slide

| # | Page | Do | Say |
|---|---|---|---|
| 1 | **Dataset upload** | Dataset: `demo_new_dataset_…`, Calibration: `flirone_one_pro_v9_found`, press **Validate** | "A new dataset with a different naming scheme: three recordings are matched and pass all 16 checks; a stray video is listed as unmatched." |
| 2 | **Calibration** | Click `flirone_one_pro_v9_found` | "This is the calibration the module found from the recordings: scale 1.27, no rotation, with a per-frame shift. For a new camera I would press Auto-calibrate." |
| 3 | **Joint view** | Original dataset → `S2_right_static`, Show | "The RGB photo shows each visible joint with its measured temperature. The thermal image shows the regions they were read from." |
| 3b | **Joint view** | `S1_left_open_close`, a frame with the hand closed | "When the hand closes, folded joints are not shown and get no temperature." |
| 4 | **Independent test** | Dataset `demo_new_dataset_…`, model `v7_mahalanobis_ledoit_wolf_…_172656`, **Run** | "The saved model is applied unchanged: one result per hand, with score, threshold and the joints that drive it." |
| 5 | **Reports** | Open the demo dataset report, scroll to the joint table | "Each joint has one final temperature: the median over its frames, with Q1–Q3 and the number of frames." |

Optional, on the command line, mapping one frame's joints into the thermal image:
```powershell
.venv312\Scripts\python.exe -m src.cli calibrate apply --profile models\calibration\flirone_one_pro_v9_found.json --rgb "C:\Users\rohit\Downloads\archive\FLIROne\S9\left hand\static_RGB.mp4" --npy "C:\Users\rohit\Downloads\archive\FLIROne\S9\left hand\static_Thermal_Radiometric_Sequence.npy" --frame 300
```

## If something fails live
* Say: "I'll show the recorded output instead." Then use the screenshots on the demo slide and the deployment slides (joint view, report table, upload page, models page, calibration page).
* For an app error, the details are in `logs\rgbthermal.log`.

## Likely questions during the demo
* **"Is this a diagnosis?"** No. It reports a distance from 17 healthy volunteers; no patients have been measured yet.
* **"How do you know the calibration is right?"** It recovered 12 of 12 known distortions, and on 12 unseen people joints land closer to the clicked positions (9.97 vs 11.04 px).
* **"Why one value per joint?"** The median over usable frames is robust to bad frames. It repeats within 0.26 °C.
