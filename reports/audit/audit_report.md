# Dataset audit report

Dataset root: `C:\Users\rohit\Downloads\archive\FLIROne`  
Audit time: 2026-11-21 12:47 · run took 8.3 min

## Summary

| Item | Value |
| --- | --- |
| Subjects found | 17 (S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17) |
| Recordings expected / complete | 68 / 68 |
| Recordings with any issue | 1 |
| RGB resolution | 640×480 |
| Radiometric shape (per frame) | (N, H=640, W=480) — portrait; MP4s show it letterboxed (resize into box, do not transpose) |
| Radiometric dtype | float32 |
| Frames per recording (.npy) | 692 – 1527 (median 802) |
| RGB FPS (measured) | 7.80 – 8.77 (median 8.31) |
| RGB FPS (header) | 5.94 – 8.77 (median 7.10) |
| RGB duration (s) | 90.6 – 179.6 (median 113.5) |
| Frame counts equal (RGB = thermal MP4 = .npy) | 68 of 68 |
| Temperature min / max (°C) | 17.86 – 25.73 (median 24.52) / 31.21 – 37.88 (median 35.24) |
| Mean frame temperature (°C) | 24.37 – 29.53 (median 27.42) |
| Recordings with NaN/inf | 0 |
| Thermal MP4 ↔ .npy spatial match r (fitted to letterbox) | -0.584 – 1.000 (median 0.998) |
| Thermal MP4 ↔ .npy frame alignment | {'aligned': 67, 'undetermined (too little motion)': 1} |
| Thermal MP4 ↔ .npy worst single frame r (all frames checked) | -0.645 – 1.000 (median 0.997) |
| Frames where thermal MP4 ≠ .npy (r<0.95), total | 833 |
| cv2 seek error at middle frame, RGB / thermal (frames) | -30 – +30 (median +18) / -30 – +30 (median +17) |
| Thermal MP4 content box | {'y0-480 x140-500': 67, 'y0-480 x140-504': 1} |
| RGB content box | {'y0-480 x140-500': 67, 'y0-480 x140-501': 1} |

## Duplicates

None found.

## Unexpected files or folders

None.

## Recordings with issues

| Recording | Issues |
| --- | --- |
| S11_left_static | thermal MP4 and .npy match weakly (r=-0.58) | 833 thermal MP4 frame(s) differ from .npy (r<0.95; first: 0,1,2,3,4,5,6,7,8,9) |

## Frame geometry

- Each .npy frame is a portrait image, H=640 x W=480, already upright. Do NOT transpose it.
- Both MP4s are 480x640 frames with black bars; the real image sits in the content box above
  (a 3:4 portrait area). To compare with a video, resize the .npy frame into that box.
- Never jump to a frame with cv2 `CAP_PROP_POS_FRAMES`: on these files the seek lands on
  the wrong frame (see 'cv2 seek error' above). Decode in order and count frames; then
  video frame i corresponds to .npy frame i.

## How to read the figures

- `*_samples.png`: frames 0, middle and last. Column 4 draws the thermal hand outline on the RGB
  frame WITHOUT registration, so any offset you see is what Stage 5 must correct.
- `*_timestamps.png`: frame intervals (variable frame rate shows as spikes) and the
  RGB − thermal timestamp difference (flat at 0 = same clock).
- `dataset_temperature_histogram.png`: background and hand form two humps; their gap sets the mask threshold.

## Not available in this dataset

Emissivity, reflected temperature, camera-to-hand distance, room temperature, humidity and
acclimatisation time are not recorded. No diagnosis or inflammation labels are present.