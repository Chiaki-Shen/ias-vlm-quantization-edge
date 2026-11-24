# DriveLM sign_box: records lost at low-bit vision

Records that Config 2 (F16 vision) got right and Config 8 (Q4_0) or Config 11 (Q3_K_M) got wrong. Language decoder Q8_0 in all configs. Checked by eye on Nov 24, 2026 (crops around the red box).

| idx | Lost at | Ground truth | Low-bit answer | DriveLM object | What is in the box |
|---|---|---|---|---|---|
| 203 | Q3 | No | Yes | Vehicle | Large white van, close, fully visible |
| 205 | Q3 | No | Yes | Pedestrian | Person walking behind a concrete barrier |
| 220 | Q4, Q3 | No | Yes | Vehicle | Bus, mid-distance, behind a car |
| 236 | Q3 | No | Yes | Vehicle | White bus at night, glare from street lights |
| 252 | Q4 | No | Yes | Vehicle | Small, distant white SUV |
| 291 | Q3 | No | Yes | Vehicle | Pickup truck in traffic, front view |
| 292 | Q3 | No | Yes | Vehicle | Small white vehicle, partly hidden by a tree |
| 298 | Q4, Q3 | No | Yes | Vehicle | Very small, distant white car (rain) |

Findings:

1. All 8 losses are "No" records (a vehicle or a person in the box) that the low-bit model calls a traffic sign or road barrier. No "Yes" record (real traffic element) was lost: Yes-class accuracy is 93.3% at F16 and 93.3 to 96.0% at Q4/Q3.
2. So the low-bit errors are not misread signs or text. The model loses the ability to recognise what is inside the box and falls back to "Yes" (Yes answers: 85 at F16, 88 at Q4_0, 91 at Q3_K_M).
3. 4 of 8 boxes are small or distant objects (220, 252, 292, 298) and 7 of 8 are white or light-coloured, which is plausibly confusable with a sign panel once fine detail is lost. Others are large and obvious (203 van, 291 truck), so it is not only a size effect.
4. The net change is small and not significant at n=150 (Q4_0 -2.0 [-4.7, +0.0], Q3_K_M -1.3 [-6.0, +2.7] points vs F16), because low-bit configs also gain a few records (Q3_K_M: 7 lost, 5 gained).
