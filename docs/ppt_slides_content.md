# PPT Content - Stage 1

## Slide 1: Title Slide
**Project/team name:** MedhaDrishti 2026 — AI-Based Object Recognition
**Challenge:** RGB-Thermal Pedestrian Detection
**Team members:** [Insert Your Team Member Names Here]

---

## Slide 2: The Problem, In Your Own Words
- "Pedestrian detection fails in poor lighting/weather using RGB alone. Thermal imaging compensates — but combining both effectively, especially for small/distant pedestrians, is the open challenge."
- **Why this matters:** Critical for reliable surveillance, autonomous systems, and search & rescue operations.
*(Visual Idea: Insert a side-by-side RGB vs thermal pair of the same scene here from your outputs)*

---

## Slide 3: Dataset Overview
| Split | Images | Pedestrian Instances | Avg per Image |
|---|---|---|---|
| Train | 1200 | 8138 | 6.78 |
| Val | 300 | 2337 | 7.79 |
| Test | 200 | 2068 | 10.34 |

---

## Slide 4: Pedestrian Scale Distribution
*(Visual Idea: Create a simple bar chart with the following data)*
- **Train**: Small: 9.9% | Medium: 67.0% | Large: 23.1%
- **Val**: Small: 18.1% | Medium: 68.1% | Large: 13.8%
- **Test**: Small: 25.6% | Medium: 61.4% | Large: 13.0%

**Callout line:** "Up to 25.6% of pedestrians in the test set fall in the 'small' category (<32² pixels) — this is the core challenge our fusion strategy must address."

---

## Slide 5: RGB-Thermal Pairs — Visual Evidence
*(Visual Idea: Insert 2-3 of your best annotated visualization pairs from `Stage1_analysis/outputs/visualizations/pair_XX.png`)*
**Caption Examples:**
- "Top: Pedestrian clearly visible in thermal but obscured in RGB due to lighting."
- "Bottom: Small/distant pedestrian correctly bounded in both modalities."

---

## Slide 6: Alignment Verification & Key Challenges
- **RGB-Thermal alignment verified** across sample pairs — consistent alignment observed with minimal shift.
- **Key challenges identified:**
  - Dominant small/tiny pedestrian instances (especially in test data).
  - Occasional occlusion cases and severe illumination variation in RGB.
  - Relying on thermal features when RGB is degraded.
