## Stage 2: Independent Modality Baselines and QFDet Benchmark

### Baseline Architecture Selection
For Stage 2, we are utilizing the organizer-provided **QFDet** (Query-based Feature Detection) implementation. Training from scratch is prohibited; all evaluations are performed strictly using the supplied pretrained QFDet weights to establish our baseline for RGB-only, Thermal-only, and the full RGB-Thermal detector. 

### Environment & Configuration
The baseline configurations interface directly with the QFDet codebase. 
- The dataset remains strictly read-only.
- Only the `pedestrian` class is evaluated.

### Baseline Evaluation Results (Validation / Test)
*Metrics are pending evaluation using the provided QFDet weights.*

| Metric | RGB-only | Thermal-only | QFDet (RGB-Thermal) |
|--------|----------|--------------|---------------------|
| mAP    | TBD      | TBD          | TBD                 |
| mAP50  | TBD      | TBD          | TBD                 |
| mAP75  | TBD      | TBD          | TBD                 |
| mAPS   | TBD      | TBD          | TBD                 |
| mAPM   | TBD      | TBD          | TBD                 |
| mAPL   | TBD      | TBD          | TBD                 |
| FPS    | TBD      | TBD          | TBD                 |
| Inference Time | TBD | TBD | TBD |
| Model Size | TBD | TBD | TBD |
| Parameters | TBD | TBD | TBD |
| FLOPs  | TBD      | TBD          | TBD                 |

### Low-Light Subset Evaluation (Additional Analysis)
To quantitatively explore the strengths of each modality, an additional experimental subset was created for low-light scenes (defined experimentally as scenes where the mean RGB brightness < 60). 
- **Subset Creation:** The original test annotations remain unchanged; a derived subset was created strictly for this analysis.
- **Results:** Performance across modalities on this low-light subset is pending evaluation.

### Recommendation for Stage 3
TBD. Analysis of QFDet's performance will dictate the specific cross-modal fusion strategy developed in Stage 3.
