## Stage 1: Dataset Exploration, Analysis, and Preparation

### Dataset Statistics

| Split | Images | Pedestrian Instances | Avg Pedestrians / Image |
|-------|--------|----------------------|-------------------------|
| train | 1200   | 8138                 | 6.78                    |
| val   | 300    | 2337                 | 7.79                    |
| test  | 200    | 2068                 | 10.34                   |

### Pedestrian Scale Distribution

- **Train**: Small: 9.9%, Medium: 67.0%, Large: 23.1%
- **Val**: Small: 18.1%, Medium: 68.1%, Large: 13.8%
- **Test**: Small: 25.6%, Medium: 61.4%, Large: 13.0%

Note: the dataset shows a strong skew toward small and medium-scale pedestrians (the vast majority of instances are < 96x96 pixels), confirming the problem statement's emphasis on this as the primary detection challenge.

### RGB-Thermal Alignment
Visual alignment verification across sample image pairs confirms that the RGB and Thermal modalities are generally well-aligned, with pedestrian positions matching consistently across both views. No severe camera shifts were observed in the checked samples.

### Key Challenges Identified
- Small/tiny pedestrian instances dominate the dataset (see scale distribution, where small and medium objects make up 75-87% of all instances).
- Extreme illumination variation in RGB images (e.g., very dark nighttime scenes) where the thermal modality will be essential for fallback.
- Some minor occlusion cases and cluttered backgrounds that can make small pedestrian detection difficult without cross-modality fusion.

### Visualizations
See `Stage1_analysis/outputs/visualizations/` for 20 annotated RGB-Thermal pairs.
