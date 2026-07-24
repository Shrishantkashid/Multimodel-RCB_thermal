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

### Dataset Format & Image Characteristics
- **Image Resolution**: Both RGB and Thermal images have a high resolution of 1920x1080 pixels.
- **Annotation Format**: The dataset uses the standard COCO JSON format, containing bounding box coordinates (`[x, y, width, height]`), image metadata, and category assignments for the "person" class.
- **Characteristics of RGB Images**: RGB images capture rich visual texture and color details but are highly susceptible to poor illumination (e.g., nighttime) and bad weather conditions.
- **Characteristics of Thermal Images**: Thermal images capture heat signatures, making them robust against illumination changes and darkness. However, they lack fine visual textures and color information, making it difficult to distinguish objects purely by texture.

### RGB-Thermal Alignment
Visual alignment verification across sample image pairs confirms that the RGB and Thermal modalities are generally well-aligned, with pedestrian positions matching consistently across both views. No severe camera shifts were observed in the checked samples.

### Key Challenges Identified
- Small/tiny pedestrian instances dominate the dataset (see scale distribution, where small and medium objects make up 75-87% of all instances).
- Extreme illumination variation in RGB images (e.g., very dark nighttime scenes) where the thermal modality will be essential for fallback.
- Some minor occlusion cases and cluttered backgrounds that can make small pedestrian detection difficult without cross-modality fusion.

### Visualizations
See `Stage1_analysis/outputs/visualizations/` for 20 annotated RGB-Thermal pairs.

### Thermal Preprocessing Analysis
An exploratory analysis was conducted on 10 sample thermal images to evaluate the effect of contrast enhancement techniques (Global Histogram Equalization and CLAHE). 
- **Histogram Equalization** stretched the contrast significantly but often amplified background noise.
- **CLAHE** successfully improved local contrast while keeping the background natural, improving small pedestrian visibility.
- Descriptive contrast statistics indicate an overall increase in contrast variance for preprocessed images.

**Conclusion**: Preprocessing is currently treated as an experimental candidate. The raw dataset remains unchanged, and the decision to adopt CLAHE will depend on quantitative detection metrics during later model evaluation rather than visual appearance alone.
