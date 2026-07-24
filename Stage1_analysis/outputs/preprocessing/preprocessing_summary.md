# Stage 1 Thermal Preprocessing Analysis

## Techniques Evaluated

- Original thermal imagery
- Global Histogram Equalization
- CLAHE

## Purpose

Preprocessing was explored to determine whether thermal contrast enhancement may improve visual separation of pedestrians in difficult thermal scenes.

## Method

Representative thermal samples were compared using original images, global histogram equalization, and CLAHE.

## Observations

- Global Histogram Equalization significantly stretched the contrast but often exaggerated background noise.
- CLAHE increased local contrast without severely overexposing the background, enhancing edges of objects in the scene.
- Both CLAHE and global histogram equalization increased image contrast in several evaluated samples, but the impact on pedestrian detection performance must be validated experimentally during later model evaluation.

## Decision

The raw dataset remains unchanged.

Preprocessing is currently treated as an experimental candidate rather than automatically applied to the complete training pipeline.

Whether preprocessing should be adopted will later be determined using detection metrics rather than visual appearance alone.
