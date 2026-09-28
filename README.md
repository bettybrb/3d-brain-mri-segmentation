# 3D Brain MRI Segmentation

A deep learning pipeline for multi-modal 3D brain tissue segmentation using a compact 3D U-Net, developed and evaluated on the MRBrainS13 dataset.

The system segments volumetric MRI scans into cerebrospinal fluid (CSF), gray matter and white matter using three complementary MRI modalities, patch-based training and full-volume sliding-window inference.

## Overview

Medical image segmentation presents additional challenges compared with conventional 2D image classification: MRI scans are volumetric, GPU memory requirements are high, labelled subjects are limited and tissue classes are heavily imbalanced.

This project addresses these constraints with a memory-efficient 3D segmentation pipeline designed around a compact 3D U-Net.

## Input Data

The model uses three aligned MRI modalities as input channels:

- T1
- T1-IR
- T2-FLAIR

The segmentation target contains four coarse classes:

- background
- cerebrospinal fluid (CSF)
- gray matter
- white matter

Each modality is independently z-score normalised over non-zero voxels before the three volumes are stacked into a multi-channel input.

## 3D U-Net

A compact 3D U-Net processes volumetric MRI patches using 3D convolutions.

The network uses a base channel size of 16, allowing the model to retain 3D spatial context while remaining computationally practical for volumetric training.

## Patch-Based Training

Training operates on 64 x 64 x 32 voxel patches rather than complete MRI volumes.

Foreground-biased sampling increases the probability that training patches contain CSF, gray matter or white matter instead of being dominated by background voxels.

This reduces GPU memory requirements while exposing the network to more useful foreground examples.

## Loss Function

The final model combines Dice loss with Cross Entropy:

`Loss = Dice Loss + Cross Entropy Loss`

Dice provides overlap-based supervision while Cross Entropy provides voxel-level class supervision.

## Data Augmentation

The training pipeline applies volumetric augmentation including:

- random flips
- Gaussian noise
- intensity scaling
- intensity shifting

These augmentations increase training variation in a setting with only five labelled training subjects.

## Sliding-Window Inference

Although the model is trained on patches, evaluation is performed on complete MRI volumes.

Sliding-window inference moves the trained network across the full 3D scan and combines overlapping predictions to reconstruct the complete segmentation volume.

## Leave-One-Out Cross-Validation

Because only five labelled training subjects were available, the main evaluation uses five-fold leave-one-subject-out cross-validation.

For each fold, four subjects are used for training and the remaining subject is held out for validation.

### LOOCV Results

| Tissue | Mean Dice |
| --- | ---: |
| CSF | 0.721 |
| Gray matter | 0.715 |
| White matter | 0.734 |
| **Mean foreground** | **0.723 ± 0.045** |

The variation between folds highlights the difficulty of generalising across subjects when very little labelled 3D medical data is available.

## Ablation Experiment

A fold-1 comparison evaluated a simpler Dice-only, no-augmentation setup against the improved Dice + Cross Entropy and augmentation pipeline.

Mean foreground Dice increased from **0.322 to 0.587** in this comparison.

Because loss and augmentation were changed together, this result should be interpreted as evidence for the combined training setup rather than attributing the improvement to either component individually.

## Final Generalisation Evaluation

After cross-validation, a final model was trained using all five labelled training subjects and evaluated on fifteen additional subjects with available coarse labels.

| Training | CSF | Gray Matter | White Matter | Mean Foreground |
| --- | ---: | ---: | ---: | ---: |
| 20 epochs | 0.771 | 0.771 | 0.795 | 0.779 |
| 40 epochs | **0.797** | **0.779** | **0.806** | **0.794** |

The 40-epoch model therefore achieved a mean foreground Dice of **0.794**.

## Qualitative Analysis

The project also evaluates segmentation quality visually using MRI slices, ground-truth masks, predicted masks and error maps.

The main failure mode is local confusion around tissue boundaries and smaller internal structures rather than failure to locate the brain or recover its overall tissue layout.

## Repository Structure

- `brain_mri_segmentation.ipynb` - complete preprocessing, training, LOOCV, inference and evaluation pipeline
- `outputs/figures/` - segmentation examples and evaluation plots
- `outputs/results/` - LOOCV, ablation and generalisation results
- `outputs/logs/` - training logs
- `requirements.txt` - Python dependencies

## Technologies

- Python
- PyTorch
- NumPy
- pandas
- NiBabel
- Matplotlib
- 3D U-Net
- NIfTI medical imaging

## Installation

```bash
pip install -r requirements.txt
```

The MRBrainS13 dataset is not included in this repository and must be obtained separately.

## Concepts Demonstrated

- 3D convolutional neural networks
- Medical image segmentation
- Multi-modal MRI processing
- 3D U-Net architectures
- NIfTI image processing
- Patch-based training
- Sliding-window inference
- Dice loss
- Cross Entropy loss
- Class imbalance handling
- Data augmentation
- Leave-one-out cross-validation
- Ablation experiments
- Quantitative and qualitative model evaluation

## Motivation

Training segmentation networks on 3D medical data requires balancing model capacity, limited labelled data and substantial memory requirements. This project demonstrates an end-to-end approach that combines multi-modal MRI information with memory-efficient patch training and full-volume inference, while evaluating generalisation across individual subjects rather than relying only on training performance.
