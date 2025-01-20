# Structure From Motion - 3D Reconstruction

This project implements a **Structure From Motion (SfM)** pipeline for 3D reconstruction from 2D images. It uses a combination of feature matching, fundamental matrix estimation, camera pose recovery, and **bundle adjustment** (via GTSAM) to compute a 3D point cloud and camera positions. 

The pipeline processes a dataset of Buddha statue images and outputs the reconstructed 3D point cloud and camera poses.

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Feature Matching**: Detects and matches features using SIFT across consecutive images.
- **Fundamental Matrix Calculation**: Implements a robust fundamental matrix estimation with normalization and RANSAC.
- **Camera Pose Recovery**: Recovers relative camera poses using the essential matrix.
- **3D Triangulation**: Computes 3D points using camera poses and matched points.
- **Bundle Adjustment**: Refines 3D points and camera poses using GTSAM optimization.
- **3D Visualization**: Renders the reconstructed point cloud and camera poses using Open3D.
- **Custom Implementation**: Includes a custom implementation of the 8-point algorithm, Sampson distance, and RANSAC.

---

## Folder Structure

```plaintext
├── 3D Recon-SFM/                # Contains final 3D reconstruction outputs (point cloud screenshots, camera poses)
│   ├── <output_images>          # Output images of 3D reconstruction
├── buddha_images/               # Input dataset of 22 images of a Buddha statue
│   ├── buddha_001.png
│   ├── buddha_002.png
│   └── ...
├── SFM_FINAL.ipynb              # Jupyter Notebook implementation of the SfM pipeline
├── SFM_FINAL.py                 # Python script for the SfM pipeline
├── README.md                    # Project documentation (this file)

