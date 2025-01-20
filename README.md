
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
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:RJ-23YP/Structure_From_Motion_3D_Reconstruction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Structure_From_Motion_3D_Reconstruction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure you have the following libraries installed:
   - OpenCV
   - NumPy
   - Matplotlib
   - GTSAM
   - Open3D
   - tqdm

5. **Optional**: Install Git LFS for large file handling (e.g., image datasets):
   ```bash
   git lfs install
   ```

---

## Usage

### Running the SfM Pipeline

1. Place input images in the `buddha_images` folder.
2. Run the Python script:
   ```bash
   python SFM_FINAL.py
   ```
   Or open and execute the Jupyter Notebook:
   ```bash
   SFM_FINAL.ipynb
   ```

3. Outputs:
   - Final 3D reconstruction images will be saved in the `3D Recon-SFM` folder.
   - Point cloud and camera poses will be rendered using Open3D.

---

## Pipeline Overview

1. **Feature Detection and Matching**:
   - Features are detected and matched across consecutive images using SIFT and FLANN-based matcher.
   - Matched points are filtered using Lowe's ratio test.

2. **Fundamental Matrix Estimation**:
   - A custom 8-point algorithm is used to compute the fundamental matrix with normalization and RANSAC.
   - Sampson distance is used for robust inlier detection.

3. **Camera Pose Recovery**:
   - Essential matrix is computed from the fundamental matrix and intrinsic camera parameters.
   - Camera poses (rotation and translation) are recovered for each image pair.

4. **3D Point Triangulation**:
   - 3D points are triangulated using projection matrices derived from camera poses.

5. **Bundle Adjustment with GTSAM**:
   - GTSAM is used to optimize camera poses and 3D points by minimizing reprojection error.

6. **Visualization**:
   - 3D point cloud and camera poses are visualized using Open3D.
   - Outputs are saved in the `3D Recon-SFM` folder.

---

## Technologies

- **Python**: Core programming language.
- **OpenCV**: For image processing, feature detection, and camera pose estimation.
- **GTSAM**: For bundle adjustment and pose graph optimization.
- **Matplotlib**: For epipolar line visualization and 3D trajectory plotting.
- **Open3D**: For rendering 3D point clouds and camera poses.
- **tqdm**: For progress visualization during processing.

---

## Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push the branch to your fork:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request on the main repository.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this project as per the license terms.

---

## Acknowledgments

- **GTSAM**: For bundle adjustment and pose optimization.
- **OpenCV**: For feature detection, matching, and pose recovery.
- **Open3D**: For 3D point cloud visualization.
- Special thanks to the open-source community for tools and resources used in this project.

---

If you have any questions or issues, feel free to [open an issue](https://github.com/RJ-23YP/Structure_From_Motion_3D_Reconstruction/issues).
```

This README reflects the pipeline, its functionalities, and the tools used. Let me know if further adjustments are needed!
