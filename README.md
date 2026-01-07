# 3D-CXR-Nodule-Vis: Voxel-Based 3D Visualization of Lung Nodules

**Official repository for the paper:** *Voxel-Based 3D Visualization Approaches for Studying Lung Nodules from Chest X-Rays or CT Scans*

**Authors:** Adnan Mustafic, Mahmoud Melkemi, Karim Hammoudi  
**Affiliation:** IRIMAS, OMEGA, Université de Haute-Alsace, Mulhouse, France

**Link to the paper:** https://doi.org/10.1109/IV68685.2025.00075

## 📄 Abstract

Lung nodule inspection is crucial for the effective diagnosis and treatment of lung cancer. While CT scans allow for detailed 3D analysis, they are expensive and carry a higher radiation dose. Traditional Chest X-Rays (CXR) are safer and more common but lack depth information.

This project introduces two voxel-based approaches to visualize lung nodules in 3D:

1.  **VVLN-CT:** A ground-truth generation method using **CT Scans** and segmentation masks.
2.  **AVVLN-DCXR:** An attention-guided approach using **Dual Chest X-Rays (Frontal & Lateral)** to reconstruct 3D nodule representations without CT data.

These methods are designed to facilitate clinical training and evaluation through an interactive virtual reality environment.

-----

## 🚀 Features

  * **Dual-Modality Support:** Generate 3D visualizations from either CT slices or 2D CXR pairs.
  * **Attention-Guided Reconstruction:** Uses CNN-derived attention maps (Grad-CAM) to localize nodules from X-rays.
  * **Voxel-Based Rendering:** Creates a 3D voxel grid representing the lung nodule within a 3D lung model.
  * **Interactive Controls:** Includes sliders for hue thresholding and intersection parameters to refine visualizations in real-time.
  * **Data Processing Pipeline:** Tools to process LIDC-IDRI data, creating ground truths by projecting 3D CT segmentations onto 2D planes.

-----

## 🛠️ Methodology

### 1\. VVLN-CT (CT Scan Approach)

This method processes CT DICOM data to create a voxel-based 3D ground truth.

  * **Segmentation:** Uses `lungmask` for lung extraction.
  * **Processing:** Applies sliding window filtering and pixel-wise major voting to handle discontinuous segmentations.
  * **Visualization:** Stacks segmented slices to form a regular voxel grid.

### 2\. AVVLN-DCXR (Dual CXR Approach)

This method reconstructs 3D information from two orthogonal 2D images (Frontal and Lateral).

  * **Lung Segmentation:**
      * **Frontal:** Uses `TorchXRayVision`.
      * **Lateral:** Uses a pre-trained `VI-FCN` model.
  * **Attention Mapping:** Generates attention maps using a DenseNet-121 model and Grad-CAM.
  * **3D Reconstruction:**
    1.  Thresholds attention maps by hue to identify regions of interest.
    2.  Traces candidate lines (rays) orthogonal to the image planes.
    3.  Calculates intersections of these lines in 3D space to generate voxels.

-----

## 📦 Prerequisites

The project relies on several deep learning and imaging libraries. Ensure you have the following installed:

  * **Python 3.x**
  * **PyTorch**
  * **TorchXRayVision**
  * **lungmask**
  * **pydicom**
  * **NumPy / SciPy**
  * **Matplotlib** (for 2D plotting)
  * *(Optional)* **VisPy** or similar for 3D rendering (depending on the specific visualization script used).

### Installation

```bash
# Clone the repository
git clone https://github.com/Adn-an/3D-CXR-Nodule-Vis.git
cd 3D-CXR-Nodule-Vis

# Install dependencies (example)
pip install torch torchvision torchxrayvision lungmask pydicom numpy
```

## 📚 Citation

If you use this code or methodology in your research, please cite the following paper:

```bibtex
@INPROCEEDINGS{11216804,
  author={Mustafic, Adnan and Melkemi, Mahmoud and Hammoudi, Karim},
  booktitle={2025 29th International Conference Information Visualisation (IV)}, 
  title={Voxel-Based 3D Visualization Approaches for Studying Lung Nodules from Chest X-Rays or CT Scans}, 
  year={2025},
  pages={396-401},
  doi={10.1109/IV68685.2025.00075}
}
```
