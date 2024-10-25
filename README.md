# Spherical Camera
Python tool that maps images taken from multiple cameras onto an egocentric sphere seen from the origin.

Adapted in part from Magnus Gaertner's Master's Thesis (ETH Zurich).

## Stages
1. Deprojects depth imagery to 3D world coordinates.
![Screenshot from 2024-10-25 11-25-24](https://github.com/user-attachments/assets/9cd61c81-7458-455c-a739-f9552ca5dce4)

2. Projects 3D points to cube, warps cube face pixels to "quad sphere".
![warp](https://github.com/user-attachments/assets/94bc2173-96c9-4575-a3c9-6f3b9157f480)

3. De-dupes points by choosing min-depth point per pixel in output sphere.
![Screenshot from 2024-10-25 11-26-01](https://github.com/user-attachments/assets/68aa0d99-dfc9-4f20-bc45-3040ab81e3f8)

## Setup
#### Clone the repository

```
git clone https://github.com/Kappibw/sphere_cam.git
cd sphere_cam
```

#### Create and activate the Conda environment
```
conda env create -f environment.yml
conda activate spherecam
```

#### Install the Python package in editable mode
```
pip install -e .
```

#### Verify the installation
```
python -c "import sphere_cam"
```


## Usage
To run on a single test image:
`python test_single_image.py`
