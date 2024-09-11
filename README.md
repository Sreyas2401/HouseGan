# Developable Surfaces

This repository contains my research and codebase for converting 2D architectural floorplans into 3D models using developable surface techniques. The project focuses on two main components: room extraction and 2D-SDF-Net, a neural network designed to approximate signed distance functions (SDFs) of 2D polygons. The extracted rooms are processed and converted into 3D developable surfaces, making the transformation of floorplans more geometrically accurate.

## Room Extraction (Data Preprocessing)

The first step in converting a 2D floorplan into a 3D model is identifying individual rooms. I've use OpenCV to invert the colors of floorplans from the rplan dataset and extract contours to identify and isolate the rooms for further processing.

### Steps to run Room Extraction:

1. ```pip install -r requirements.txt``` (this is recommended to be done in a virtual environment)
2. ```python invert_image.py```
3. ```python seperate_rooms.py```

## 2D-SDF-Net

2D-SDF-Net is a neural network designed to approximate the 2D signed distance functions of polygons in the extracted rooms.
The network structure references DeepSDF: <https://github.com/facebookresearch/DeepSDF> and mintpancake's neural network.

By applying a combination of L1 loss and Hessian loss, the network is optimized to handle floorplans with circular boundaries, ensuring the generated 3D surfaces are developable.

The SDF output represents the distance from any point in the 2D space to the nearest boundary, which is crucial for developing the floorplan into a smooth 3D surface.

### Steps to run 2D-SDF-Net:

1. ```pip install -r requirements.txt``` (this is also recommended to be run in the same virtual environment)
2. ```python drawer.py```
3. ```python sampler.py```
4. ```python trainer.py```
5. ```python renderer.py```
