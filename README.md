# Developable Surfaces

This repository consists my research on generating developable surfaces for converting 2D floorplans to 3D. It includes 2 parts :

1. Room Extraction
2. 2D-SDF-Net

## Room Extraction (Data Preprocessing)

I've used OpenCV to invert floorplans color from rplan dataset and then used contours to extract individual rooms.

To run this section, run the following steps in the workspace:

1. pip install -r requirements.txt (this is recommended to be done in a virtual environment)
2. python invert_image.py
3. python seperate_rooms.py

## 2D-SDF-Net

2D-SDF-Net is a neural network approximating the two-dimensional signed distance functions of polygons.
The network structure references DeepSDF: <https://github.com/facebookresearch/DeepSDF>.

I've applied hessian loss as an additional loss function to L1 loss function and optimized the model to be trained on floorplans with circular boundaries to make the surface developable.

2D-SDF-Net is trained on the extracted rooms from Room-Extraction directory to generate Signed Distance Fields for developing the surface from 2D to 3D.

To run this section, run the following steps in the workspace:

1. pip install -r requirements.txt (this is also recommended to be run in the same virtual environment)
2. python drawer.py
3. python sampler.py
4. python trainer.py
5. python renderer.py
