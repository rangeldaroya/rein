# REIN
REIN: Flexible Mesh Generation from Point Clouds to be presented at the 2020 WiCV CVPR Workshop.

This is the code implementation for the Recurrent Edge Inference Network (REIN) to generate meshes from point clouds.

## Dataset

## Train the Model
- Train the model from scratch (as a default, it will train on the dataset ShapeNet Split) by running <code>python3 main.py</code>
  - To specify a specific dataset to train on, run <code>python3 main.py --dataset 'shapenet-split'</code>


## Network Model
![network architecture of REIN](https://github.com/rangeldaroya/rein/blob/master/images/network_architecture.png)
## Sample Mesh Generation on Test Set
The image below shows sample results of REIN by reconstructing by parts on the ShapeNet dataset. The generated meshes are compared with results from Ball Pivoting Algorithm (BPA) and Poisson Surface Reconstruction (PSR).

![sample results on shapenet patched](https://github.com/rangeldaroya/rein/blob/master/images/patching_blender_shapeNet_results.png)
