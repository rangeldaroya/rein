# REIN
REIN: Flexible Mesh Generation from Point Clouds to be presented at the 2020 WiCV CVPR Workshop.

This is the code implementation for the Recurrent Edge Inference Network (REIN) to generate meshes from point clouds.

## Dataset
- Download the dataset ![here](https://drive.google.com/file/d/1DTUWK-Xn9-I4R5_GcehIvcFn2y_YLa7X/view?usp=sharing)


## Train the Model
- Train the model from scratch (as a default, it will train on the dataset ShapeNet Split): <code>python3 main.py</code>
  - To specify a specific dataset to train on: <code>python3 main.py --dataset 'shapenet-split'</code>
- To load and train on a pretrained model: <code>python3 main.py --pretrain True --pretrain_ae_path <ae_path> --pretrain_rnn_path <rnn_path> --pretrain_output_path <output_path></code>
  - <code>ae_path</code> is the path for the autoencoder network
  - <code>rnn_path</code> is the path for the rnn network
  - <code>output_path</code> is the path for the output layer

## Network Model
![network architecture of REIN](https://github.com/rangeldaroya/rein/blob/master/images/network_architecture.png)
## Sample Mesh Generation on Test Set

### Quantitative Results on ShapeNet and ModelNet10
|             |            |            |            |            |            |            |
| :---------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
|                                    ShapeNet                                               |
| Algorithm   |       Hull |  Butterfly |   Midpoint |       Hull |  Butterfly |   Midpoint |
| BPA         |     0.0052 |     0.0075 |     0.0177 |     0.6224 |     0.8143 |     0.6995 |
| PSR         |     0.0871 |     0.0227 |     0.0200 |     0.5795 |     0.7762 |     0.7367 |
| Ours (REIN) | **0.0003** | **0.0033** | **0.0028** | **0.8317** | **0.8181** | **0.8313** |
|             |            |            |            |            |            |            |
|               ModelNet10                                                                    |
| Algorithm   |       Hull |  Butterfly |   Midpoint |       Hull |  Butterfly |   Midpoint |
| BPA         |     0.0088 |     0.0106 |     0.0573 |     0.7210 |     0.8062 |     0.6032 |
| PSR         |     0.0292 |     0.0224 |     0.0292 | **0.8273** |     0.7938 |     0.7583 |
| Ours (REIN) | **0.0050** | **0.0056** | **0.0073** |     0.8259 | **0.8285** | **0.8288** |

### Qualitative Results on ShapeNet Patched

The image below shows sample results of REIN by reconstructing by parts on the ShapeNet dataset. The generated meshes are compared with results from Ball Pivoting Algorithm (BPA) and Poisson Surface Reconstruction (PSR).

![sample results on shapenet patched](https://github.com/rangeldaroya/rein/blob/master/images/patching_blender_shapeNet_results.png)
