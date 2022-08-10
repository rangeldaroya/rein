# REIN
REIN: Flexible Mesh Generation from Point Clouds to be presented at the 2020 WiCV CVPR Workshop. Video can be found [here](https://youtu.be/ksIrWHyS39c).

This is the code implementation for the Recurrent Edge Inference Network (REIN) to generate meshes from point clouds.

## Dataset
- Download the dataset [here](https://1drv.ms/u/s!AvF6swdBjYzAg5lQ20tVJfjwa8D_Hw?e=TuCXba)


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
The table below shows the results of REIN compared to BPA and PSR on the ShapeNet and ModelNet10 datasets. BPA and PSR meshes were generated with Meshlab computed normals.

<div id="tab:shapenetmodelnetresults">
<table>
<tbody>
<tr class="even">
<td colspan='7'; style="text-align: center;">ShapeNet</td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td colspan='3'; style="text-align: center;">Chamfer Distance</td>
<td colspan='3'; style="text-align: center;">Point Normal Similarity</td>
</tr>
<tr class="even">
<td style="text-align: left;">Algorithm</td>
<td style="text-align: right;">Hull</td>
<td style="text-align: right;">Butterfly</td>
<td style="text-align: right;">Midpoint</td>
<td style="text-align: right;">Hull</td>
<td style="text-align: right;">Butterfly</td>
<td style="text-align: right;">Midpoint</td>
</tr>
<tr class="odd">
<td style="text-align: left;">BPA</td>
<td style="text-align: right;">0.0052</td>
<td style="text-align: right;">0.0075</td>
<td style="text-align: right;">0.0177</td>
<td style="text-align: right;">0.6224</td>
<td style="text-align: right;">0.8143</td>
<td style="text-align: right;">0.6995</td>
</tr>
<tr class="even">
<td style="text-align: left;">PSR</td>
<td style="text-align: right;">0.0871</td>
<td style="text-align: right;">0.0227</td>
<td style="text-align: right;">0.0200</td>
<td style="text-align: right;">0.5795</td>
<td style="text-align: right;">0.7762</td>
<td style="text-align: right;">0.7367</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Ours (REIN)</td>
<td style="text-align: right;"><strong>0.0003</strong></td>
<td style="text-align: right;"><strong>0.0033</strong></td>
<td style="text-align: right;"><strong>0.0028</strong></td>
<td style="text-align: right;"><strong>0.8317</strong></td>
<td style="text-align: right;"><strong>0.8181</strong></td>
<td style="text-align: right;"><strong>0.8313</strong></td>
</tr>
<tr></tr>
<tr class="even">
<td colspan='7'; style="text-align: center;">ModelNet10</td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td colspan='3'; style="text-align: center;">Chamfer Distance</td>
<td colspan='3'; style="text-align: center;">Point Normal Similarity</td>
</tr>
<tr class="even">
<td style="text-align: left;">Algorithm</td>
<td style="text-align: right;">Hull</td>
<td style="text-align: right;">Butterfly</td>
<td style="text-align: right;">Midpoint</td>
<td style="text-align: right;">Hull</td>
<td style="text-align: right;">Butterfly</td>
<td style="text-align: right;">Midpoint</td>
</tr>
<tr class="odd">
<td style="text-align: left;">BPA</td>
<td style="text-align: right;">0.0088</td>
<td style="text-align: right;">0.0106</td>
<td style="text-align: right;">0.0573</td>
<td style="text-align: right;">0.7210</td>
<td style="text-align: right;">0.8062</td>
<td style="text-align: right;">0.6032</td>
</tr>
<tr class="even">
<td style="text-align: left;">PSR</td>
<td style="text-align: right;">0.0292</td>
<td style="text-align: right;">0.0224</td>
<td style="text-align: right;">0.0292</td>
<td style="text-align: right;"><strong>0.8273</strong></td>
<td style="text-align: right;">0.7938</td>
<td style="text-align: right;">0.7583</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Ours (REIN)</td>
<td style="text-align: right;"><strong>0.0050</strong></td>
<td style="text-align: right;"><strong>0.0056</strong></td>
<td style="text-align: right;"><strong>0.0073</strong></td>
<td style="text-align: right;">0.8259</td>
<td style="text-align: right;"><strong>0.8285</strong></td>
<td style="text-align: right;"><strong>0.8288</strong></td>
</tr>
</tbody>
</table>
</div>

### Qualitative Results on ShapeNet Patched

The image below shows sample results of REIN by reconstructing by parts on the ShapeNet dataset. The generated meshes are compared with results from Ball Pivoting Algorithm (BPA) and Poisson Surface Reconstruction (PSR).

![sample results on shapenet patched](https://github.com/rangeldaroya/rein/blob/master/images/patching_blender_shapeNet_results.png)
