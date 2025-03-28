# Awesome Point Cloud

This is a curated list of cool papers in the point cloud realm.

- Abbreviation Link: Points to the official source (if available).
- Title Link: Directs you to the paper’s URL.
- Citation Rating: Based on my fun scale according to citation numbers – it doesn’t reflect true impact, just gives a shout-out to all the hardworking researchers!

  - 😀: 0 ~ 50
  - 🌗: 50 ~ 100
  - 🌕: 100 ~ 500
  - 🌕🌗: 500 ~ 1k
  - 🌕🌕: 1k ~ 5k
  - 🌕🌕🌗: 5k ~ 10k
  - 🌕🌕🌕: 10k ~

> I’m starting with the classic papers – stay tuned for the latest breakthroughs!

<!-- 🌕🌗🌑 -->
<!-- 🌑🌑🌑: 0 ~ 50 -->
<!-- 🌗🌑🌑: 50 ~ 100 -->
<!-- 🌕🌑🌑: 100 ~ 500 -->
<!-- 🌕🌗🌑: 500 ~ 1,000 -->
<!-- 🌕🌕🌑: 1,000 ~ 5,000 -->
<!-- 🌕🌕🌗: 5,000 ~ 10,000 -->
<!-- 🌕🌕🌕: greater than 10,000-->
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [3D Object Detection](#3d-object-detection)
- [Multi-Modality](#multi-modality)
- [Representation Learning](#representation-learning)
- [Tracking/Forecasting](#trackingforecasting)
- [Framework](#framework)


## Dataset

| Abbrev. | Title | Venue | Year | Cite | 
| :---: | :---: | :---: | :---: | :---: | 
| KITTI | [Are we ready for autonomous driving? The KITTI vision benchmark suite](https://ieeexplore.ieee.org/abstract/document/6248074) | CVPR | 2012 | 🌕🌕🌕 | 
| nuScene | [nuScenes: A Multimodal Dataset for Autonomous Driving](https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html) | CVPR | 2020 | 🌕🌕🌕 | 

## 3D Object Detection

| Abbrev. | Title | Venue | Year | Cite | 
| :---: | :---: | :---: | :---: | :---: | 
| [PointNet](https://github.com/charlesq34/pointnet) | [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) | CVPR | 2017 | 🌕🌕🌕 | 
| [PointNet++](https://github.com/charlesq34/pointnet2) | [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://proceedings.neurips.cc/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html) | NeurIPS | 2017 | 🌕🌕🌕 |
| [VoxelNet](https://github.com/steph1793/Voxelnet) | [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html) | CVPR | 2018 | 🌕🌕 |
| PIXOR | [PIXOR: Real-time 3D Object Detection from Point Clouds](https://arxiv.org/abs/1902.06326) | CVPR | 2018 | 🌕🌕🌑 |
| [SO-Net](https://github.com/lijx10/SO-Net) | [SO-Net: Self-Organizing Network for Point Cloud Analysis](https://arxiv.org/abs/1803.04249) | CVPR | 2018 | 🌕🌕 |
| PointFusion | [PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation](https://arxiv.org/abs/1711.10871) | CVPR | 2018 | 🌕🌗 |
| FaF | [Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net](https://arxiv.org/abs/2012.12395) | CVPR | 2018 | 🌕🌗 |
| [PointCNN](https://github.com/yangyanli/PointCNN) | [PointCNN: Convolution On X-Transformed Points](https://proceedings.neurips.cc/paper/2018/hash/f5f8590cd58a54e94377e6ae2eded4d9-Abstract.html) | NeurIPS | 2018 | 🌕🌕 |
| [CenterNet](https://github.com/dreamway/CenterNet-objects-as-points) | [Objects as Points](https://arxiv.org/abs/1904.07850) | arXiv | 2019 | 🌕🌕 |
| [PointPillars](https://github.com/zhulf0804/PointPillars) | [PointPillars: Fast Encoders for Object Detection From Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html) | CVPR | 2019 | 🌕🌕 |
| [PointRCNN](https://github.com/sshaoshuai/PointRCNN) | [PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html) | CVPR | 2019 | 🌕🌕 |
| [PointConv](https://github.com/DylanWusee/pointconv) | [PointConv: Deep Convolutional Networks on 3D Point Clouds](https://arxiv.org/abs/1811.07246) | CVPR | 2019 | 🌕🌕 |
| [KPConv](https://github.com/HuguesTHOMAS/KPConv) | [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/abs/1904.08889) | ICCV | 2019 | 🌕🌕 |
| [VoteNet](https://github.com/facebookresearch/votenet) | [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/abs/1904.09664) | ICCV | 2019 | 🌕🌕 |
| [PVCNN](https://github.com/mit-han-lab/pvcnn) | [Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/abs/1907.03739) | NeurIPS | 2019 | 🌕🌗 | 
| [SPVNAS](https://github.com/mit-han-lab/spvnas) | [Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution](https://arxiv.org/abs/2007.16100) | ECCV | 2020 | 🌕🌗 |
| [PV-RCNN](https://github.com/sshaoshuai/PV-RCNN) | [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.html) | CVPR | 2020 | 🌕🌕 |
| [3DSSD](https://github.com/dvlab-research/3DSSD) | [3DSSD: Point-Based 3D Single Stage Object Detector](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_3DSSD_Point-Based_3D_Single_Stage_Object_Detector_CVPR_2020_paper.html) | CVPR | 2020 | 🌕🌕 |
| [Point-GNN](https://github.com/WeijingShi/Point-GNN) | [Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.html) | CVPR | 2020 | 🌕🌕 |
| [Point Transformer](https://github.com/pointcept/pointcept) | [Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html?ref=;) | ICCV | 2021 | 🌕🌕 |
| [CenterPoint](https://github.com/tianweiy/CenterPoint) | [Center-Based 3D Object Detection and Tracking](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html) | CVPR | 2021 | 🌕🌕 |
| [Voxel R-CNN](https://github.com/djiajunustc/Voxel-R-CNN) | [Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection](https://arxiv.org/abs/2012.15712) | AAAI | 2021 | 🌕🌗 |
| DETR3D | [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://arxiv.org/abs/2110.06922) | CORL | 2021 | 🌕🌗 |
| [PCT](https://github.com/qinglew/PointCloudTransformer) | [PCT: Point cloud transformer](https://arxiv.org/abs/2012.09688) | arXiv | 2021 | 🌕🌕 |
| [BEVDet](https://github.com/HuangJunJie2017/BEVDet) | [BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View](https://arxiv.org/abs/2112.11790) | arXiv | 2021 | 🌕🌗 |


## Multi-Modality

| Abbrev. | Title | Venue | Year | Cite | 
| :---: | :---: | :---: | :---: | :---: | 
| Deep Continuous Fusion ([code1](https://github.com/Chanuk-Yang/Deep_Continuous_Fusion_for_Multi-Sensor_3D_Object_Detection)/[code2](https://github.com/JaHorL/Contfuse)) | [Deep Continuous Fusion for Multi-Sensor 3D Object Detection](https://arxiv.org/abs/2012.10992) | ECCV | 2018 | 🌕🌕 |
| MMF | [Multi-Task Multi-Sensor Fusion for 3D Object Detection](https://arxiv.org/abs/2012.12397) | CVPR | 2019 | 🌕🌗 |

## Representation Learning

To be updated

## Tracking/Forecasting

To be updated

## Framework

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): Open source object detection toolbox based on PyTorch for general 3D detection from OpenMMLab.
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet): Open source project for LiDAR-based 3D object detection from OpenMMLab


- [Paddle3D](https://github.com/PaddlePaddle/Paddle3D): Open-source end-to-end deep learning 3D perception suite for paddles
- [torch-points3d](https://github.com/torch-points3d/torch-points3d): A framework for point cloud analysis tasks.
- [Open3D](https://github.com/isl-org/Open3D): A Modern Library for 3D Data Processing
- [Open3D-ML](https://github.com/isl-org/Open3D-ML): An extension of Open3D for 3D machine learning tasks.
- [python-pcl](https://github.com/strawlab/python-pcl): Python binding to the pointcloud library.