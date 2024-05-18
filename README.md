
<p align="center">
     <a href="https://arxiv.org/abs/2405.05173">
<img width="765" alt="image" src="assets/title.png">
     </a>
   <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=kpMGaNIAAAAJ&hl=zh-CN"><strong>Huaiyuan Xu </strong></a>
    .
    <a href="https://scholar.google.com/citations?user=kqU2NJIAAAAJ&hl=zh-CN"><strong>Junliang Chen </strong></a>
    .
    <strong>Shiyu Meng</strong></a>
    .
    <a href="https://scholar.google.com/citations?user=MAG909MAAAAJ&hl=en"><strong>Yi Wang</strong></a>
    .
    <a href="https://scholar.google.com/citations?user=MYREIH0AAAAJ&hl=zh-CN"><strong>Lap-Pui Chau<sup>*</strong></a>
    
</p>
    
<p align="center">
    <a href='https://arxiv.org/abs/2405.05173'>
      <img src='https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
         </a>


 
  

# We research 3D Occupancy Perception for Autonomous Driving

This work focuses on 3D dense perception in autonomous driving, encompassing **LiDAR-Centric Occupancy Perception**, **Vision-Centric Occupancy Perception**, and **Multi-Modal Occupancy Perception**. Information fusion techniques for this field are discussed. We believe this will be the most comprehensive survey to date on 3D Occupancy Perception. Please stay tuned!😉😉😉

This is an active repository, you can watch for following the latest advances. If you find it useful, please kindly star this repo.

### ✨You are welcome to provide us your work with a topic related to 3D occupancy for autonomous driving (involving not only perception, but also applications)!!!

If you discover any missing work or have any suggestions, please feel free to submit a [pull request](https://github.com/HuaiyuanXu/3D-Occupancy-Perception/issues) or [contact us](#contact). We will promptly add the missing papers to this repository.

###  ✨Highlight!!!

[1]  A systematically survey for the latest research on 3D occupancy
perception in the field of autonomous driving.

[2]  The survey provides the taxonomy of 3D occupancy perception,
and elaborate on core methodological issues, including
network pipelines, multi-source information fusion, and
effective network training.

[3]  The survey presents evaluations for 3D occupancy perception, and
offers detailed performance comparisons. Furthermore,
current limitations and future research directions are discussed.

#

## Introduction

3D occupancy perception technology aims to observe and understand dense 3D environments for autonomous vehicles. Owing to its comprehensive perception capability, this technology is emerging as a trend in autonomous driving perception systems, and is attracting significant attention from both industry and academia. Similar to traditional bird's-eye view (BEV) perception, 3D occupancy perception has the nature of multi-source input and the necessity for information fusion. However, the difference is that it captures vertical structures that are ignored by 2D BEV. In this survey, we review the most recent works on 3D occupancy perception, and provide in-depth analyses of methodologies with various input modalities. Specifically, we summarize general network pipelines, highlight information fusion techniques, and discuss effective network training. We evaluate and analyze the occupancy perception performance of the state-of-the-art on the most popular datasets. Furthermore, challenges and future research directions are discussed. We hope this paper will inspire the community and encourage more research work on 3D occupancy perception.

</p>
<p align='center'>
<img src="assets/autonomous driving vehicle system.png" width="500px">
</p>

## Summary of Contents
- [Introduction](#introduction)
- [Summary of Contents](#summary-of-contents)
- [Methods: A Survey](#methods-a-survey)
  - [LiDAR-Centric Occupancy Perception](#liDAR-centric-occupancy-perception)
  - [Vision-Centric Occupancy Perception](#vision-centric-occupancy-perception)
  - [Multi-Modal Occupancy Perception](#multi-modal-occupancy-perception)
- [3D Occupancy Datasets](#3D-Occupancy-Datasets)
- [Occupancy-based Applications](#occupancy-based-applications)
- [Cite The Survey](#Cite-The-Survey)
- [Contact](#contact)


## Methods: A Survey

### LiDAR-Centric Occupancy Perception
|Year|Venue|Paper Title|Link|
|:-:|:-:|-|-|
|2024|AAAI|Semantic Complete Scene Forecasting from a 4D Dynamic Point Cloud Sequence|[Project Page](https://scsfnet.github.io)|
|2023|T-IV|Occupancy-MAE: Self-supervised Pre-training Large-scale LiDAR Point Clouds with Masked Occupancy Autoencoders|[Code](https://github.com/chaytonmin/Occupancy-MAE)|
|2023|arXiv|PointOcc: Cylindrical Tri-Perspective View for Point-based 3D Semantic Occupancy Prediction|[Code](https://github.com/wzzheng/PointOcc)|
|2023|arXiv|LiDAR-based 4D Occupancy Completion and Forecasting|[Project Page](https://ai4ce.github.io/Occ4cast/)|
|2021|T-PAMI|Semantic Scene Completion using Local Deep Implicit Functions on LiDAR Data|-|
|2021|AAAI|Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion|[Code](https://github.com/yanx27/JS3C-Net)|
|2020|CoRL|S3CNet: A Sparse Semantic Scene Completion Network for LiDAR Point Clouds|-|
|2020|3DV|LMSCNet: Lightweight Multiscale 3D Semantic Completion|[Code](https://github.com/astra-vision/LMSCNet)|



### Vision-Centric Occupancy Perception
|Year|Venue|Paper Title|Link|
|:-:|:-:|-|-|
|2024|IJCAI|Bridging Stereo Geometry and BEV Representation with Reliable Mutual Interaction for Semantic Scene Completion|[Code](https://github.com/Arlo0o/StereoScene)|
|2024|ICRA|RenderOcc: Vision-Centric 3D Occupancy Prediction with 2D Rendering Supervision|[Code](https://github.com/pmj110119/RenderOcc)|
|2024|ICRA|MonoOcc: Digging into Monocular Semantic Occupancy Prediction|[Code](https://github.com/ucaszyp/MonoOcc)|
|2024|ICRA|FastOcc: Accelerating 3D Occupancy Prediction by Fusing the 2D Bird’s-Eye View and Perspective View|-|
|2024|CVPR|Symphonize 3D Semantic Scene Completion with Contextual Instance Queries|[Code](https://github.com/hustvl/Symphonies)|
|2024|CVPR|SparseOcc: Rethinking Sparse Latent Representation for Vision-Based Semantic Occupancy Prediction|[Project Page](https://pintang1999.github.io/sparseocc.html)|
|2024|CVPR|SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction|[Project Page](https://huang-yh.github.io/SelfOcc/)|
|2024|CVPR|PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation|[Code](https://github.com/Robertwyq/PanoOcc)|
|2024|CVPR|Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation|[Code](https://github.com/songw-zju/HASSC)|
|2024|CVPR|COTR: Compact Occupancy TRansformer for Vision-based 3D Occupancy Prediction|[Code](https://github.com/NotACracker/COTR)|
|2024|CVPR|Collaborative Semantic Occupancy Prediction with Hybrid Feature Fusion in Connected Automated Vehicles|[Project Page](https://rruisong.github.io/publications/CoHFF/)|
|2024|CVPR|Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications|[Code](https://github.com/haomo-ai/Cam4DOcc)|
|2024|CVPR|Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation|[Project Page](https://keonhee-han.github.io/publications/kdbts/)|
|2024|CVPR|DriveWorld: 4D Pre-trained Scene Understanding viaWorld Models for Autonomous Driving|-|
|2024|AAAI|Regulating Intermediate 3D Features for Vision-Centric Autonomous Driving|[Code](https://github.com/cskkxjk/Vampire)|
|2024|AAAI|One at a Time: Progressive Multi-step Volumetric Probability Learning for Reliable 3D Scene Perception|-|
|2024|RA-L|Multi-Camera Unified Pre-Training via 3D Scene Reconstruction|[Code](https://github.com/chaytonmin/UniScene)|
|2024|arXiv|OccFlowNet: Towards Self-supervised Occupancy Estimation via Differentiable Rendering and Occupancy Flow|-|
|2024|arXiv|OccFiner: Offboard Occupancy Refinement with Hybrid Propagation|-|
|2024|arXiv|InverseMatrixVT3D: An Efficient Projection Matrix-Based Approach for 3D Occupancy Prediction|[Code](https://github.com/DanielMing123/InverseMatrixVT3D)|
|2024|arXiv|Unified Spatio-Temporal Tri-Perspective View Representation for 3D Semantic Occupancy Prediction|[Project Page](https://cepdnaclk.github.io/e17-4yp-S2TPVFormer/)|
|2024|arXiv|ViewFormer: Exploring Spatiotemporal Modeling for Multi-View 3D Occupancy Perception via View-Guided Transformers|-|
|2023|T-IV|3DOPFormer: 3D Occupancy Perception from Multi-Camera Images with Directional and Distance Enhancement|[Code](https://github.com/lvchuandong/ML3DOP)|
|2023|NeurIPS|POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images|[Project Page](https://vobecant.github.io/POP3D/)|
|2023|NeurIPS|Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving|[Project Page](https://tsinghua-mars-lab.github.io/Occ3D/)|
|2023|ICCV|SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving|[Project Page](https://weiyithu.github.io/SurroundOcc/)|
|2023|ICCV|Scene as Occupancy|[Code](https://github.com/OpenDriveLab/OccNet/tree/occnet)|
|2023|ICCV|OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction|[Code](https://github.com/zhangyp15/OccFormer)|
|2023|ICCV|NDC-Scene: Boost Monocular 3D Semantic Scene Completion in Normalized Device Coordinates Space|[Code](https://github.com/Jiawei-Yao0812/NDCScene)|
|2023|CVPR|VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion|[Code](https://github.com/NVlabs/VoxFormer)|
|2023|CVPR|Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction|[Project Page](https://wzzheng.net/TPVFormer/)|
|2023|arXiv|SSCBench: Monocular 3D Semantic Scene Completion Benchmark in Street Views|[Code](https://github.com/ai4ce/SSCBench)|
|2023|arXiv|SOccDPT: Semi-Supervised 3D Semantic Occupancy from Dense Prediction Transformers trained under memory constraints|-|
|2023|arXiv|OVO: Open-Vocabulary Occupancy|[Code](https://github.com/dzcgaara/OVO-Open-Vocabulary-Occupancy)|
|2023|arXiv|OctreeOcc: Efficient and Multi-Granularity Occupancy Prediction Using Octree Queries|[Code](https://github.com/4DVLab/OctreeOcc)|
|2023|arXiv|OccWorld: Learning a 3D OccupancyWorld Model for Autonomous Driving|[Project Page](https://wzzheng.net/OccWorld/)|
|2023|arXiv|OccNeRF: Self-Supervised Multi-Camera Occupancy Prediction with Neural Radiance Fields|[Project Page](https://linshan-bin.github.io/OccNeRF/)|
|2023|arXiv|OccDepth: A Depth-Aware Method for 3D Semantic Scene Completion|[Code](https://github.com/megvii-research/OccDepth)|
|2023|arXiv|Fully Sparse 3D Occupancy Prediction|[Code](https://github.com/MCG-NJU/SparseOcc)|
|2023|arXiv|FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin|[Code](https://github.com/Yzichen/FlashOCC)|
|2023|arXiv|FB-OCC: 3D Occupancy Prediction based on Forward-Backward View Transformation|[Code](https://github.com/NVlabs/FB-BEV)|
|2023|arXiv|DepthSSC: Depth-Spatial Alignment and Dynamic Voxel Resolution for Monocular 3D Semantic Scene Completion|-|
|2023|arXiv|Camera-based 3D Semantic Scene Completion with Sparse Guidance Network|[Code](https://github.com/Jieqianyu/SGN)|
|2023|arXiv|A Simple Framework for 3D Occupancy Estimation in Autonomous Driving|[Code](https://github.com/GANWANSHUI/SimpleOccupancy)|
|2023|arXiv|UniWorld: Autonomous Driving Pre-training via World Models|[Code](https://github.com/chaytonmin/UniWorld)|
|2022|CVPR|MonoScene: Monocular 3D Semantic Scene Completion|[Project Page](https://astra-vision.github.io/MonoScene/)|



### Multi-Modal Occupancy Perception
|Year|Venue|Paper Title|Code|
|:-:|:-:|-|-|
|2024|arXiv|Real-time 3D semantic occupancy prediction for autonomous vehicles using memory-efficient sparse convolution|-|
|2024|arXiv|OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving|[Project Page](https://occgen-ad.github.io/)|
|2024|arXiv|OccFusion: A Straightforward and Effective Multi-Sensor Fusion Framework for 3D Occupancy Prediction|-|
|2024|arXiv|Co-Occ: Coupling Explicit Feature Fusion with Volume Rendering Regularization for Multi-Modal 3D Semantic Occupancy Prediction|[Project Page](https://rorisis.github.io/Co-Occ_project-page/)|
|2024|arXiv|Unleashing HyDRa: Hybrid Fusion, Depth Consistency and Radar for Unified 3D Perception|-|
|2023|ICCV|OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception|[Code](https://github.com/JeffWang987/OpenOccupancy)|

## 3D Occupancy Datasets

|Dataset|Year|Venue|Modality|# of Classes|Flow|Link|
|:-:|:-:|:-:|-|-|-|-|
|OpenScene|2024|CVPR 2024 Challenge|Camera|-| ✔️|[Intro.](https://github.com/OpenDriveLab/OpenScene)|
|Cam4DOcc|2024|CVPR|Camera+LiDAR|2| ✔️|[Intro.](https://github.com/haomo-ai/Cam4DOcc)|
|Occ3D|2024|NeurIPS|Camera|14 (Occ3D-Waymo), 16 (Occ3D-nuScenes)| ❌|[Intro.](https://github.com/Tsinghua-MARS-Lab/Occ3D)|
|OpenOcc|2023|ICCV|Camera|16| ❌|[Intro.](https://github.com/OpenDriveLab/OccNet)|
|OpenOccupancy|2023|ICCV|Camera+LiDAR|16| ❌|[Intro.](https://github.com/JeffWang987/OpenOccupancy)|
|SurroundOcc|2023|ICCV|Camera|16| ❌|[Intro.](https://github.com/weiyithu/SurroundOcc)|
|OCFBench|2023|arXiv|LiDAR|-(OCFBench-Lyft), 17(OCFBench-Argoverse), 25(OCFBench-ApolloScape), 16(OCFBench-nuScenes)| ❌|[Intro.](https://huggingface.co/datasets/ai4ce/OCFBench)|
|SSCBench|2023|arXiv|Camera|19(SSCBench-KITTI-360), 16(SSCBench-nuScenes), 14(SSCBench-Waymo)| ❌|[Intro.](https://github.com/ai4ce/SSCBench)|
|SemanticKITT|2019|ICCV|Camera+LiDAR|19(Semantic Scene Completion task)| ❌|[Intro.](http://www.semantic-kitti.org/tasks.html#ssc)|


## Occupancy-based Applications
### Segmentation
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|BEV Segmentation|2024|arXiv|OccFeat: Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation Networks|-|
|3D Panoptic Segmentation|2024|CVPR|PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation|[Code](https://github.com/Robertwyq/PanoOcc)|

### Flow
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|3D Flow Prediction|2024|CVPR|Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications|[Code](https://github.com/haomo-ai/Cam4DOcc)|

### Detection
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|3D Object Detection|2024|CVPR|Learning Occupancy for Monocular 3D Object Detection|[Code](https://github.com/SPengLiang/OccupancyM3D)|
|3D Object Detection|2023|arXiv|SOGDet: Semantic-Occupancy Guided Multi-view 3D Object Detection|[Code](https://github.com/zhouqiu/SOGDet)|


### Generation
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|Scene Generation|2024|CVPR|SemCity: Semantic Scene Generation with Triplane Diffusion|[Code](https://github.com/zoomin-lee/SemCity)|

### Unified Framework
|Specific Tasks|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|Occupancy Prediction, 3D Object Detection, Online Mapping, Multi-object Tracking, Motion Prediction, Motion Planning|2024|CVPR|DriveWorld: 4D Pre-trained Scene Understanding viaWorld Models for Autonomous Driving|-|
|Occupancy Prediction, 3D Object Detection, BEV segmentation, Motion Planning|2023|ICCV|Scene as Occupancy|[Code](https://github.com/OpenDriveLab/OccNet/tree/occnet)|



## Cite The Survey
If you find our survey and repository useful for your research project, please consider citing our paper:

```bibtex
@misc{xu2024survey,
      title={A Survey on Occupancy Perception for Autonomous Driving: The Information Fusion Perspective}, 
      author={Huaiyuan Xu and Junliang Chen and Shiyu Meng and Yi Wang and Lap-Pui Chau},
      year={2024},
      eprint={2405.05173},
      archivePrefix={arXiv}
}
```
## Contact

```
huaiyuan.xu@polyu.edu.hk
```
```
lap-pui.chau@polyu.edu.hk
```

