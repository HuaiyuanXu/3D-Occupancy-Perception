
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

### ✨You are welcome to provide us your work with a topic related to 3D occupancy for autonomous driving (involving not only perception, but also applications)!

If you discover any missing work or have any suggestions, please feel free to submit a [pull request](https://github.com/HuaiyuanXu/3D-Occupancy-Perception/issues) or [contact us](#contact). We will promptly add the missing papers to this repository.

###  ✨Highlight

[1]  A systematically survey for the latest research on 3D occupancy
perception in the field of autonomous driving.

[2]  The survey provides the taxonomy of 3D occupancy perception,
and elaborate on core methodological issues, including
network pipelines, multi-source information fusion, and
effective network training.

[3]  The survey presents evaluations for 3D occupancy perception, and
offers detailed performance comparisons. Furthermore,
current limitations and future research directions are discussed.

###  🔥 News
- [2024-09-03]  This survey got accepted by Information Fusion (Impact factor: 14.7).
- [2024-07-21]  More representative works and benchmarking comparisons have been incorporated, bringing the total to 192 literature references.
- [2024-05-18]  More figures have been added to the survey. We reorganize the occupancy-based applications.
- [2024-05-08] The first version of the survey is available on [arXiv](https://arxiv.org/abs/2405.05173). We curate this repository.




## Introduction

3D occupancy perception technology aims to observe and understand dense 3D environments for autonomous vehicles. Owing to its comprehensive perception capability, this technology is emerging as a trend in autonomous driving perception systems, and is attracting significant attention from both industry and academia. Similar to traditional bird's-eye view (BEV) perception, 3D occupancy perception has the nature of multi-source input and the necessity for information fusion. However, the difference is that it captures vertical structures that are ignored by 2D BEV. In this survey, we review the most recent works on 3D occupancy perception, and provide in-depth analyses of methodologies with various input modalities. Specifically, we summarize general network pipelines, highlight information fusion techniques, and discuss effective network training. We evaluate and analyze the occupancy perception performance of the state-of-the-art on the most popular datasets. Furthermore, challenges and future research directions are discussed. We hope this paper will inspire the community and encourage more research work on 3D occupancy perception.

</p>
<p align='center'>
<img src="assets/autonomous driving vehicle system.png" width="500px">
<p align='center'>
<img src="assets/a brief history.png" width="1000px">
</p>

## Summary of Contents
- [Introduction](#introduction)
- [Summary of Contents](#summary-of-contents)
- [Methods: A Survey](#methods-a-survey)
  - [LiDAR-Centric Occupancy Perception](#liDAR-centric-occupancy-perception)
  - [Vision-Centric Occupancy Perception](#vision-centric-occupancy-perception)
  - [Radar-Centric Occupancy Perception](#radar-centric-occupancy-perception)
  - [Multi-Modal Occupancy Perception](#multi-modal-occupancy-perception)
- [3D Occupancy Datasets](#3D-Occupancy-Datasets)
- [Occupancy-based Applications](#occupancy-based-applications)
- [Cite The Survey](#Cite-The-Survey)
- [Contact](#contact)


## Methods: A Survey

### LiDAR-Centric Occupancy Perception
|Year|Venue|Paper Title|Link|
|:-:|:-:|-|-|
|2024|NeurIPS|[TALoS: Enhancing Semantic Scene Completion via Test-time Adaptation on the Line of Sight](https://arxiv.org/abs/2410.15674) |[Code](https://github.com/blue-531/TALoS)|
|2024|CVPR|[PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness](https://arxiv.org/abs/2312.02158) (Best paper award candidate)|[Project Page](https://astra-vision.github.io/PaSCo/)|
|2024|IROS|[LiDAR-based 4D Occupancy Completion and Forecasting](https://arxiv.org/abs/2310.11239)|[Project Page](https://ai4ce.github.io/Occ4cast/)|
|2024|arXiv|[Towards 3D Semantic Scene Completion for Autonomous Driving: A Meta-Learning Framework Empowered by Deformable Large-Kernel Attention and Mamba Model](https://arxiv.org/abs/2411.03672)|-|
|2024|arXiv|[OccRWKV: Rethinking Efficient 3D Semantic Occupancy Prediction with Linear Complexity](https://www.arxiv.org/abs/2409.19987)|[Project Page](https://jmwang0117.github.io/OccRWKV/)|
|2024|arXiv|[DiffSSC: Semantic LiDAR Scan Completion using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2409.18092)|-|
|2024|arXiv|[MergeOcc: Bridge the Domain Gap between Different LiDARs for Robust Occupancy Prediction](https://arxiv.org/abs/2403.08512)|-|
|2023|T-IV|[Occupancy-MAE: Self-supervised Pre-training Large-scale LiDAR Point Clouds with Masked Occupancy Autoencoders](https://arxiv.org/abs/2206.09900)|[Code](https://github.com/chaytonmin/Occupancy-MAE)|
|2023|arXiv|[PointOcc: Cylindrical Tri-Perspective View for Point-based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2308.16896)|[Code](https://github.com/wzzheng/PointOcc)|
|2021|T-PAMI|[Semantic Scene Completion using Local Deep Implicit Functions on LiDAR Data](https://arxiv.org/abs/2011.09141)|-|
|2021|AAAI|[Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion](https://arxiv.org/abs/2012.03762)|[Code](https://github.com/yanx27/JS3C-Net)|
|2020|CoRL|[S3CNet: A Sparse Semantic Scene Completion Network for LiDAR Point Clouds](https://arxiv.org/abs/2012.09242)|-|
|2020|3DV|[LMSCNet: Lightweight Multiscale 3D Semantic Completion](https://arxiv.org/abs/2008.10559)|[Code](https://github.com/astra-vision/LMSCNet)|



### Vision-Centric Occupancy Perception
|Year|Venue|Paper Title|Link|
|:-:|:-:|-|-|
|2025|CVPR|[3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](https://arxiv.org/abs/2503.15185)|[Project Page](https://kuai-lab.github.io/cvpr2025protoocc/)| 
|2025|T-RO|[Particle-based Instance-aware Semantic Occupancy Mapping in Dynamic Environments](https://arxiv.org/abs/2409.11975)|-| 
|2025|AAAI|[VLScene: Vision-Language Guidance Distillation for Camera-Based 3D Semantic Scene Completion](https://arxiv.org/abs/2503.06219) (Oral)|[Code](https://github.com/willemeng/VLScene)| 
|2025|AAAI|[Skip Mamba Diffusion for Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2501.07260)|[Code](https://github.com/xrkong/skimba)| 
|2025|AAAI|[ViPOcc: Leveraging Visual Priors from Vision Foundation Models for Single-View 3D Occupancy Prediction](https://arxiv.org/abs/2412.11210)|[Project Page](https://mias.group/ViPOcc/)|
|2025|AAAI|[ProtoOcc: Accurate, Efficient 3D Occupancy Prediction Using Dual Branch Encoder-Prototype Query Decoder](https://arxiv.org/abs/2412.08774)|[Code](https://github.com/SPA-junghokim/ProtoOcc)|
|2025|AAAI|[LOMA: Language-assisted Semantic Occupancy Network via Triplane Mamba](https://www.arxiv.org/abs/2412.08388)|-|
|2025|AAAI|[Semi-supervised 3D Semantic Scene Completion with 2D Vision Foundation Model Guidance](https://arxiv.org/abs/2408.11559v3)|-|
|2025|ICRA|[OCCUQ: Exploring Efficient Uncertainty Quantification for 3D Occupancy Prediction](https://arxiv.org/abs/2503.10605)|[Code](https://github.com/ika-rwth-aachen/OCCUQ)|
|2024|AAAIW|[A Spatiotemporal Approach to Tri-Perspective Representation for 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2401.13785)|[Project Page](https://cepdnaclk.github.io/e17-4yp-S2TPVFormer/)|
|2025|arXiv|[OccLinker: Deflickering Occupancy Networks through Lightweight Spatio-Temporal Correlation](https://arxiv.org/abs/2502.15438)|-|
|2025|arXiv|[Learning A Zero-shot Occupancy Network from Vision Foundation Models via Self-supervised Adaptation](https://arxiv.org/abs/2503.07125)|-|
|2025|arXiv|[Vision-based 3D Semantic Scene Completion via Capture Dynamic Representations](https://arxiv.org/abs/2503.06222)|-|
|2025|arXiv|[TT-GaussOcc: Test-Time Compute for Self-Supervised Occupancy Prediction via Spatio-Temporal Gaussian Splatting](https://arxiv.org/abs/2503.08485)|-|
|2025|arXiv|[AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](https://arxiv.org/abs/2502.04981)|-|
|2025|arXiv|[GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](https://arxiv.org/abs/2502.17288)|-|
|2025|arXiv|[Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](https://arxiv.org/abs/2502.14520)|-|
|2025|arXiv|[GaussRender: Learning 3D Occupancy with Gaussian Rendering](https://arxiv.org/abs/2502.05040)|[Code](https://github.com/valeoai/GaussRender)|
|2025|arXiv|[Event-aided Semantic Scene Completion](https://arxiv.org/abs/2502.02334)|[Code](https://github.com/Pandapan01/EvSSC)|
|2024|NeurIPS|[OctreeOcc: Efficient and Multi-Granularity Occupancy Prediction Using Octree Queries](https://arxiv.org/abs/2312.03774)|[Code](https://github.com/4DVLab/OctreeOcc)|
|2024|NeurIPS|[Context and Geometry Aware Voxel Transformer for Semantic Scene Completion](https://arxiv.org/abs/2405.13675) (Spotlight paper)|[Code](https://github.com/pkqbajng/CGFormer)|
|2024|NeurIPS|[OPUS: Occupancy Prediction Using a Sparse Set](https://www.arxiv.org/abs/2409.09350)|[Code](https://github.com/jbwang1997/OPUS)|
|2024|ECCV|[ViewFormer: Exploring Spatiotemporal Modeling for Multi-View 3D Occupancy Perception via View-Guided Transformers](https://arxiv.org/abs/2405.04299)|[Code](https://github.com/ViewFormerOcc/ViewFormer-Occ)|
|2024|ECCV|[CVT-Occ: Cost Volume Temporal Fusion for 3D Occupancy Prediction](https://arxiv.org/abs/2409.13430)|[Code](https://github.com/Tsinghua-MARS-Lab/CVT-Occ)|
|2024|ECCV|[VEON: Vocabulary-Enhanced Occupancy Prediction](https://arxiv.org/abs/2407.12294)|[Code](https://github.com/MCG-NJU/SparseOcc)|
|2024|ECCV|[Fully Sparse 3D Occupancy Prediction](https://arxiv.org/abs/2312.17118)|[Code](https://github.com/MCG-NJU/SparseOcc)|
|2024|ECCV|[GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2405.17429)|[Project Page](https://wzzheng.net/GaussianFormer/)|
|2024|ECCV|[Occupancy as Set of Points](https://arxiv.org/abs/2407.04049)|[Code](https://github.com/hustvl/osp)|
|2024|ECCV|[Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion](https://arxiv.org/abs/2407.02077)|[Code](https://github.com/Arlo0o/HTCL)|
|2024|CVPR|[LowRankOcc: Tensor Decomposition and Low-Rank Recovery for Vision-based 3D Semantic Occupancy Prediction](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_LowRankOcc_Tensor_Decomposition_and_Low-Rank_Recovery_for_Vision-based_3D_Semantic_CVPR_2024_paper.pdf)|-|
|2024|CVPR|[Bi-SSC: Geometric-Semantic Bidirectional Fusion for Camera-based 3D Semantic Scene Completion](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Bi-SSC_Geometric-Semantic_Bidirectional_Fusion_for_Camera-based_3D_Semantic_Scene_Completion_CVPR_2024_paper.pdf)|-|
|2024|CVPR|[Symphonize 3D Semantic Scene Completion with Contextual Instance Queries](https://arxiv.org/abs/2306.15670)|[Code](https://github.com/hustvl/Symphonies)|
|2024|CVPR|[SparseOcc: Rethinking Sparse Latent Representation for Vision-Based Semantic Occupancy Prediction](https://arxiv.org/abs/2404.09502)|[Project Page](https://pintang1999.github.io/sparseocc.html)|
|2024|CVPR|[SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction](https://arxiv.org/abs/2311.12754)|[Project Page](https://huang-yh.github.io/SelfOcc/)|
|2024|CVPR|[PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation](https://arxiv.org/abs/2306.10013)|[Code](https://github.com/Robertwyq/PanoOcc)|
|2024|CVPR|[Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation](https://arxiv.org/abs/2404.11958)|[Code](https://github.com/songw-zju/HASSC)|
|2024|CVPR|[COTR: Compact Occupancy TRansformer for Vision-based 3D Occupancy Prediction](https://arxiv.org/abs/2312.01919)|[Code](https://github.com/NotACracker/COTR)|
|2024|CVPR|[Collaborative Semantic Occupancy Prediction with Hybrid Feature Fusion in Connected Automated Vehicles](https://arxiv.org/abs/2402.07635)|[Project Page](https://rruisong.github.io/publications/CoHFF/)|
|2024|CVPR|[Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications](https://arxiv.org/abs/2311.17663)|[Code](https://github.com/haomo-ai/Cam4DOcc)|
|2024|CVPR|[Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation](https://arxiv.org/abs/2404.07933)|[Project Page](https://keonhee-han.github.io/publications/kdbts/)|
|2024|CVPR|[DriveWorld: 4D Pre-trained Scene Understanding viaWorld Models for Autonomous Driving](https://arxiv.org/abs/2405.04390)|-|
|2024|T-IP|[Camera-based 3D Semantic Scene Completion with Sparse Guidance Network](https://arxiv.org/abs/2312.05752)|[Code](https://github.com/Jieqianyu/SGN)|
|2024|CoRL|[Let Occ Flow: Self-Supervised 3D Occupancy Flow Prediction](https://arxiv.org/abs/2407.07587)|[Project Page](https://eliliu2233.github.io/letoccflow/)|
|2024|IJCAI|[Label-efficient Semantic Scene Completion with Scribble Annotations](https://arxiv.org/abs/2405.15170)|[Code](https://github.com/songw-zju/Scribble2Scene)|
|2024|IJCAI|[Bridging Stereo Geometry and BEV Representation with Reliable Mutual Interaction for Semantic Scene Completion](https://arxiv.org/abs/2303.13959)|[Code](https://github.com/Arlo0o/StereoScene)|
|2024|ICRA|[The RoboDrive Challenge: Drive Anytime Anywhere in Any Condition](https://arxiv.org/abs/2405.08816)|[Project Page](https://robodrive-24.github.io/)|
|2024|ICRA|[RenderOcc: Vision-Centric 3D Occupancy Prediction with 2D Rendering Supervision](https://arxiv.org/abs/2309.09502)|[Code](https://github.com/pmj110119/RenderOcc)|
|2024|ICRA|[MonoOcc: Digging into Monocular Semantic Occupancy Prediction](https://arxiv.org/abs/2403.08766)|[Code](https://github.com/ucaszyp/MonoOcc)|
|2024|ICRA|[FastOcc: Accelerating 3D Occupancy Prediction by Fusing the 2D Bird’s-Eye View and Perspective View](https://arxiv.org/abs/2403.02710)|-|
|2024|AAAI|[Regulating Intermediate 3D Features for Vision-Centric Autonomous Driving](https://arxiv.org/abs/2312.11837)|[Code](https://github.com/cskkxjk/Vampire)|
|2024|AAAI|[One at a Time: Progressive Multi-step Volumetric Probability Learning for Reliable 3D Scene Perception](https://arxiv.org/abs/2306.12681)|-|
|2024|RA-L|[HybridOcc: NeRF Enhanced Transformer-based Multi-Camera 3D Occupancy Prediction](https://www.arxiv.org/abs/2408.09104)|-|
|2024|RA-L|[UniScene: Multi-Camera Unified Pre-Training via 3D Scene Reconstruction](https://arxiv.org/abs/2305.18829)|[Code](https://github.com/chaytonmin/UniScene)|
|2024|AAIML|[SOccDPT: Semi-Supervised 3D Semantic Occupancy from Dense Prediction Transformers trained under memory constraints](https://arxiv.org/abs/2311.11371)|[Project Page](https://adityang.github.io/SOccDPT)|
|2024|3DV|[PanoSSC: Exploring Monocular Panoptic 3D Scene Reconstruction for Autonomous Driving](https://arxiv.org/abs/2406.07037)|-|
|2024|IROS|[SSCBench: Monocular 3D Semantic Scene Completion Benchmark in Street Views](https://arxiv.org/abs/2306.09001)|[Code](https://github.com/ai4ce/SSCBench)|
|2024|arXiv|[GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting](https://arxiv.org/abs/2412.14579)|- |
|2024|arXiv|[GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding](https://arxiv.org/abs/2412.13193)|[Code](https://github.com/hustvl/GaussTR) |
|2024|arXiv|[GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](https://arxiv.org/abs/2412.10373)|[Code](https://github.com/zuosc19/GaussianWorld) |
|2024|arXiv|[GaussianAD: Gaussian-Centric End-to-End Autonomous Driving](https://arxiv.org/abs/2412.10371)|[Project Page](https://wzzheng.net/GaussianAD) |
|2024|arXiv|[Hierarchical Context Alignment with Disentangled Geometric and Temporal Modeling for Semantic Occupancy Prediction](https://arxiv.org/abs/2412.08243)|-|
|2024|arXiv|[Fast Occupancy Network](https://arxiv.org/abs/2412.07163)|-|
|2024|arXiv|[Lightweight Spatial Embedding for Vision-based 3D Occupancy Prediction](https://arxiv.org/abs/2412.05976)|-|
|2024|arXiv|[GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](https://arxiv.org/abs/2412.04384)|[Code](https://github.com/huang-yh/GaussianFormer)|
|2024|arXiv|[Language Driven Occupancy Prediction](https://arxiv.org/abs/2411.16072)|[Code](https://github.com/pkqbajng/LOcc)|
|2024|arXiv|[GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving](https://arxiv.org/abs/2411.12452)|[Code](https://github.com/Public-BOTs/GaussianPretrain)|
|2024|arXiv|[ET-Former: Efficient Triplane Deformable Attention for 3D Semantic Scene Completion From Monocular Camera](https://arxiv.org/abs/2410.11019)|-|
|2024|arXiv|[ReliOcc: Towards Reliable Semantic Occupancy Prediction via Uncertainty Learning](https://www.arxiv.org/abs/2409.18026)|-|
|2024|arXiv|[Deep Height Decoupling for Precise Vision-based 3D Occupancy Prediction](https://arxiv.org/abs/2409.07972)|[Code](https://github.com/yanzq95/DHD)|
|2024|arXiv|[AdaOcc: Adaptive-Resolution Occupancy Prediction](https://arxiv.org/abs/2408.13454)|-|
|2024|arXiv|[GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting](https://arxiv.org/abs/2408.11447)|[Project Page](https://ganwanshui.github.io/GaussianOcc/)|
|2024|arXiv|[MambaOcc: Visual State Space Model for BEV-based Occupancy Prediction with Local Adaptive Reordering](https://arxiv.org/abs/2408.11464)|[Code](https://github.com/Hub-Tian/MambaOcc)|
|2024|arXiv|[VPOcc: Exploiting Vanishing Point for Monocular 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2408.03551)|-|
|2024|arXiv|[UniVision: A Unified Framework for Vision-Centric 3D Perception](https://arxiv.org/abs/2401.06994)|[Code](https://github.com/Cc-Hy/UniVision)|
|2024|arXiv|[LangOcc: Self-Supervised Open Vocabulary Occupancy Estimation via Volume Rendering](https://arxiv.org/abs/2407.17310)|-|
|2024|arXiv|[Real-Time 3D Occupancy Prediction via Geometric-Semantic Disentanglement](https://arxiv.org/abs/2407.13155)|-|
|2024|arXiv|[α-SSC: Uncertainty-Aware Camera-based 3D Semantic Scene Completion](https://arxiv.org/abs/2406.11021)|-|
|2024|arXiv|[Panoptic-FlashOcc: An Efficient Baseline to Marry Semantic Occupancy with Panoptic via Instance Center](https://arxiv.org/abs/2406.10527)|[Code](https://github.com/Yzichen/FlashOCC)|
|2024|arXiv|[BDC-Occ: Binarized Deep Convolution Unit For Binarized Occupancy Network](https://arxiv.org/abs/2405.17037)|[Code](https://github.com/zzk785089755/BDC)|
|2024|arXiv|[GEOcc: Geometrically Enhanced 3D Occupancy Network with Implicit-Explicit Depth Fusion and Contextual Self-Supervision](https://arxiv.org/abs/2405.10591)|-|
|2024|arXiv|[OccFlowNet: Towards Self-supervised Occupancy Estimation via Differentiable Rendering and Occupancy Flow](https://arxiv.org/abs/2402.12792)|-|
|2024|arXiv|[OccFiner: Offboard Occupancy Refinement with Hybrid Propagation](https://arxiv.org/abs/2403.08504)|-|
|2024|arXiv|[InverseMatrixVT3D: An Efficient Projection Matrix-Based Approach for 3D Occupancy Prediction](https://arxiv.org/abs/2401.12422)|[Code](https://github.com/DanielMing123/InverseMatrixVT3D)|
|2023|CVPR|[VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion](https://arxiv.org/abs/2302.12251)|[Code](https://github.com/NVlabs/VoxFormer)|
|2023|CVPR|[Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2302.07817)|[Project Page](https://wzzheng.net/TPVFormer/)|
|2023|NeurIPS|[POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images](https://arxiv.org/abs/2401.09413)|[Project Page](https://vobecant.github.io/POP3D/)|
|2023|NeurIPS|[Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/abs/2304.14365)|[Project Page](https://tsinghua-mars-lab.github.io/Occ3D/)|
|2023|ICCV|[SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving](https://arxiv.org/abs/2303.09551)|[Project Page](https://weiyithu.github.io/SurroundOcc/)|
|2023|ICCV|[Scene as Occupancy](https://arxiv.org/abs/2306.02851)|[Code](https://github.com/OpenDriveLab/OccNet/tree/occnet)|
|2023|ICCV|[OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2304.05316)|[Code](https://github.com/zhangyp15/OccFormer)|
|2023|ICCV|[NDC-Scene: Boost Monocular 3D Semantic Scene Completion in Normalized Device Coordinates Space](https://arxiv.org/abs/2309.14616)|[Code](https://github.com/Jiawei-Yao0812/NDCScene)|
|2023|T-IV|[3DOPFormer: 3D Occupancy Perception from Multi-Camera Images with Directional and Distance Enhancement](https://ieeexplore.ieee.org/document/10363646)|[Code](https://github.com/lvchuandong/ML3DOP)|
|2023|arXiv|[OccupancyDETR: Using DETR for Mixed Dense-sparse 3D Occupancy Prediction](https://arxiv.org/abs/2309.08504)|-|
|2023|arXiv|[OVO: Open-Vocabulary Occupancy](https://arxiv.org/abs/2305.16133)|[Code](https://github.com/dzcgaara/OVO-Open-Vocabulary-Occupancy)|
|2023|arXiv|[OccNeRF: Advancing 3D Occupancy Prediction in LiDAR-Free Environments](https://arxiv.org/abs/2312.09243v2)|[Project Page](https://linshan-bin.github.io/OccNeRF/)|
|2023|arXiv|[OccDepth: A Depth-Aware Method for 3D Semantic Scene Completion](https://arxiv.org/abs/2302.13540)|[Code](https://github.com/megvii-research/OccDepth)|
|2023|arXiv|[FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin](https://arxiv.org/abs/2311.12058)|[Code](https://github.com/Yzichen/FlashOCC)|
|2023|arXiv|[FB-OCC: 3D Occupancy Prediction based on Forward-Backward View Transformation](https://arxiv.org/abs/2307.01492)|[Code](https://github.com/NVlabs/FB-BEV)|
|2023|arXiv|[DepthSSC: Depth-Spatial Alignment and Dynamic Voxel Resolution for Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2311.17084)|-|
|2023|arXiv|[A Simple Framework for 3D Occupancy Estimation in Autonomous Driving](https://arxiv.org/abs/2303.10076)|[Code](https://github.com/GANWANSHUI/SimpleOccupancy)|
|2023|arXiv|[UniWorld: Autonomous Driving Pre-training via World Models](https://arxiv.org/abs/2308.07234)|[Code](https://github.com/chaytonmin/UniWorld)|
|2022|CVPR|[MonoScene: Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2112.00726)|[Project Page](https://astra-vision.github.io/MonoScene/)|


### Radar-Centric Occupancy Perception
|Year|Venue|Paper Title|Link|
|:-:|:-:|-|-|
|2024|NeurIPS|[RadarOcc: Robust 3D Occupancy Prediction with 4D Imaging Radar](https://arxiv.org/abs/2405.14014)|-|


### Multi-Modal Occupancy Perception
|Year|Venue|Paper Title|Code|
|:-:|:-:|-|-|
|2025|arXiv|[OccGS: Zero-shot 3D Occupancy Reconstruction with Semantic and Geometric-Aware Gaussian Splatting](https://arxiv.org/abs/2502.04981)|-|
|2025|arXiv|[MetaOcc: Surround-View 4D Radar and Camera Fusion Framework for 3D Occupancy Prediction with Dual Training Strategies](https://arxiv.org/abs/2501.15384)|[Code](https://github.com/LucasYang567/MetaOcc)|
|2025|arXiv|[DORACAMOM: Joint 3D Detection and Occupancy Prediction with Multi-view 4D Radars and Cameras for Omnidirectional Perception](https://arxiv.org/abs/2501.15394)|-|
|2024|ECCV|[OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](https://arxiv.org/abs/2404.15014)|[Project Page](https://occgen-ad.github.io/)|
|2024|RA-L|[Co-Occ: Coupling Explicit Feature Fusion with Volume Rendering Regularization for Multi-Modal 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2404.04561)|[Project Page](https://rorisis.github.io/Co-Occ_project-page/)|
|2024|arXiv|[MR-Occ: Efficient Camera-LiDAR 3D Semantic Occupancy Prediction Using Hierarchical Multi-Resolution Voxel Representation](https://arxiv.org/abs/2412.20480)|-|
|2024|arXiv|[PVP: Polar Representation Boost for 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2412.07616)|-|
|2024|arXiv|[Robust 3D Semantic Occupancy Prediction with Calibration-free Spatial Transformation](https://arxiv.org/abs/2411.12177)|[Code](https://github.com/ICEORY/REO)|
|2024|arXiv|[OccLoff: Learning Optimized Feature Fusion for 3D Occupancy Prediction](https://arxiv.org/abs/2411.03696)|-|
|2024|arXiv|[DAOcc: 3D Object Detection Assisted Multi-Sensor Fusion for 3D Occupancy Prediction](https://arxiv.org/abs/2409.19972)|[Code](https://github.com/AlphaPlusTT/DAOcc)|
|2024|arXiv|[OccMamba: Semantic Occupancy Prediction with State Space Models](https://arxiv.org/abs/2408.09859)|-|
|2024|arXiv|[LiCROcc: Teach Radar for Accurate Semantic Occupancy Prediction using LiDAR and Camera](https://www.arxiv.org/abs/2407.16197)|[Project Page](https://hr-zju.github.io/LiCROcc/)|
|2024|arXiv|[OccFusion: Depth Estimation Free Multi-sensor Fusion for 3D Occupancy Prediction](https://arxiv.org/abs/2403.05329)|-|
|2024|arXiv|[EFFOcc: A Minimal Baseline for EFficient Fusion-based 3D Occupancy Network](https://arxiv.org/abs/2406.07042)|[Code](https://github.com/synsin0/EFFOcc)|
|2024|arXiv|[Real-time 3D semantic occupancy prediction for autonomous vehicles using memory-efficient sparse convolution](https://arxiv.org/abs/2403.08748)|-|
|2024|arXiv|[OccFusion: A Straightforward and Effective Multi-Sensor Fusion Framework for 3D Occupancy Prediction](https://arxiv.org/abs/2403.01644)|-|
|2024|arXiv|[Unleashing HyDRa: Hybrid Fusion, Depth Consistency and Radar for Unified 3D Perception](https://arxiv.org/abs/2403.07746)|-|
|2023|ICCV|[OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception](https://arxiv.org/abs/2303.03991)|[Code](https://github.com/JeffWang987/OpenOccupancy)|

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
|3D Panoptic Segmentation|2024|CVPR|[PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation](https://arxiv.org/abs/2306.10013)|[Code](https://github.com/Robertwyq/PanoOcc)|
|BEV Segmentation|2024|CVPRW|[OccFeat: Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation Networks](https://arxiv.org/abs/2404.14027)|[Code](https://github.com/valeoai/Occfeat)|

### Detection
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|3D Object Detection|2024|NeurIPS|[Towards Flexible 3D Perception: Object-Centric Occupancy Completion Augments 3D Object Detection](https://arxiv.org/abs/2412.05154)|[Code](https://github.com/Ghostish/ObjectCentricOccCompletion)|
|3D Object Detection|2024|CVPR|[Learning Occupancy for Monocular 3D Object Detection](https://arxiv.org/abs/2305.15694)|[Code](https://github.com/SPengLiang/OccupancyM3D)|
|3D Object Detection|2024|AAAI|[SOGDet: Semantic-Occupancy Guided Multi-view 3D Object Detection](https://arxiv.org/abs/2308.13794)|[Code](https://github.com/zhouqiu/SOGDet)|
|3D Object Detection|2024|arXiv|[UltimateDO: An Efficient Framework to Marry Occupancy Prediction with 3D Object Detection via Channel2height](https://arxiv.org/abs/2409.11160)|-|

### Tracking
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|Object Tracking|2025|ICRA|[TrackOcc: Camera-based 4D Panoptic Occupancy Tracking](https://arxiv.org/abs/2503.08471)|[Code](https://github.com/Tsinghua-MARS-Lab/TrackOcc)|

### Dynamic Perception
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|3D Flow Prediction|2024|CVPR|[Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications](https://arxiv.org/abs/2311.17663)|[Code](https://github.com/haomo-ai/Cam4DOcc)|
|3D Flow Prediction|2024|arXiv|[Let Occ Flow: Self-Supervised 3D Occupancy Flow Prediction](https://arxiv.org/abs/2407.07587)|[Project Page](https://eliliu2233.github.io/letoccflow/)|

### Generation
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|Scene Generation|2024|ECCV|[Pyramid Diffusion for Fine 3D Large Scene Generation](https://arxiv.org/abs/2311.12085) (Oral paper)|[Code](https://github.com/yuhengliu02/pyramid-discrete-diffusion)|
|Scene Generation|2024|CVPR|[SemCity: Semantic Scene Generation with Triplane Diffusion](https://arxiv.org/abs/2403.07773)|[Code](https://github.com/zoomin-lee/SemCity)|
|Scene Generation|2024|arXiv|[OccScene: Semantic Occupancy-based Cross-task Mutual Learning for 3D Scene Generation](https://arxiv.org/abs/2412.11183)|-|
|Scene Generation|2024|arXiv|[UniScene: Unified Occupancy-centric Driving Scene Generation](https://arxiv.org/abs/2412.05435)|[Project Page](https://arlo0o.github.io/uniscene/)|
|Scene Generation|2024|arXiv|[InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models](https://arxiv.org/abs/2412.03934)|[Project Page](https://research.nvidia.com/labs/toronto-ai/infinicube/)|
|Scene Generation|2024|arXiv|[SyntheOcc: Synthesize Geometric-Controlled Street View Images through 3D Semantic MPIs](https://arxiv.org/abs/2410.00337)|[Project Page](https://len-li.github.io/syntheocc-web/)|


### Navigation
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|Navigation for Air-Ground Robots|2024|RA-L|[HE-Nav: A High-Performance and Efficient Navigation System for Aerial-Ground Robots in Cluttered Environments](https://arxiv.org/abs/2410.05079)|[Project Page](https://jmwang0117.github.io/HE-Nav/)|
|Navigation for Air-Ground Robots|2024|ICRA|[AGRNav: Efficient and Energy-Saving Autonomous Navigation for Air-Ground Robots in Occlusion-Prone Environments](https://arxiv.org/abs/2403.11607)|[Code](https://github.com/jmwang0117/AGRNav)|
|Navigation for Air-Ground Robots|2024|arXiv|[OMEGA: Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Model](https://arxiv.org/abs/2408.10618)|[Project Page](https://jmwang0117.github.io/OMEGA/)|

### World Models
|Specific Task|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|4D Occupancy Forecasting|2025|ICLR|[OccProphet: Pushing Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with Observer-Forecaster-Refiner Framework](https://arxiv.org/abs/2502.15180)|[Code](https://github.com/JLChen-C/OccProphet)|
|4D Occupancy Generation|2025|ICLR|[DynamicCity: Large-Scale LiDAR Generation from Dynamic Scenes](https://arxiv.org/abs/2410.18084) (Spotlight)|[Project Page](https://dynamic-city.github.io/)|
|4D Occupancy Forecasting and Motion Planing|2025|ICLR|[Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving](https://arxiv.org/abs/2502.07309)|[Code](https://github.com/getterupper/PreWorld)|
|4D Occupancy Forecasting and Generation|2025|AAAI|[Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving](https://arxiv.org/abs/2408.14197)|[Project Page](https://drive-occworld.github.io/)|
|4D Occupancy Forecasting and Motion Planing|2025|ICRA|[RenderWorld: World Model with Self-Supervised 3D Label](https://arxiv.org/abs/2409.11356)|-|
|4D Occupancy Forecasting, Motion Planing, and Scene Understanding|2025|ICRA|[Occ-LLM: Enhancing Autonomous Driving with Occupancy-Based Large Language Models](https://arxiv.org/abs/2502.06419)|-|
|4D Occupancy Forecasting|2025|arXiv|[LEAP: Enhancing Vision-Based Occupancy Networks with Lightweight Spatio-Temporal Correlation](https://arxiv.org/abs/2502.15438)|-|
|4D Occupancy Forecasting and Motion Planing|2024|ECCV|[OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](https://arxiv.org/abs/2311.16038)|[Project Page](https://wzzheng.net/OccWorld/)|
|4D Occupancy Forecasting|2024|CVPR|[UnO: Unsupervised Occupancy Fields for Perception and Forecasting](https://arxiv.org/abs/2406.08691) (Oral paper)|[Project Page](https://waabi.ai/uno/)|
|4D Representation Learning Framework|2024|CVPR|[DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving](https://arxiv.org/abs/2405.04390)|-|
|4D Occupancy Forecasting|2024|CVPR|[Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications](https://arxiv.org/abs/2311.17663)|[Code](https://github.com/haomo-ai/Cam4DOcc)|
|4D Occupancy Forecasting|2024|AAAI|[Semantic Complete Scene Forecasting from a 4D Dynamic Point Cloud Sequence](https://arxiv.org/abs/2312.08054)|[Project Page](https://scsfnet.github.io)|
|4D Occupancy Forecasting and Motion Planing|2024|arXiv|[An Efficient Occupancy World Model via Decoupled Dynamic Flow and Image-assisted Training](https://arxiv.org/abs/2412.13772)|-|
|4D Occupancy Forecasting|2024|arXiv|[Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting](https://arxiv.org/abs/2411.14169)|-|
|4D Occupancy Forecasting and Generation|2024|arXiv|[DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model](https://arxiv.org/abs/2410.10429)|[Project Page](https://gusongen.github.io/DOME/)|
|4D Occupancy Forecasting|2024|arXiv|[FSF-Net: Enhance 4D Occupancy Forecasting with Coarse BEV Scene Flow for Autonomous Driving](https://arxiv.org/abs/2409.15841)|-|
|4D Occupancy Forecasting, Motion Planing, and Reasoning|2024|arXiv|[OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving](https://arxiv.org/abs/2409.03272)|-|
|4D Occupancy Generation|2024|arXiv|[OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving](https://arxiv.org/abs/2405.20337)|[Project Page](https://wzzheng.net/OccSora/)|
|4D Occupancy Forecasting|2023|CVPR|[Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting](https://arxiv.org/abs/2302.13130)|[Project Page](https://www.cs.cmu.edu/~tkhurana/ff4d/index.html)|



### Unified Autonomous Driving Algorithm Framework
|Specific Tasks|Year|Venue|Paper Title|Link|
|:-:|:-:|:-:|-|-|
|Occupancy Prediction, 3D Object Detection, Online Mapping, Multi-object Tracking, Motion Prediction, Motion Planning|2024|CVPR|[DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving](https://arxiv.org/abs/2405.04390)|-|
|Occupancy Prediction, 3D Object Detection|2024|RA-L|[UniScene: Multi-Camera Unified Pre-training via 3D Scene Reconstruction for Autonomous Driving](https://arxiv.org/abs/2305.18829)|[Code](https://github.com/chaytonmin/UniScene)|
|Occupancy Prediction, 3D Object Detection, HD map reconstruction|2024|arXiv|[GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving](https://arxiv.org/abs/2411.12452)|[Code](https://github.com/Public-BOTs/GaussianPretrain)|
|Occupancy Forecasting, Motion Planning|2024|arXiv|[Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving](https://arxiv.org/abs/2408.14197)|-|
|Occupancy Prediction, 3D Object Detection, BEV segmentation, Motion Planning|2023|ICCV|[Scene as Occupancy](https://arxiv.org/abs/2306.02851)|[Code](https://github.com/OpenDriveLab/OccNet/tree/occnet)|



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
If you have any questions, please feel free to get in touch:
```
lap-pui.chau@polyu.edu.hk
huaiyuan.xu@polyu.edu.hk
```
If you are interested in joining us as a Ph.D. student to research **computer vision, machine learning**, please feel free to contact [Professor Chau](https://www.polyu.edu.hk/eee/people/academic-staff-and-teaching-staff/prof-chau-lap-pui/):
```
lap-pui.chau@polyu.edu.hk
```


