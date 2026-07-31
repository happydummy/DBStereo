<div align="center">
<h1>DBStereo</h1>
<h3>Decoupling Bidirectional Geometric Representations of 4D cost volume with 2D convolution</h3>


</sup> Carizon

Arxiv 2025


<!-- <a href="https://arxiv.org/abs/2509.02415"><img src='https://img.shields.io/badge/arXiv-DBStereo-red' alt='Paper PDF'></a>   -->

</div>

## News
* **` Dec. 10th, 2025`:** We release the code and weight of DBStereo, along with documentation and training/evaluation scripts.
* **` Sep. 2th, 2025`:** We released our paper on [Arxiv](https://arxiv.org/abs/2509.02415). Code/Models are coming soon. Please stay tuned! ☕️


## Getting Started

Please see [get_started.md](docs/get_started.md). 
<!-- <img width="1564" height="1018" alt="image" src="https://github.com/user-attachments/assets/8f845630-62fe-4e75-880b-08df7dfe6118" />

<img width="2238" height="910" alt="image" src="https://github.com/user-attachments/assets/bf229dd0-0e58-4b43-9c50-1ca9b11732d6" /> -->
## Checkpoint
|                         Model                          |         `EPE`         |                                                                      Configuration |                                                                              Download |
|:------------------------------------------------------:|:---------------------:|-----------------------------------------------------------------------------------:|------------------------------------------------------------------------------------:|
|     [DBStereo-S](https://arxiv.org/abs/2406.19833) (*Ours*)  |         0.64          |   [dbstereo_s_sceneflow.yaml](./cfgs/dbstereo/dbstereo_s_sceneflow.yaml) |   [ckpt](https://drive.google.com/drive/folders/1ESSzg1J7TNWaoYeAneV1aR3mubcxd0gn) |
|     [DBStereo-M](https://arxiv.org/abs/2406.19833) (*Ours*)  |         0.51          |   [dbstereo_m_sceneflow.yaml](./cfgs/dbstereo/dbstereo_m_sceneflow.yaml) |   [ckpt](https://drive.google.com/drive/folders/1ESSzg1J7TNWaoYeAneV1aR3mubcxd0gn) |
|     [DBStereo-L](https://arxiv.org/abs/2406.19833) (*Ours*)  |         0.46          |   [dbstereo_l_sceneflow.yaml](./cfgs/dbstereo/dbstereo_l_sceneflow.yaml) |   [ckpt](https://drive.google.com/drive/folders/1ESSzg1J7TNWaoYeAneV1aR3mubcxd0gn) |

## Acknowledgement
DBStereo is greatly inspired by the following outstanding contributions to the open-source community: [OpenStereo & LightStereo](https://github.com/XiandaGuo/OpenStereo)
## Citation
```
@article{wei2025decoupling,
  title={Decoupling Bidirectional Geometric Representations of 4D cost volume with 2D convolution},
  author={Wei, Xiaobao and Shu, Changyong and Yue, Zhaokun and Huang, Chang and Liu, Weiwei and Yang, Shuai and Yang, Lirong and Gao, Peng and Zhang, Wenbin and Zhu, Gaochao and others},
  journal={arXiv preprint arXiv:2509.02415},
  year={2025}
}
@article{OpenStereo,
        title={OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline},
        author={Guo, Xianda and Zhang, Chenming and Lu, Juntao  and Wang, Yiqi and Duan, Yiqun and Yang, Tian and Zhu, Zheng and Chen, Long},
        journal={arXiv preprint arXiv:2312.00343},
        year={2023}
}
@inproceedings{guo2025lightstereo,
        title={Lightstereo: Channel boost is all you need for efficient 2d cost aggregation},
        author={Guo, Xianda and Zhang, Chenming and Zhang, Youmin and Zheng, Wenzhao and Nie, Dujun and Poggi, Matteo and Chen, Long},
        booktitle={ICRA},
        year={2025}
}
```





