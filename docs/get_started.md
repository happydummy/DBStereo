# Get Started

## Installation

1. clone this repo.
    ```
    git clone https://github.com/happydummy/DBStereo.git
    ```
2. Create a conda environment by:
   ```
   conda create -n dbstereo python=3.8 
   ```

   
3. Install pytorch by [Anaconda](https://pytorch.org/get-started/locally/):
   ```
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   ```
4. Install other dependencies by pip:
   ```
   pip install -r requirements.txt
   ```

## Prepare dataset

See [prepare dataset](2.prepare_dataset.md).

## Train

Train a model with a Single GPU
```
python tools/train.py --cfg_file cfgs/dbstereo/dbstereo_s_sceneflow.yaml
```
Multi-GPU Training on Single Node
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:23456 tools/train.py --dist_mode --cfg_file cfgs/dbstereo/dbstereo_s_sceneflow.yaml
```

- `--config` The path to the config file.
- `--dist_mode` If specified, the program will use DDP to train.
- your exp will saved in '/save_root_dir/DATASET_NAME/MODEL_NAME/config_file_perfix/extra_tag', save_root_dir and extra_tag can specified in train argparse

## Val

Evaluate the trained model by
```
python tools/eval.py --cfg_file cfgs/dbstereo/dbstereo_l_sceneflow.yaml --eval_data_cfg_file cfgs/sceneflow_eval.yaml --pretrained_model your_pretrained_ckpt_path
```
Generalization Evaluation:
```
python tools/eval.py --cfg_file cfgs/dbstereo/dbstereo_s_sceneflow.yaml --eval_data_cfg_file cfgs/eth3d_eval.yaml --pretrained_model your_pretrained_ckpt_path
python tools/eval.py --cfg_file cfgs/dbstereo/dbstereo_s_sceneflow.yaml --eval_data_cfg_file cfgs/middlebury_eval.yaml --pretrained_model your_pretrained_ckpt_path
python tools/eval.py --cfg_file cfgs/dbstereo/dbstereo_s_sceneflow.yaml --eval_data_cfg_file cfgs/kitti15_eval.yaml --pretrained_model your_pretrained_ckpt_path
python tools/eval.py --cfg_file cfgs/dbstereo/dbstereo_s_sceneflow.yaml --eval_data_cfg_file cfgs/kitti12_eval.yaml --pretrained_model your_pretrained_ckpt_path
```

- `--cfg_file` The path to the config file.
- `--eval_data_cfg_file` The dataset config you want to eval.
- `--pretrained_model` your pre-trained checkpoint
