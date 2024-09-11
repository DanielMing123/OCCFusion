# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n OCCFusion python=3.8 -y
conda activate OCCFusion
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**c. Install mmengine, mmcv, mmdet, mmdet3d, and mmseg.**
```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim install "mmdet3d>=1.1.0"
pip install "mmsegmentation>=1.0.0"
```

**d. Install others.**
```shell
pip install focal_loss_torch
```
**e. Download code and backbone pretrain weight.**
```shell
git clone https://github.com/DanielMing123/OCCFusion.git
cd OCCFusion
mkdir ckpt
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```
**f. Download Fixed param [here](https://drive.google.com/drive/folders/15riDPe25gVZ79jGeamfftBrzRBbcfQjP?usp=sharing). The OCCFusion repo core structure should be like the following**
```
OCCFusion
├── ckpt/
├── configs/
├── fix_param_small/
├── occfusion/
├── tools/
├── data/
│   ├── nuscenes/
```
