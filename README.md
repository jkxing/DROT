# DROT
Code for SIGGRAPH ASIA 2022 paper [*Differentiable Rendering using RGBXY Derivatives and Optimal Transport*](https://jkxing.github.io/academic/publication/DROT)

## Installation

Clone this repo first

We suggest create a new Anaconda environment.
```
conda create -n DROT python=3.8
#Install PyTorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install geomloss 
```
pip install pykeops
pip install geomloss
```
Install customized PyTorch3D
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
git clone https://github.com/jkxing/pytorch3d
cd pytorch3d
python setup.py install
```
Install Nvdiffrast
```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```
Install other tools
```
conda install -c pytorch ignite 
pip install tensorboard,matplotlib,xmltodict,pyglm,imageio,lpips,opencv-python
```
if some error occurs, the following tips may help:
- GCC version should under 10
- Install correct version of nvidia-gl driver(server version may fail)
- CUDA version/Python-dev version should exactly match everywhere

## Demo
Please download sample data [here](https://drive.google.com/file/d/1roNV8_sZxD11R0FwktTkyu_PL2Q7TrAA/view?usp=sharing) and extract it to `data/`.

### Evaluation 

We provide all 8 scenes used in evaluation part. Please refer to experiments/evaluation/*.json for detail settings.

**Usage**
```
python experiments/evaluation/optim.py [translation/rotation/...]
```

### Furniture Layout 

We provide 12 scenes shown in paper and supplemental materials collected from [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) for furniture layout application. Please refer to experiments/furniture/*.json for detail settings.

**Usage**
```
python experiments/furniture/optim.py [1-12]
```

### Human Pose Fitting
Please download SMPL model and sample texture from [here](https://smpl.is.tue.mpg.de/) and extract it to data/human/, files should include
```
data/human/
├── basicModel_m_lbs_10_207_0_v1.0.0.pkl
├── tex.png
├── UV_Processed.mat
└── UV_symmetry_transforms.mat
```
**Usage**
```
python experiments/smpl/optim.py
```
You can specify random seed for different initialization.

### Facial Expression Reconstruction

Because the face model are not free for public, we do not provide the model and optimization code. But the code are similar to other applications, so you can implement it for your own model.


## Jittor(JRender) Version
Since Jittor doesn't provide method such as multi object rendering, we only provide a core implementation and a single object translation demo in submodule jittor.  Please install jittor follow jittor/readme.md.