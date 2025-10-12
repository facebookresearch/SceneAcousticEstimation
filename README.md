# Scene-wide Acoustic Parameter Estimation

From WASPAA 2025 Paper: Scene-wide Acoustic Parameter Estimation

Authors:
Ricardo Falcon-Perez¹, Ruohan Gao², Gregor Mueckl, Sebastia V. Amengual Gari³, Ishwarya Ananthabhotla³
<br>
¹ Aalto University, Finland
² University of Maryland, College Park
³ Meta - Reality Labs Research, USA

### Citation
If you use the model, code, or the MRAS dataset please cite this work as:

```bibtex
@inproceedings{falconperez2025,
    author  = "{Falcón Pérez}, Ricardo and Gao, Ruohan and Mueckl, Gregor, and {Amengual Gari}, Sebastiva V. and Ananthabhotla, Ishwarya",
    year    = {2025},
    title   = {Scene Wide Acoustic Parameter Estimation},
    booktitle   = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
}
```


## Overview
This is the official repository for the WASPAA 2025 paper Scene-wide Acoustic Parameter Estimation. The goal of this project is to predict heatmaps of acoustic parameters covering the full scene given a particular source in a single inference step. 
We also include the **Multi-room Apartments Simulation** (MRAS) dataset. This is a large scale synthetic dataset that models indoor scenes by connecting multiple rooms.
For more details of the dataset please refer to the paper.

This repo contains: 
- Pre trained model  [**coming soon**]
- Code (training, inference, dataset preprocessing)
- MRAS Dataset
    - Raw [**coming soon**]
        - Scene geometries
        - 2nd order ambisonid rirs
    - Preprocessed
        - Scene floormaps
        - RIRs packed as LMDB




## Prerequisites

### Creating the environment
Use conda (or mamba) to install the pacakges in ```environment.yml```. For example:

```bash
conda config --add channels conda-forge  
conda config --show channels
conda env create -f environment.yml -n sceacoustics 
```

Alternatively, the environment can be created manually. This can make it easier to avoid conflicts in some systems:
```bash
conda config --add channels conda-forge  
conda config --show channels
conda env create -n sceacoustics python=3.10 numpy matplotlib seaborn 
source activate sceacoustics
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tensorboard torchinfo scikit-learn mlflow scipy lmdb python-lmdb tqdm pyyaml easydict configargparse jupyterlab pot wandb
conda install librosa -c conda-forge   # We need librosa to enable soundfile backend for torchaudio
pip install spaudiopy hexagdly open3d torchmetrics ptwt  # NOTE: open3d is tricky to install in headless mode in some linux environments
```

Then we can initialize the submodules and download them.
```bash
git submodule init
git submodule update --recursive
```
Finally, add the submodules to the environment.

```bash
cd ranger...  # navigate to local ranger directory in the repo
pip install -e .
```



```
conda create env -n scenewide -f environment.yml 
```

### Get the submodules

```
git sub update decayfitnet
```






---

## 📄 License

**SceneAcousticEstimation** is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share and adapt the dataset, even for commercial use, as long as proper attribution is given.

See the [LICENSE](./LICENSE) file for full terms.




