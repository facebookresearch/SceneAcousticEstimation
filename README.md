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
        - 2nd order ambisonic rirs
        - Acoustic parameters and other metadata
    - Preprocessed (packed as LMDB)
        - Scene floormaps
        - Acoustic heatmaps
        - RIRs


***
## Data
This repo includes the preprocessed MRAS/Replica data (via Git LFS) as ZIP files and expects the following layout:
```bash
Data/
  preprocessed/         # LMDBs as zip files (tracked with Git LFS)
    mras_relfloor_10x10_moreparams.lmdb.zip
    replica_relcenter_10x10_moreparams.lmdb.zip
    rirs_mono_mras_grids.lmdb.zip
    rirs_mono_scenes_18.lmdb.zip
  raw/                  # Raw data (not yet included)
```
### Getting the data
Install and pull LFS content:
```bash
git lfs install
git lfs pull
```
Unzip LMDB archives in place:
```bash
unzip Data/preprocessed/*.zip -d Data/preprocessed/
```
<b>Note on raw data:</b> Data/raw/ is reserved for the raw MRAS assets (scene geometries, ambisonic RIRs, etc.) and is not included yet.


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


## Training: Reproducing the Main Experiments

The paths/filenames below assume you’ve unzipped LMDBs into Data/preprocessed/.

Training with MRAS:
```
  python train_basic.py -c configs/train_more_parameters.yaml \
    --exp_name $exp_n --use_triton --job_id $job_id --task_id $param \
    --num_workers $num_w --seed 1111 \
    --dataset 'mras' --fold 'fixed_1' \
    --n_files_per_scene 10000000 --max_length 24000 --use_augmentation_getitem \
    --fmap_use_soft_sources \
    --read_lmdb --fname_lmdb 'rirs_mono_mras_grids.lmdb' \
    --read_lmdb_maps --fname_lmdb_maps 'mras_relfloor_10x10_moreparams.lmdb' \
    --rir_output_channels 0
```

Training with Replica
```bash
  python train_basic.py -c configs/mras_more_parameters.yaml \
    --exp_name $exp_n --use_triton --job_id $job_id --task_id $param \
    --num_workers $num_w --seed 1111 \
    --dataset 'replica' --fold 'balanced_1' \
    --n_files_per_scene 10000000 --max_length 24000 --use_augmentation_getitem \
    --read_lmdb --fname_lmdb 'rirs_mono_scenes_18.lmdb' \
    --read_lmdb_maps --fname_lmdb_maps 'replica_relcenter_10x10_moreparams.lmdb' \
    --rir_output_channels 0
```

### Key flags & expected files
```python
--read_lmdb --fname_lmdb:
    rirs_mono_mras_grids.lmdb (MRAS RIRs)
    rirs_mono_scenes_18.lmdb (Replica RIRs)

--read_lmdb_maps --fname_lmdb_maps:
    mras_relfloor_10x10_moreparams.lmdb (MRAS maps)
    replica_relcenter_10x10_moreparams.lmdb (Replica maps)

Output channels: --rir_output_channels 0 (mono)
```

Inference only:
```bash    
python train_basic.py -c configs/train_more_parameters.yaml --exp_name $exp_n --use_triton  --job_id $job_id --task_id $param  --num_workers $num_w --seed 1111 \
    --validation_checkpoint 7033621_3_table01_1111_10x10_moreparams_replica_triton_replica_balanced_4 --do_validation \
    --dataset 'replica' --fold 'balanced_4' --n_files_per_scene 10000000 --max_length 24000 --use_augmentation_getitem \
    --read_lmdb --fname_lmdb 'rirs_mono_scenes_18.lmdb' --read_lmdb_maps --fname_lmdb_maps 'replica_relcenter_10x10_moreparams.lmdb' --rir_output_channels 0 \
```

```bash
python train_basic.py -c configs/mras_more_parameters.yaml --exp_name $exp_n --use_triton  --job_id $job_id --task_id $param  --num_workers $num_w --seed 1111 \
    --validation_checkpoint 7046468_0_mras_1111_10x10_moreparams_June15_triton_mras_fixed_1 --do_validation \
    --dataset 'mras' --fold 'fixed_1' --n_files_per_scene 10000000 --max_length 24000 --use_augmentation_getitem --fmap_use_soft_sources \
    --read_lmdb --fname_lmdb 'rirs_mono_mras_grids.lmdb' --read_lmdb_maps --fname_lmdb_maps 'mras_relfloor_10x10_moreparams.lmdb' --rir_output_channels 0 \
```
---

## 📄 License

**SceneAcousticEstimation** is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share and adapt the dataset, even for commercial use, as long as proper attribution is given.

See the [LICENSE](./LICENSE) file for full terms.




