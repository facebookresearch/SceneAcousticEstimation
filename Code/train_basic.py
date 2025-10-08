# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import math
import argparse
#import configargparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms
import torchvision
import torch.optim as optim
import torch.profiler
import matplotlib.pyplot as plt
import mlflow
import ot
import wandb

mlflow.set_tracking_uri(uri="http://localhost:8080")

from easydict import EasyDict
from datetime import timedelta
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from mlflow.models import infer_signature
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSIM
from torchmetrics.functional.image import image_gradients
from torchmetrics.functional.image import spatial_correlation_coefficient as SCC


from models.map_models import UnetBasic, UnetFilm, Encoder, Decoder, UnetConcatBottleNeck, CustomThreshold, UnetBottleNeck
from datasets.soundspaces_dataset import SoundspacesDataset, split_by_srcs
import augmentation
import features
import utils
import losses
import pytorch_acoustics
import GWA_sampling_analysis
import parameters

def get_params(config: Dict,  # config from parameters.py
               dataset_name: str = 'soundspaces', 
               n_files_per_scene=1e7, #100000
               scenes: List[int] = [3],  # [3, 7], # 5 is apartment_0
               mode: str = 'debug') -> Tuple[Dict, Dict, Dict]:
    
    if dataset_name == 'soundspaces':
        params = {'datapath': '/home/ricfalcon/00data/soundspaces/data',
                  'scenes': scenes,
                  'n_files_per_scene': config['n_files_per_scene'],
                  'max_length': config['max_length'],  # 48000 for replica, 24000 for mras
                  'read_lmdb': config['read_lmdb'],
                  'fname_lmdb': config['fname_lmdb'],  # 'fname_lmdb': 'rirs_mono_scenes_17.lmdb', 'rirs_1storder_scenes_17.lmdb', 'rirs_2ndorder_scenes_17.lmdb'
                  'read_lmdb_maps': config['read_lmdb_maps'],
                  'fname_lmdb_maps': config['fname_lmdb_maps'],  # maps.lmdb, maps_relcenter_10x10.lmdb maps_mp3d_relcenter_25x25.lmdb  maps_replica_relcenter_10x10.lmdb
                  'rir_output_channels': config['rir_output_channels']}
    else:
        raise ValueError(f'ERROR: Unrecognized dataset {dataset_name}')

    # Processors using configs
    params_processor = {}
    params_processor['resolution'] = config['fmap_resolution']
    params_processor['height_selection'] = config['fmap_height_selection'] # relative_ceiling  # relative_center
    params_processor['slice_coord'] = config['fmap_slice_coord']   # 1.0 for relative_celing, 0.5 is best for relative_center
    params_processor['use_slices_variance'] = config['fmap_use_slices_variance'] 
    params_processor['bbox_channel'] = config['fmap_bbox_channel'] 
    params_processor['pos_enc_d'] = config['fmap_pos_enc_d'] 
    params_processor['xlim'] = config['fmap_xlim']   # -25, 25 for mp3d
    params_processor['use_soft_sources'] = config['fmap_use_soft_sources'] 

    params_acu_processor = {}
    params_acu_processor['resolution'] = config['fmap_resolution']
    params_acu_processor['k'] = config['acumap_k']
    params_acu_processor['s'] = config['acumap_s']
    params_acu_processor['p'] = config['acumap_p']
    params_acu_processor['std'] = config['acumap_std']
    params_acu_processor['xlim'] = params_processor['xlim']
    params_acu_processor['parameters'] = config['acumap_parameters']
    params_acu_processor['freq_bands'] = config['acumap_frequency_bands']
    params_acu_processor['distances'] = config['acumap_distances']

    params['normalizer_vmins'] = config['normalizer_vmins']
    params['normalizer_vmaxs'] = config['normalizer_vmaxs']
    params['normalizer_logchannels'] = config['normalizer_logchannels']

    if False:
        params_processor = {}
        params_processor['resolution'] = 128
        params_processor['height_selection'] = ['relative_center']  # relative_ceiling  # relative_center
        params_processor['slice_coord'] = 0.5  # 1.0 for relative_celing, 0.5 is best for relative_center
        params_processor['use_slices_variance'] = False
        params_processor['bbox_channel'] = 1
        params_processor['pos_enc_d'] = 64
        params_processor['xlim'] = [-10, 10]  # -25, 25 for mp3d

        if False:
            # Realtive to the center
            params_processor = {}
            params_processor['resolution'] = 128
            params_processor['height_selection'] = ['relative_center']
            params_processor['slice_coord'] = 0.25
            params_processor['use_slices_variance'] = False
            params_processor['bbox_channel'] = 1
            params_processor['pos_enc_d'] = 64
            params_processor['xlim'] = [-10, 10]

        if False:
            # Using 2 fixed slices
            params_processor = {}
            params_processor['resolution'] = 128
            params_processor['height_selection'] = ['relative_ceiling', 'relative_floor']
            params_processor['slice_coord'] = [0.5, 0.5]
            params_processor['use_slices_variance'] = False
            params_processor['bbox_channel'] = 2
            params_processor['pos_enc_d'] = 64
            params_processor['xlim'] = [-10, 10]

        # Alternative inputs with many random slices of the floormaps
        if False:
            params_processor = {}
            params_processor['resolution'] = 128
            params_processor['height_selection'] = ['random', 'random', 'random', 'random', 'random', 'relative_floor', 'relative_ceiling']
            params_processor['slice_coord'] = [None, None, None, None, None, 0.5, 0.5]
            params_processor['use_slices_variance'] = True
            params_processor['bbox_channel'] = 8
            params_processor['pos_enc_d'] = 64
            params_processor['xlim'] = [-10, 10]

        params_acu_processor = {}
        params_acu_processor['resolution'] = 128
        params_acu_processor['k'] = 5
        params_acu_processor['s'] = 1
        params_acu_processor['p'] = 2
        params_acu_processor['std'] = 1
        params_acu_processor['xlim'] = params_processor['xlim']

        # Only for matterport
        if False:
            params_acu_processor = {}
            params_acu_processor['resolution'] = 128
            params_acu_processor['k'] = 9
            params_acu_processor['s'] = 1
            params_acu_processor['p'] = 4
            params_acu_processor['std'] = 1
            params_acu_processor['xlim'] = params_processor['xlim']
    
    return params, params_processor, params_acu_processor

def plot_and_save_floormap(floormap, floormap_processor, fname='test', scene_name=None):
    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(1,floormap.shape[-3], figsize=(4 * floormap.shape[-3], 4))
    axs = axs.flatten()
    ctr = 0
    all_imgs = []
    for c in range(floormap.shape[-3]):
        im = axs[ctr].imshow(floormap[c].numpy(), origin='lower', extent=[floormap_processor.xlim[0], floormap_processor.xlim[1], floormap_processor.xlim[0], floormap_processor.xlim[1]], aspect=1, interpolation='antialiased', interpolation_stage='data') 
        all_imgs.append(im)
        if c < 2:
            axs[ctr].scatter(-1 * src[..., 1].numpy() - 0, src[..., 0].numpy() - 0, s=100, marker='x')
            axs[ctr].scatter(-1 * rec[..., 1].numpy() + 0, rec[..., 0].numpy() - 0, s=100, marker='o')
        axs[ctr].grid(visible=False)
        axs[ctr].set_title(f'{floormap_processor.channel_names[c]}')
        ctr += 1
    plt.tight_layout()
    plt.show()

    plt.savefig(f'{output_dir}/{scene_name}_floormap_{fname}.pdf', format='pdf')
    #print(f'floormap {floormap[2].min()}  {floormap[2].max()}')

    # Dump tensors, for further plotting later
    torch.save(floormap, f'{output_dir}/{scene_name}_floormap_{fname}.pt')

def plot_and_save_acumap(acumap, src, acumap_processor, freqs_toplot=None, cbar_labels=None, fname='test', scene_name=None):
    # Plot acumap
    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.rcParams['font.size'] = '10'
    if scene_name is not None:
        scene_name = scene_name.replace('/', '-')

    # Useful when we want to plot less channels than are available
    if freqs_toplot is None:
        freqs_toplot = np.arange(len(acumap_processor.freq_bands)).tolist()
    fig, axs = plt.subplots(len(acumap_processor.parameters), len(freqs_toplot), figsize=(16,14), sharex=True, sharey=True)
    axs = axs.flatten()

    if cbar_labels is None:
        cbar_labels = [""] * len(acumap_processor.parameters)
    
    ctr = 0
    ctr_freq = -1
    all_imgs = []
    print(freqs_toplot)
    print(acumap.shape[-3])
    for c in range(acumap.shape[-3]):
        if ctr_freq >= len(acumap_processor.freq_bands) - 1:
            ctr_freq = 0
        else:
            ctr_freq += 1
        if ctr_freq not in freqs_toplot:
            #print(f"SKIP CHANNEL {ctr_freq}")
            continue
        #print(c)
        im = axs[ctr].imshow(acumap[c].numpy(), origin='lower', extent=[acumap_processor.xlim[0], acumap_processor.xlim[1], acumap_processor.xlim[0], acumap_processor.xlim[1]], aspect=1, interpolation='antialiased', interpolation_stage='data') 
        all_imgs.append(im)
        axs[ctr].scatter(-1 * src[..., 1].numpy() - 0, src[..., 0].numpy() - 0, s=100, marker='x')
        #axs[ctr].scatter(-1 * src[..., 0].numpy() + 20, -1 * src[..., 1].numpy() + 20, s=100, marker='x')  # For scenes_proposed_v2
    
        #axs[ctr].scatter(-1 * src[..., 1].numpy() + 20, src[..., 0].numpy() - 0, s=100, marker='x')  # for scenes_processed_v3
        
        axs[ctr].grid(visible=False)
        if len(acumap_processor.channel_names) > 0:
            axs[ctr].set_title(f'{acumap_processor.channel_names[c]}')
        ctr += 1
    #plt.tight_layout()
    #plt.show()
    
    # Colorbars
    tmp_id = len(freqs_toplot)
    all_c50 = all_imgs[tmp_id*0:tmp_id*1]
    all_t30 = all_imgs[tmp_id*1:tmp_id*2]
    all_drr = all_imgs[tmp_id*2:tmp_id*3]
    all_edt = all_imgs[tmp_id*3:tmp_id*4]
    vmin_row0 = min([im.get_clim()[0] for im in all_c50])
    vmax_row0 = max([im.get_clim()[1] for im in all_c50])
    vmin_row1 = min([im.get_clim()[0] for im in all_t30])
    vmax_row1 = max([im.get_clim()[1] for im in all_t30])
    vmin_row2 = min([im.get_clim()[0] for im in all_drr])
    vmax_row2 = max([im.get_clim()[1] for im in all_drr])
    vmin_row3 = min([im.get_clim()[0] for im in all_edt])
    vmax_row3 = max([im.get_clim()[1] for im in all_edt])
    
    # manual limits
    #vmax_row0 = 30
    #vmax_row1 = 4
    #vmax_row2 = 30
    for im in all_c50:
        im.set_clim(vmin_row0, vmax_row0)
    for im in all_t30:
        im.set_clim(vmin_row1, vmax_row1)
    for im in all_drr:
        im.set_clim(vmin_row2, vmax_row2)
    for im in all_edt:
        im.set_clim(vmin_row3, vmax_row3)
    
    if scene_name is not None:
        plt.suptitle(scene_name)

    #fig.subplots_adjust(bottom=.75, right=0.8)
    #plt.tight_layout(w_pad=10)
    plt.tight_layout()
    
    fig.colorbar(all_c50[0], ax=axs[tmp_id*0:tmp_id*1], use_gridspec=True, label=cbar_labels[0])
    fig.colorbar(all_t30[0], ax=axs[tmp_id*1:tmp_id*2], use_gridspec=True, label=cbar_labels[1])
    fig.colorbar(all_drr[0], ax=axs[tmp_id*2:tmp_id*3], use_gridspec=True, label=cbar_labels[2])
    fig.colorbar(all_edt[0], ax=axs[tmp_id*3:tmp_id*4], use_gridspec=True, label=cbar_labels[3])
    
    #plt.tight_layout(h_pad=25)
    plt.show()
    plt.savefig(f'{output_dir}/{scene_name}_acumap_{fname}.pdf', format='pdf')
    #print(f'floormap {floormap[2].min()}  {floormap[2].max()}')

    # Dump tensors, for further plotting later
    torch.save(acumap, f'{output_dir}/{scene_name}_acumap_{fname}.pt')

def get_dataset(dataset_name: str, params: Dict, floormap_processor: object, acumap_processor: object) -> torch.utils.data.Dataset:
    if dataset_name == 'soundspaces':
        dset = SoundspacesDataset(datapath=params['datapath'],
                                  scenes=params['scenes'],
                                  n_files_per_scene=params['n_files_per_scene'],
                                  read_rirs=True,
                                  read_scenes=True,
                                  read_floormaps=True,
                                  read_acumaps=True,
                                  max_length=params['max_length'],
                                  floormap_processor=floormap_processor,
                                  acumap_processor=acumap_processor)
    else:
        raise ValueError(f'ERROR: Unrecognized dataset {dataset_name}')

    

    return dset

def get_dataset_full(dataset_name: str, params: Dict, floormap_processor: object, acumap_processor: object, args: object) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    scenes_train = [0,1,2,3,4,6,7,8,9,10,11,12]  # No arpartment_0
    scenes_test = [13,14,15,16,17]
    #scenes_train = [3]  # No arpartment_0
    #scenes_test = [13]
    scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'frl_apartment_0', 'frl_apartment_1', 'apartment_1', 'room_2', 'room_0', 'office_3', 'office_2', 'office_4', 'hotel_0']
    scenes_test = ['room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
    #scenes_train = ['frl_apartment_4']  # 'frl_apartment_3', room_2 had problems
    #scenes_test = ['frl_apartment_4']  # apartment_2, office_1 is the smallest one

    # All scenes, just to compute min and max, or lmdbs, no apartment_0
    #scenes_train = ['apartment_1', 'apartment_2', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_3', 'frl_apartment_5', 'office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'room_2', 'hotel_0']
    scenes_train = ['apartment_0', 'apartment_1', 'apartment_2', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_3', 'frl_apartment_5', 'office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'room_2', 'hotel_0']

    # All scenes, matterport, to compute lmdbs
    #scenes_train = ['pLe4wQe7qrG', 'YFuZgdQ5vWj', 'oLBMNvg9in8', 'cV4RVeZvu5T', 'x8F5xyUWy9e', 's8pcmisQ38h', 'V2XKFyX4ASd', 'EDJbREhghzL', 'pRbA3pwrgk9', '17DRP5sb8fy', 'XcA2TqTSSAj', '29hnd4uzFmX', 'YmJkqBEsHnH', 'D7G3Y4RVNrH', 'i5noydFURQK', 'JF19kD82Mey', 'JmbYfDe2QKZ', 'EU6Fwq7SyZv', 'HxpKQynjfin', 'WYY7iVyf5p8', 'yqstnuAEVhm', 'aayBHfsNo7d', '8194nk5LbLH', 'e9zR4mvMWw7', 'jh4fc5c5qoQ', 'Pm6F8kyY3z2', 'GdvgFV5R1Z5', 'TbHJrupSAjP']

    if args['dataset'] == 'replica':
        if args['fold'] == 'all':  # All scenes, just to compute min and max, or lmdbs
            scenes_train = ['apartment_0', 'apartment_1', 'apartment_2', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_3', 'frl_apartment_5', 'office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'room_2', 'hotel_0']
        elif args['fold'] == 'all_no_zero':  # All scenes, just to compute min and max, or lmdbs, but not apartment_0
            scenes_train = ['apartment_1', 'apartment_2', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_3', 'frl_apartment_5', 'office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'room_2', 'hotel_0']
            #scenes_train = scenes_train[5::]  # Only when dumping maps, manually, to avoid OOM problems
        elif args['fold'] == 'debug':
            scenes_train = ['frl_apartment_4']  # 'frl_apartment_4', room_2 had problems
            scenes_test = ['frl_apartment_4']  # apartment_2, office_1 is the smallest one
        elif args['fold'] == '00':
            # Fold 00  (standard split)
            scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'room_2', 'room_0', 'frl_apartment_1', 'apartment_1', 'office_3', 'frl_apartment_0', 'office_2', 'hotel_0']
            scenes_test = ['room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
        elif args['fold'] == '00plus':
            scenes_train = ['apartment_0', 'frl_apartment_3', 'frl_apartment_5', 'frl_apartment_0', 'frl_apartment_1', 'apartment_1', 'room_2', 'room_0', 'office_3', 'office_2', 'office_4', 'hotel_0']
            scenes_test = ['room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
        elif args['fold'] == '0':
            # Fold 0 (same as fold 00, but with different order and 1 extra scene in train)
            scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'frl_apartment_0', 'frl_apartment_1', 'apartment_1', 'room_2', 'room_0', 'office_3', 'office_2', 'office_4', 'hotel_0']
            scenes_test = ['room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
        elif args['fold'] == '1':
            # Fold 1, similar to fold 0, but not the same
            scenes_train = ['frl_apartment_2', 'frl_apartment_4', 'room_1', 'room_0', 'frl_apartment_1', 'apartment_2', 'frl_apartment_0', 'office_0', 'office_1', 'office_3', 'office_4', 'hotel_0']
            scenes_test = ['room_2', 'frl_apartment_3', 'apartment_1', 'office_2', 'frl_apartment_5']
        elif args['fold'] == '2':
            # Fold 2, test apartments and rooms
            scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_1', 'frl_apartment_0', 'office_0', 'office_2', 'office_1', 'office_3', 'office_4', 'hotel_0']
            scenes_test = [ 'room_0', 'room_1', 'room_2', 'apartment_1', 'apartment_2']
        elif args['fold'] == '3':
            # Fold 3, test frl aprtment
            scenes_train = ['room_0', 'room_1', 'room_2', 'apartment_1', 'apartment_2', 'office_0', 'office_2', 'office_1', 'office_3', 'office_4', 'hotel_0']
            scenes_test = [ 'frl_apartment_3', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_1', 'frl_apartment_0' ]
        elif args['fold'] == '4':
            # Fold 4, test offices and hotel
            scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_1', 'frl_apartment_0', 'room_0', 'room_1', 'room_2', 'apartment_1', 'apartment_2',]
            scenes_test = ['office_0', 'office_2', 'office_1', 'office_3', 'office_4', 'hotel_0']

        # Balanced splits, for the final results
        elif args['fold'] == 'balanced_1':
            scenes_train = ['apartment_0', 'apartment_1', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_1', 'frl_apartment_4', 'office_2', 'office_1', 'office_3', 'office_4', 'room_1', 'hotel_0']
            scenes_test = ['apartment_2', 'frl_apartment_3', 'frl_apartment_0', 'office_0', 'room_2', 'room_0']
        elif args['fold'] == 'balanced_2':
            scenes_train = ['apartment_1', 'apartment_0', 'frl_apartment_4', 'frl_apartment_2', 'frl_apartment_5', 'frl_apartment_1', 'office_1', 'office_0', 'room_0', 'office_4', 'room_1', 'hotel_0']
            scenes_test = ['apartment_2', 'frl_apartment_0', 'frl_apartment_3', 'office_2', 'office_3', 'room_2']
        elif args['fold'] == 'balanced_3':   # this one is not that good, because apartment 0 is in test
            scenes_train = ['apartment_1', 'apartment_2', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_0', 'frl_apartment_4', 'office_2', 'office_0', 'room_0', 'room_2', 'room_1', 'hotel_0']
            scenes_test = ['apartment_0', 'frl_apartment_1', 'frl_apartment_3', 'office_1', 'office_3', 'office_4']
        elif args['fold'] == 'balanced_4':
            scenes_train = ['apartment_0', 'apartment_2', 'frl_apartment_3', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_5', 'office_0', 'office_2', 'room_1', 'office_3', 'room_2', 'room_0']
            scenes_test = ['apartment_1', 'frl_apartment_4', 'frl_apartment_2', 'office_1', 'office_4', 'hotel_0']
            
        elif args['fold'] == 'balanced_1_no_zero':  # no apartment_0
            scenes_train = ['apartment_1', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_1', 'frl_apartment_4', 'office_2', 'office_1', 'office_3', 'office_4', 'room_1', 'hotel_0']
            scenes_test = ['apartment_2', 'frl_apartment_3', 'frl_apartment_0', 'office_0', 'room_2', 'room_0']

        # INRAS scenes, for within scene generalization
        elif args['fold'] == 'inras':
            scenes_train = ['frl_apartment_2', 'frl_apartment_4', 'apartment_1', 'apartment_2', 'office_4', 'office_3']
            #scenes_train = ['frl_apartment_2']  # for debugging, single scene
            scenes_test = scenes_train
        elif args['fold'] == 'inras_3scenes':
            scenes_train = ['frl_apartment_4', 'apartment_1', 'office_4']
            #scenes_train = ['frl_apartment_2']  # for debugging, single scene
            scenes_test = scenes_train
        else:
            raise ValueError(f'ERROR: Wrong dataset fold {args["fold"]}')

    elif args['dataset'] == 'mp3d':
        scenes_all = ['pLe4wQe7qrG', 'YFuZgdQ5vWj', 'oLBMNvg9in8', 'cV4RVeZvu5T', 'x8F5xyUWy9e', 's8pcmisQ38h', 'V2XKFyX4ASd', 'EDJbREhghzL', 'pRbA3pwrgk9', '17DRP5sb8fy', 'XcA2TqTSSAj', '29hnd4uzFmX', 'YmJkqBEsHnH', 'D7G3Y4RVNrH', 'i5noydFURQK', 'JF19kD82Mey', 'JmbYfDe2QKZ', 'EU6Fwq7SyZv', 'HxpKQynjfin', 'WYY7iVyf5p8', 'yqstnuAEVhm', 'aayBHfsNo7d', '8194nk5LbLH', 'e9zR4mvMWw7', 'jh4fc5c5qoQ', 'Pm6F8kyY3z2', 'GdvgFV5R1Z5', 'TbHJrupSAjP']
        if args['fold'] == '0':
            train_ids = [25, 14, 7, 27, 16, 21, 11, 15, 1, 23, 17, 26, 12, 5, 8, 18, 2, 22, 0, 20, 19, 24]
            test_ids = [3, 4, 6, 9, 10, 13]
            scenes_train = [scenes_all[id] for id in train_ids]
            scenes_test = [scenes_all[id] for id in test_ids]
        elif args['fold'] == '1':
            train_ids = [10, 25, 8, 1, 24, 4, 2, 15, 0, 19, 9, 16, 12, 5, 7, 3, 6, 22, 20, 27, 23, 11]
            test_ids = [13, 14, 17, 18, 21, 26]
            scenes_train = [scenes_all[id] for id in train_ids]
            scenes_test = [scenes_all[id] for id in test_ids]
        elif args['fold'] == '2':
            train_ids = [10, 8, 17, 4, 3, 21, 12, 2, 20, 6, 11, 9, 25, 18, 13, 24, 15, 0, 1, 5, 26, 27]
            test_ids = [7, 14, 16, 19, 22, 23]
            scenes_train = [scenes_all[id] for id in train_ids]
            scenes_test = [scenes_all[id] for id in test_ids]
        elif args['fold'] == '3':
            train_ids = [24, 20, 18, 14, 8, 17, 16, 4, 1, 15, 5, 6, 26, 11, 22, 2, 25, 13, 21, 3, 9, 19]
            test_ids = [0, 7, 10, 12, 23, 27]
            scenes_train = [scenes_all[id] for id in train_ids]
            scenes_test = [scenes_all[id] for id in test_ids]
        elif args['fold'] == '4':
            train_ids = [20, 4, 21, 14, 1, 6, 27, 5, 15, 8, 24, 7, 13, 19, 12, 18, 2, 16, 25, 17, 3, 22]
            test_ids = [0, 9, 10, 11, 23, 26]
            scenes_train = [scenes_all[id] for id in train_ids]
            scenes_test = [scenes_all[id] for id in test_ids]
        else:
            raise ValueError(f'ERROR: Wrong dataset fold {args["fold"]}')

    elif args['dataset'] == 'mras':
        #scenes_train = ['grid21_materials_2']  # v4
        #scenes_test = ['grid21_materials_2']  # v4

        scenes_with_errors = ['grid99_materials_4', 'grid36_materials_3', 'line1_materials_0', 'line36_materials_3']

        # for v5, with all scenes
        if args['fold'] == 'all':
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            line_list = ["line" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            scenes_train =  grid_list + line_list
        elif args['fold'] == 'all_grids':
            # NOTE: use range(100) for all, or range(start,stop) for a subset
            # This is useful when dumping maps to LMDB, because otherwise it uses too much memory
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            scenes_train =  grid_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'all_lines':
            line_list = ["line" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            scenes_train =  line_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'grids_a':
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(0, 40) for mater in range(5)]
            scenes_train =  grid_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'grids_b':
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(40, 80) for mater in range(5)]
            scenes_train =  grid_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'grids_c':
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(80, 101) for mater in range(5)]
            scenes_train =  grid_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'lines_a':
            line_list = ["line" + str(i) + "_materials_" + str(mater) for i in range(0, 40) for mater in range(5)]
            scenes_train =  line_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'lines_b':
            line_list = ["line" + str(i) + "_materials_" + str(mater) for i in range(40, 80) for mater in range(5)]
            scenes_train =  line_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'lines_c':
            line_list = ["line" + str(i) + "_materials_" + str(mater) for i in range(80, 101) for mater in range(5)]
            scenes_train =  line_list
            scenes_test = scenes_train[0:5]
        elif args['fold'] == 'extra_lines':
            # These are some scenes that had errors in the google Drive
            scenes_train = ['line3_materials_2', 'line3_materials_3', 'line3_materials_4',  
                            'line40_materials_0', 'line40_materials_2', 'line40_materials_4',
                            'line41_materials_0', 'line41_materials_1', 'line41_materials_2', 'line41_materials_4',
                            'line42_materials_1', 'line42_materials_3', 
                            'line43_materials_0', 'line43_materials_3', 'line43_materials_4',
                            'line44_materials_0', 'line44_materials_1', 'line44_materials_4',
                            'line45_materials_0', 'line45_materials_3', 
                            'line48_materials_3']
        elif args['fold'] == 'extra_grids':
            # These are some scenes that had errors in the google Drive
            scenes_train = ['grid50_materials_0', 'grid50_materials_1', 'grid50_materials_3', 'grid50_materials_4',
                            'grid51_materials_0', 'grid51_materials_1',
                            'grid52_materials_0',
                            'grid53_materials_2',] 
                            #'grid85_materials_1', 'grid85_materials_4', ]  # avoid repeated scenes
                            #'grid86_materials_0', 
                            #]
            # grid that I am missing
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(84,100) for mater in range(5)]
            scenes_train = scenes_train + grid_list
        elif args['fold'] == 'fixed_1':
            ids_grids_test = [0, 5, 16, 21, 24, 26, 28, 33, 38, 39, 40, 41, 44, 54, 57, 59, 61, 64, 68, 69, 70, 73, 75, 79, 82, 84, 87, 89, 94, 97]
            ids_lines_test = [3, 5, 11, 18, 19, 23, 24, 25, 27, 31, 33, 34, 36, 38, 45, 49, 50, 53, 60, 62, 67, 70, 74, 75, 80, 82, 83, 91, 92, 93]
            ids_grids_train = [x for x in np.arange(100) if x not in set(ids_grids_test)]
            ids_lines_train = [x for x in np.arange(100) if x not in set(ids_lines_test)]

            grids_train = [f'grid{x}_materials_{y}' for x in ids_grids_train for y in range(5)]
            lines_train = [f'line{x}_materials_{y}' for x in ids_lines_train for y in range(5)]
            scenes_train = grids_train + lines_train

            grids_test = [f'grid{x}_materials_{y}' for x in ids_grids_test for y in range(5)]
            lines_test = [f'line{x}_materials_{y}' for x in ids_lines_test for y in range(5)]
            scenes_test = grids_test + lines_test

            for bad_scene in scenes_with_errors:
                if bad_scene in scenes_train: scenes_train.remove(bad_scene)
                if bad_scene in scenes_test: scenes_test.remove(bad_scene)
#            if 'grid99_materials_4' in scenes_train: scenes_train.remove('grid99_materials_4')  # remove this scene that creates nans during training
        elif args['fold'] == 'fixed_2':
            ids_grids_test = [4, 5, 14, 15, 17, 18, 20, 26, 29, 31, 39, 41, 42, 43, 49, 55, 56, 59, 60, 63, 77, 78, 79, 80, 81, 82, 93, 94, 96, 99]
            ids_lines_test = [2, 3, 6, 8, 10, 12, 15, 26, 30, 31, 36, 37, 38, 43, 45, 48, 49, 54, 55, 57, 58, 64, 65, 74, 78, 79, 80, 84, 93, 96]
            ids_grids_train = [x for x in np.arange(100) if x not in set(ids_grids_test)]
            ids_lines_train = [x for x in np.arange(100) if x not in set(ids_lines_test)]

            grids_train = [f'grid{x}_materials_{y}' for x in ids_grids_train for y in range(5)]
            lines_train = [f'line{x}_materials_{y}' for x in ids_lines_train for y in range(5)]
            scenes_train = grids_train + lines_train

            grids_test = [f'grid{x}_materials_{y}' for x in ids_grids_test for y in range(5)]
            lines_test = [f'line{x}_materials_{y}' for x in ids_lines_test for y in range(5)]
            scenes_test = grids_test + lines_test

            for bad_scene in scenes_with_errors:
                if bad_scene in scenes_train: scenes_train.remove(bad_scene)
                if bad_scene in scenes_test: scenes_test.remove(bad_scene)
        elif args['fold'] == 'fixed_3':
            ids_grids_test = [1, 5, 7, 8, 13, 14, 17, 19, 25, 31, 32, 45, 47, 51, 52, 59, 64, 67, 69, 70, 72, 76, 78, 85, 86, 89, 91, 92, 94, 97]
            ids_lines_test = [2, 3, 5, 10, 11, 13, 14, 17, 21, 29, 31, 35, 36, 38, 45, 48, 50, 53, 54, 55, 63, 65, 69, 71, 74, 79, 82, 86, 94, 99]
            ids_grids_train = [x for x in np.arange(100) if x not in set(ids_grids_test)]
            ids_lines_train = [x for x in np.arange(100) if x not in set(ids_lines_test)]

            grids_train = [f'grid{x}_materials_{y}' for x in ids_grids_train for y in range(5)]
            lines_train = [f'line{x}_materials_{y}' for x in ids_lines_train for y in range(5)]
            scenes_train = grids_train + lines_train

            grids_test = [f'grid{x}_materials_{y}' for x in ids_grids_test for y in range(5)]
            lines_test = [f'line{x}_materials_{y}' for x in ids_lines_test for y in range(5)]
            scenes_test = grids_test + lines_test

            for bad_scene in scenes_with_errors:
                if bad_scene in scenes_train: scenes_train.remove(bad_scene)
                if bad_scene in scenes_test: scenes_test.remove(bad_scene)
        elif args['fold'] == 'debug_grids':
            ids_grids = [78, 31, 14, 17]
            ids_grids = [0, 4, 10, 12, 16, 22, 26, 28, 30, 34, 36, 37, 38, 39]
            ids_grids = [40, 41, 42, 43, 44, 46, 48, 49, 53, 55, 58, 60, 61, 62, 63, 65, 66, 68, 71, 73, 74, 75, 77]
            ids_grids = [40]
            scenes_train = [f'grid{x}_materials_{y}' for x in ids_grids for y in range(5)]
            scenes_test = scenes_train
        elif args['fold'] == 'debug_lines':
            ids_lines = [65, 2, 10, 79, 48, 54, 55]
            scenes_train = [f'line{x}_materials_{y}' for x in ids_lines for y in range(5)]
            scenes_test = scenes_train
        elif args['fold'] == 'debug_nans':
            tmp = ['grid51_materials_4/2_27', 'line87_materials_2/8_2', 'grid88_materials_3/1_263', 
            'grid95_materials_1/22_310', 'grid46_materials_0/18_115', 'grid95_materials_4/11_398', 'line94_materials_2/5_212', 
            'grid34_materials_0/17_80', 'grid99_materials_4/13_412', 'grid27_materials_1/0_162', 'grid36_materials_1/5_409', 
            'grid2_materials_4/2_384', 'grid31_materials_3/3_36', 'grid99_materials_1/8_111', 'grid62_materials_1/17_4', 
            'grid4_materials_1/0_28', 'grid56_materials_1/2_50', 'grid85_materials_3/6_108', 'grid10_materials_1/0_226', 
            'grid42_materials_2/3_53', 'line35_materials_2/0_195', 'line86_materials_3/2_74', 'line6_materials_2/5_119', 
            'grid34_materials_4/6_310', 'line63_materials_4/8_91', 'grid77_materials_1/3_250', 'grid67_materials_1/3_136', 
            'line44_materials_3/5_148', 'grid95_materials_4/18_140', 'line8_materials_2/3_243', 'line63_materials_0/3_94', 
            'line85_materials_2/10_146', 'grid98_materials_2/4_284', 'grid55_materials_4/2_233', 'grid47_materials_1/15_51', 
            'grid65_materials_3/15_156', 'line95_materials_4/7_52', 'grid76_materials_1/4_44', 'line48_materials_2/1_170', 
            'grid50_materials_2/14_57', 'grid14_materials_3/7_23', 'line58_materials_2/7_210', 'grid27_materials_2/11_163', 
            'grid20_materials_2/11_196', 'grid55_materials_4/8_380', 'grid18_materials_2/14_191', 'line56_materials_3/7_176', 
            'grid23_materials_3/0_226', 'line71_materials_0/2_117', 'grid88_materials_4/15_141', 'grid4_materials_2/11_189', 
            'grid80_materials_4/14_148', 'grid85_materials_3/21_153', 'grid15_materials_2/1_56', 'grid95_materials_0/14_381', 
            'grid98_materials_0/1_279', 'grid67_materials_3/7_63', 'grid17_materials_2/4_268', 'line64_materials_0/3_92', 
            'grid53_materials_3/12_87', 'line6_materials_3/2_32', 'line13_materials_3/0_13', 'line44_materials_1/4_304', 
            'grid95_materials_1/6_417', 'grid2_materials_2/6_364', 'grid90_materials_0/6_181', 'grid46_materials_0/16_121', 
            'grid95_materials_4/7_234', 'grid42_materials_0/16_138', 'grid42_materials_2/17_39', 'grid91_materials_4/5_10', 
            'grid55_materials_2/15_279', 'grid2_materials_4/14_309', 'grid49_materials_3/6_304', 'grid46_materials_4/7_371', 
            'line79_materials_3/8_72', 'grid62_materials_4/0_429', 'grid51_materials_0/4_171', 'grid62_materials_2/13_162', 
            'grid92_materials_3/8_56', 'grid10_materials_3/0_149', 'grid34_materials_2/13_1', 'grid6_materials_3/5_35', 
            'grid49_materials_2/15_63', 'grid99_materials_4/9_173', 'grid80_materials_3/8_278', 'line54_materials_3/1_103', 
            'grid62_materials_3/4_10', 'line81_materials_2/0_118', 'grid27_materials_0/0_207', 'line86_materials_3/10_268', 
            'grid29_materials_3/5_166', 'grid22_materials_0/0_144', 'grid13_materials_4/6_76', 'grid34_materials_2/15_207', 
            'grid74_materials_2/0_2', 'grid72_materials_2/4_357', 'line6_materials_2/3_205', 'line13_materials_0/4_138', 
            'line72_materials_3/8_230', 'line63_materials_2/10_155', 'line6_materials_4/8_138', 'line61_materials_2/9_239', 
            'grid58_materials_2/12_146', 'grid29_materials_4/1_205', 'grid67_materials_1/4_329', 'grid2_materials_2/13_42', 
            'grid25_materials_1/7_69', 'grid36_materials_2/4_239', 'grid56_materials_1/5_265', 'grid42_materials_2/5_206', 
            'line22_materials_0/1_270', 'grid58_materials_2/3_399', 'line88_materials_1/5_3', 'grid85_materials_1/11_163', 
            'grid74_materials_4/1_224', 'grid46_materials_0/12_305', 'line21_materials_1/7_96', 'grid53_materials_0/9_43', 
            'grid13_materials_3/15_128', 'grid12_materials_2/1_61', 'line99_materials_1/5_203', 'grid22_materials_2/4_318', 
            'line48_materials_1/9_186', 'grid3_materials_4/16_316', 'grid30_materials_4/18_41', 'grid62_materials_0/11_212', 
            'line71_materials_1/8_249']

            tmp = ['grid71_materials_0/6_11', 'grid61_materials_3/25_195', 'grid24_materials_0/2_153', 'line84_materials_1/2_115', 'line4_materials_1/8_17', 'line58_materials_4/3_214', 'line30_materials_1/6_4', 'line78_materials_2/4_96', 'line27_materials_0/8_144', 'grid23_materials_3/13_88', 'grid95_materials_2/22_228', 'line90_materials_4/2_138', 'line42_materials_3/2_67', 'grid41_materials_0/15_219', 'grid95_materials_4/21_360', 'grid30_materials_1/24_316', 'line22_materials_1/8_50', 'line81_materials_3/2_24', 'line64_materials_1/2_118', 'line92_materials_4/2_75', 'line66_materials_0/0_105', 'grid71_materials_1/3_85', 'line40_materials_2/11_99', 'grid23_materials_0/6_380', 'line59_materials_1/7_86', 'line95_materials_3/3_207', 'grid46_materials_4/22_117', 'grid10_materials_4/0_24', 'grid21_materials_0/7_126', 'grid28_materials_3/6_228', 'line61_materials_0/6_124', 'grid36_materials_1/4_130', 'grid2_materials_2/17_148', 'grid61_materials_1/2_344', 'grid0_materials_0/0_212', 'grid66_materials_3/9_278', 'grid10_materials_4/1_321', 'grid68_materials_0/12_166', 'grid50_materials_1/0_234', 'grid53_materials_4/14_73', 'line39_materials_4/1_26', 'line44_materials_4/9_34', 'grid54_materials_0/2_120', 'line75_materials_0/7_152', 'grid0_materials_2/9_211', 'line70_materials_4/1_49', 'grid87_materials_2/16_127', 'grid44_materials_3/8_19', 'grid48_materials_0/7_500', 'line30_materials_3/0_170', 'line37_materials_3/2_256', 'line19_materials_4/0_223', 'grid54_materials_0/10_49', 'line30_materials_3/9_114', 'grid23_materials_2/2_117', 'grid57_materials_4/7_144', 'grid40_materials_0/8_60', 'line72_materials_4/4_168', 'grid95_materials_4/7_34', 'grid61_materials_4/4_243', 'grid23_materials_4/0_370', 'grid84_materials_2/17_158', 'line49_materials_1/8_49', 'line20_materials_1/1_12', 'grid28_materials_1/7_168', 'grid42_materials_2/5_218', 'line78_materials_4/8_214', 'grid88_materials_2/4_368', 'line25_materials_3/6_66', 'grid71_materials_1/17_23', 'grid99_materials_1/11_379', 'grid37_materials_2/3_27', 'grid22_materials_0/3_356', 'grid2_materials_1/10_4', 'grid21_materials_2/6_230', 'grid84_materials_4/9_177', 'line64_materials_1/3_101', 'line25_materials_1/2_155', 'line25_materials_4/2_60', 'line77_materials_1/6_13', 'grid65_materials_0/26_294', 'line0_materials_1/3_120', 'grid55_materials_3/9_245', 'grid61_materials_3/14_379', 'line64_materials_2/1_28', 'line12_materials_2/2_1', 'grid90_materials_4/1_183', 'grid11_materials_3/8_196', 'line93_materials_1/8_52', 'line83_materials_1/4_15', 'grid99_materials_0/14_25', 'grid0_materials_1/5_275', 'grid24_materials_0/16_320', 'line62_materials_2/6_26', 'grid2_materials_4/5_326', 'line64_materials_2/9_78', 'grid38_materials_2/8_326', 'line64_materials_0/1_28', 'line43_materials_4/4_158', 'grid44_materials_1/13_155', 'grid10_materials_2/19_157', 'grid49_materials_1/9_55', 'grid24_materials_4/9_182', 'grid80_materials_0/15_153', 'grid84_materials_3/9_232', 'grid0_materials_2/26_113', 'line44_materials_3/10_80', 'grid60_materials_3/2_98', 'line83_materials_4/4_27', 'grid44_materials_0/3_435', 'line81_materials_4/2_244', 'line77_materials_0/6_107', 'line19_materials_4/0_76', 'grid10_materials_4/10_309', 'grid84_materials_2/3_207', 'grid87_materials_1/9_143', 'line20_materials_4/5_153', 'line40_materials_2/7_182', 'line64_materials_1/8_43', 'grid2_materials_4/6_277', 'grid66_materials_0/9_99', 'line27_materials_1/13_220', 'line70_materials_4/5_100', 'grid36_materials_2/24_164', 'grid16_materials_4/7_205', 'grid46_materials_1/11_240', 'grid29_materials_2/13_104', 'grid4_materials_2/2_235']
            tmp = ['grid6_materials_4/3_247', 'grid31_materials_0/1_199', 'grid58_materials_1/4_263', 'grid36_materials_4/25_180', 'grid80_materials_2/15_442', 'grid99_materials_2/10_460', 'grid53_materials_2/17_290', 'grid2_materials_2/10_40', 'grid98_materials_0/2_145', 'grid18_materials_3/2_181', 'grid53_materials_3/10_9', 'grid31_materials_3/16_194', 'grid49_materials_0/14_339', 'grid62_materials_0/0_412', 'grid47_materials_4/15_207', 'grid76_materials_0/7_98', 'line63_materials_3/3_77', 'line42_materials_4/13_116', 'line85_materials_3/2_166', 'grid25_materials_3/7_87', 'line72_materials_0/1_41', 'line12_materials_2/6_101', 'grid80_materials_0/7_92', 'grid4_materials_3/9_70', 'grid36_materials_0/21_224', 'line51_materials_0/5_176', 'grid80_materials_4/16_18', 'grid96_materials_1/4_200', 'grid72_materials_2/15_7', 'line6_materials_2/4_106', 'line1_materials_0/0_16', 'grid22_materials_3/4_248', 'grid18_materials_0/16_214', 'grid72_materials_2/17_346', 'grid36_materials_2/4_199', 'grid63_materials_2/16_28', 'grid71_materials_4/7_313', 'line16_materials_2/0_78', 'grid23_materials_3/17_222', 'grid32_materials_3/10_412', 'grid52_materials_1/1_78', 'grid55_materials_0/16_186', 'grid81_materials_2/5_61', 'grid8_materials_4/1_146', 'grid30_materials_1/21_222', 'grid32_materials_1/1_289', 'grid36_materials_2/17_290', 'grid23_materials_2/2_108', 'line63_materials_1/9_95', 'grid81_materials_3/12_179', 'line99_materials_1/2_242', 'line8_materials_3/8_178', 'grid20_materials_1/8_145', 'grid3_materials_1/19_144', 'line30_materials_3/12_185', 'grid80_materials_2/10_108', 'grid4_materials_2/4_188', 'grid29_materials_1/17_117', 'line65_materials_2/4_137', 'line95_materials_4/8_148', 'grid29_materials_4/11_87', 'grid90_materials_1/6_84', 'grid52_materials_2/9_11', 'line12_materials_3/0_22', 'grid23_materials_3/14_68', 'grid34_materials_0/2_388', 'line2_materials_1/7_216', 'line81_materials_0/3_26', 'grid31_materials_4/24_163', 'grid85_materials_0/8_63', 'line2_materials_3/4_51', 'line73_materials_1/8_188', 'grid65_materials_0/11_337', 'grid46_materials_1/5_161', 'line12_materials_2/0_116', 'grid13_materials_3/0_232', 'grid10_materials_0/15_228', 'grid78_materials_3/4_262', 'grid66_materials_3/8_79', 'grid30_materials_2/3_63', 'line43_materials_1/10_114', 'grid45_materials_1/14_287', 'grid99_materials_1/15_416', 'grid3_materials_4/0_297', 'grid31_materials_2/25_164', 'line22_materials_3/6_249', 'line90_materials_0/6_147', 'grid91_materials_0/1_205', 'grid36_materials_3/2_282', 'grid88_materials_2/10_79', 'grid10_materials_1/13_194', 'grid50_materials_1/1_293', 'grid10_materials_0/10_258', 'grid80_materials_2/12_99', 'grid60_materials_2/7_1', 'grid31_materials_1/8_111', 'line79_materials_4/5_313', 'grid36_materials_1/0_394', 'line68_materials_4/2_167', 'line37_materials_1/7_4', 'line55_materials_1/4_55', 'grid30_materials_1/25_25', 'line29_materials_3/8_139', 'line65_materials_3/4_198', 'grid31_materials_3/16_97', 'line79_materials_0/10_231', 'grid9_materials_4/8_145', 'line6_materials_1/9_212', 'grid83_materials_4/4_138', 'grid62_materials_4/9_415', 'grid51_materials_0/7_112', 'grid3_materials_1/3_45', 'grid25_materials_4/16_134', 'grid72_materials_2/11_270', 'grid12_materials_1/5_277', 'line41_materials_4/4_175', 'grid2_materials_1/5_223', 'line63_materials_4/0_70', 'line94_materials_1/2_218', 'grid3_materials_1/6_306', 'line43_materials_2/0_129', 'grid12_materials_1/7_156', 'line57_materials_3/2_59', 'grid88_materials_1/11_106', 'line14_materials_4/6_71', 'grid78_materials_2/6_296', 'line26_materials_3/2_112', 'grid25_materials_1/1_152']

            scenes_train = []
            #tmp = ['grid34_materials_0/1']
            #tmp = ['grid99_materials_4/1']  # this is the one that creates nans, but the GT acumaps are ok
            for i in tmp:
                a, b = i.split('/')
                if a == 'grid99_materials_4': continue
                if a == 'grid36_materials_3': continue
                if a == 'line1_materials_0': continue
                scenes_train.append(a)
            scenes_train = list(set(scenes_train))  # remove duplicates
            scenes_train = scenes_train[0:128]  # 7:10
            scenes_test = scenes_train[0:1]
        else:
            raise ValueError(f'ERROR: Wrong dataset fold {args["fold"]}')

    if False:
        # Fold 00
        scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'room_2', 'room_0', 'frl_apartment_1', 'apartment_1', 'office_3', 'frl_apartment_0', 'office_2', 'hotel_0']
        scenes_test = ['room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
    if False:
        # Fold 0 (same as fold 00, but with different order and 1 extra scene in train)
        scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'frl_apartment_0', 'frl_apartment_1', 'apartment_1', 'room_2', 'room_0', 'office_3', 'office_2', 'office_4', 'hotel_0']
        scenes_test = ['room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']

    if False:
        # Fold 00 plus apartment 0 , because now we are processing multi story scenes
        scenes_train = ['apartment_0', 'frl_apartment_3', 'frl_apartment_5', 'frl_apartment_0', 'frl_apartment_1', 'apartment_1', 'room_2', 'room_0', 'office_3', 'office_2', 'office_4', 'hotel_0']
        scenes_test = ['room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
    
    if False:
        # Fold 1, similar to fold 0, but not the same
        scenes_train = ['frl_apartment_2', 'frl_apartment_4', 'room_1', 'room_0', 'frl_apartment_1', 'apartment_2', 'frl_apartment_0', 'office_0', 'office_1', 'office_3', 'office_4', 'hotel_0']
        scenes_test = ['room_2', 'frl_apartment_3', 'apartment_1', 'office_2', 'frl_apartment_5']
        
        # Fold 2, test apartments and rooms
        scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_1', 'frl_apartment_0', 'office_0', 'office_2', 'office_1', 'office_3', 'office_4', 'hotel_0']
        scenes_test = [ 'room_0', 'room_1', 'room_2', 'apartment_1', 'apartment_2']
    
        # Fold 3, test frl aprtment
        scenes_train = ['room_0', 'room_1', 'room_2', 'apartment_1', 'apartment_2', 'office_0', 'office_2', 'office_1', 'office_3', 'office_4', 'hotel_0']
        scenes_test = [ 'frl_apartment_3', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_1', 'frl_apartment_0' ]
    
        # Fold 4, test offices and hotel
        scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_1', 'frl_apartment_0', 'room_0', 'room_1', 'room_2', 'apartment_1', 'apartment_2',]
        scenes_test = ['office_0', 'office_2', 'office_1', 'office_3', 'office_4', 'hotel_0']

    if False:
        scenes_all = ['pLe4wQe7qrG', 'YFuZgdQ5vWj', 'oLBMNvg9in8', 'cV4RVeZvu5T', 'x8F5xyUWy9e', 's8pcmisQ38h', 'V2XKFyX4ASd', 'EDJbREhghzL', 'pRbA3pwrgk9', '17DRP5sb8fy', 'XcA2TqTSSAj', '29hnd4uzFmX', 'YmJkqBEsHnH', 'D7G3Y4RVNrH', 'i5noydFURQK', 'JF19kD82Mey', 'JmbYfDe2QKZ', 'EU6Fwq7SyZv', 'HxpKQynjfin', 'WYY7iVyf5p8', 'yqstnuAEVhm', 'aayBHfsNo7d', '8194nk5LbLH', 'e9zR4mvMWw7', 'jh4fc5c5qoQ', 'Pm6F8kyY3z2', 'GdvgFV5R1Z5', 'TbHJrupSAjP']
        # Fold 00, mp3d
        train_ids = [25, 14, 7, 27, 16, 21, 11, 15, 1, 23, 17, 26, 12, 5, 8, 18, 2, 22, 0, 20, 19, 24]
        test_ids = [3, 4, 6, 9, 10, 13]
        scenes_train = [scenes_all[id] for id in train_ids]
        scenes_test = [scenes_all[id] for id in test_ids]

    if False:
        # Fold 01, mp3d
        train_ids = [10, 25, 8, 1, 24, 4, 2, 15, 0, 19, 9, 16, 12, 5, 7, 3, 6, 22, 20, 27, 23, 11]
        test_ids = [13, 14, 17, 18, 21, 26]
        scenes_train = [scenes_all[id] for id in train_ids]
        scenes_test = [scenes_all[id] for id in test_ids]

        # Fold 02, mp3d
        train_ids = [10, 8, 17, 4, 3, 21, 12, 2, 20, 6, 11, 9, 25, 18, 13, 24, 15, 0, 1, 5, 26, 27]
        test_ids = [7, 14, 16, 19, 22, 23]
        scenes_train = [scenes_all[id] for id in train_ids]
        scenes_test = [scenes_all[id] for id in test_ids]

        # Fold 03, mp3d
        train_ids = [24, 20, 18, 14, 8, 17, 16, 4, 1, 15, 5, 6, 26, 11, 22, 2, 25, 13, 21, 3, 9, 19]
        test_ids = [0, 7, 10, 12, 23, 27]
        scenes_train = [scenes_all[id] for id in train_ids]
        scenes_test = [scenes_all[id] for id in test_ids]

        # Fold 04, mp3d
        train_ids = [20, 4, 21, 14, 1, 6, 27, 5, 15, 8, 24, 7, 13, 19, 12, 18, 2, 16, 25, 17, 3, 22]
        test_ids = [0, 9, 10, 11, 23, 26]
        scenes_train = [scenes_all[id] for id in train_ids]
        scenes_test = [scenes_all[id] for id in test_ids]

    # For DGX only
    if args['use_dgx']:
        params['datapath'] = '/mnt/audio/home/ricfalcon/data/soundspaces/data/'
        directory_jsons = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/mnt/audio/home/ricfalcon/data/replica/{:s}/mesh.ply'
        directory_geometry_matterport ='/mnt/audio/home/ricfalcon/data/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/outputs/'  # scene_name
    elif args['use_triton']:
        params['datapath'] = '/scratch/cs/sequentialml/datasets/soundspaces/data/'
        directory_jsons = '/scratch/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/scratch/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport ='/scratch/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/scratch/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/scratch/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/scratch/cs/sequentialml/datasets/multiroom3/{:s}/outputs/'  # scene_name
    elif args['use_vrgpu']:
        params['datapath'] = '/m/triton/cs/sequentialml/datasets/soundspaces/data'
        directory_jsons = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport ='/m/triton/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/m/triton/cs/sequentialml/datasets/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/m/triton/cs/sequentialml/datasets/multiroom3/{:s}/outputs/'  # scene_name
    else:
        directory_jsons = '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
        directory_geometry = '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
        directory_lmdb = '/home/ricfalcon/00data/soundspaces_processed/lmdb/'
    
    if dataset_name == 'soundspaces':
        dset_train = SoundspacesDataset(datapath=params['datapath'],
                                        directory_geometry=directory_geometry,
                                        directory_geometry_matterport=directory_geometry_matterport,
                                        directory_geometry_mras=directory_geometry_mras,
                                        directory_jsons=directory_jsons,
                                        directory_jsons_matterport=directory_jsons_matterport,
                                        directory_jsons_mras=directory_jsons_mras,
                                        directory_lmdb=directory_lmdb,
                                        directory_lmdb_maps=directory_lmdb_maps,
                                        directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                        directory_rir_mras=directory_rir_mras,
                                        fname_lmdb=params['fname_lmdb'],
                                        fname_lmdb_maps=params['fname_lmdb_maps'],
                                        scenes=scenes_train,
                                        n_files_per_scene=args['n_files_per_scene'],
                                        read_rirs=True,
                                        read_scenes=True,
                                        read_floormaps=True,
                                        read_acumaps=True,
                                        read_lmdb=params['read_lmdb'],
                                        read_lmdb_maps=params['read_lmdb_maps'],
                                        max_length=params['max_length'],
                                        floormap_processor=floormap_processor,
                                        acumap_processor=acumap_processor,
                                        rir_output_channels=params['rir_output_channels'],
                                        multi_story_removal=True,
                                        return_rot_angle=args['fmap_add_pose'])
        dset_test = SoundspacesDataset(datapath=params['datapath'],
                                       directory_geometry=directory_geometry,
                                       directory_geometry_matterport=directory_geometry_matterport,
                                       directory_geometry_mras=directory_geometry_mras,
                                       directory_jsons=directory_jsons,
                                       directory_jsons_matterport=directory_jsons_matterport,
                                       directory_jsons_mras=directory_jsons_mras,
                                       directory_lmdb=directory_lmdb,
                                       directory_lmdb_maps=directory_lmdb_maps,
                                       directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                       directory_rir_mras=directory_rir_mras,
                                       fname_lmdb=params['fname_lmdb'],
                                       fname_lmdb_maps=params['fname_lmdb_maps'],
                                       scenes=scenes_test,
                                       n_files_per_scene=args['n_files_per_scene'],
                                       read_rirs=True,
                                       read_scenes=True,
                                       read_floormaps=True,
                                       read_acumaps=True,
                                       read_lmdb=params['read_lmdb'],
                                       read_lmdb_maps=params['read_lmdb_maps'],
                                       max_length=params['max_length'],
                                       floormap_processor=floormap_processor,
                                       acumap_processor=acumap_processor,
                                       rir_output_channels=params['rir_output_channels'],
                                       multi_story_removal=True,
                                       return_rot_angle=args['fmap_add_pose'])

        if args['do_baseline_spatial_force_omni']:
            # This is a special case, when I want to use the acumaps from the omni dataset,
            # but applied to the spatial targets
            # So we need to load another dataset.
            # I also have to fix the channels, because the spatial case now has only C50, at 5 diretions, 3 freq bands
            # So I need to take the C50 of the main maps and replicate accordingly
            dset_test_omni = SoundspacesDataset(datapath=params['datapath'],
                                       directory_geometry=directory_geometry,
                                       directory_geometry_matterport=directory_geometry_matterport,
                                       directory_geometry_mras=directory_geometry_mras,
                                       directory_jsons=directory_jsons,
                                       directory_jsons_matterport=directory_jsons_matterport,
                                       directory_jsons_mras=directory_jsons_mras,
                                       directory_lmdb=directory_lmdb,
                                       directory_lmdb_maps=directory_lmdb_maps,
                                       directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                       directory_rir_mras=directory_rir_mras,
                                       fname_lmdb=params['fname_lmdb'],
                                       fname_lmdb_maps='maps_relcenter_10x10.lmdb',
                                       scenes=scenes_test,
                                       n_files_per_scene=args['n_files_per_scene'],
                                       read_rirs=False,
                                       read_scenes=True,
                                       read_floormaps=True,
                                       read_acumaps=True,
                                       read_lmdb=params['read_lmdb'],
                                       read_lmdb_maps=params['read_lmdb_maps'],
                                       max_length=params['max_length'],
                                       floormap_processor=floormap_processor,
                                       acumap_processor=acumap_processor,
                                       rir_output_channels=[0],
                                       multi_story_removal=True,
                                       return_rot_angle=False,
                                       avoid_storing_maps_locally=True)


    else:
        raise ValueError(f'ERROR: Unrecognized dataset {dataset_name}')
    
    if False and not args['use_dgx']:  # disabled for DGX
        dset_train = prune_dataset_by_src_and_scene(dset_train)
        dset_test = prune_dataset_by_src_and_scene(dset_test)
        print(dset_train)
        print(dset_test)

    if args['do_baseline_spatial_force_omni']:
        return dset_train, dset_test, dset_test_omni
    else:
        return dset_train, dset_test

def get_dataset_selected_examples(dataset_name: str, params: Dict, floormap_processor: object, acumap_processor: object, args: object) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """ This returns a single scene, source. Mostly just to run inference and get the figures for the paper that 
    show some examples."""

    # Train datasets are needed for some baselines (mostly, RIR based baselines)
    # NOTE: I removed the aparmtnet_0 from training sets, because these ar enot ready in some pre computed maps (e.g. spatial case)
    if args['dataset'] == 'replica':
        if args['fold'] == 'test_1':
            scenes_train = ['apartment_1', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_1', 'frl_apartment_4', 'office_2', 'office_1', 'office_3', 'office_4', 'room_1', 'hotel_0']
            scenes_test = ['apartment_2']  # choose from ['apartment_2', 'frl_apartment_3', 'frl_apartment_0', 'office_0', 'room_2', 'room_0']
        elif args['fold'] == 'test_2':
            scenes_train = ['apartment_1', 'frl_apartment_4', 'frl_apartment_2', 'frl_apartment_5', 'frl_apartment_1', 'office_1', 'office_0', 'room_0', 'office_4', 'room_1', 'hotel_0']
            scenes_test = ['apartment_2']  # ['apartment_2', 'frl_apartment_0', 'frl_apartment_3', 'office_2', 'office_3', 'room_2']
        elif args['fold'] == 'test_3':  # this one is not that good, because apartment 0 is in test
            scenes_train = ['apartment_1', 'apartment_2', 'frl_apartment_5', 'frl_apartment_2', 'frl_apartment_0', 'frl_apartment_4', 'office_2', 'office_0', 'room_0', 'room_2', 'room_1', 'hotel_0']
            scenes_test = ['apartment_0']  # ['apartment_0', 'frl_apartment_1', 'frl_apartment_3', 'office_1', 'office_3', 'office_4']
        elif args['fold'] == 'test_4':
            scenes_train = ['apartment_2', 'frl_apartment_3', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_5', 'office_0', 'office_2', 'room_1', 'office_3', 'room_2', 'room_0']
            scenes_test = ['office_1']  # ['apartment_1', 'frl_apartment_4', 'frl_apartment_2', 'office_1', 'office_4', 'hotel_0']
            scenes_train = ['apartment_1']  # NOTE: The code is messy. So for acumap baselines, I need to use train scenes as test scenes
        else:
            raise ValueError(f'ERROR: Wrong dataset fold {args["fold"]}')

    elif args['dataset'] == 'mp3d':
        raise ValueError(f'ERROR: Wrong dataset fold {args["fold"]}')

    elif args['dataset'] == 'mras':
        scenes_with_errors = ['grid99_materials_4', 'grid36_materials_3', 'line1_materials_0', 'line36_materials_3']

        # for v5, with all scenes
        if args['fold'] == 'fixed_1':
            ids_grids_test = [0, 5, 16, 21, 24, 26, 28, 33, 38, 39, 40, 41, 44, 54, 57, 59, 61, 64, 68, 69, 70, 73, 75, 79, 82, 84, 87, 89, 94, 97]
            ids_lines_test = [3, 5, 11, 18, 19, 23, 24, 25, 27, 31, 33, 34, 36, 38, 45, 49, 50, 53, 60, 62, 67, 70, 74, 75, 80, 82, 83, 91, 92, 93]
            ids_grids_train = [] # 1 or 2
            ids_lines_train = [40] # 1   # fo acumap bases baseline this should match the test
            assert ids_grids_train not in ids_grids_test, "Wrong ids"
            assert ids_lines_train not in ids_lines_test, "Wrong ids"
            ids_grids_test = []  # Manually select this to get nice plots, 1 works ok (overfitted), or 2
            ids_lines_test = [40]
            materials_id = [1]

            grids_train = [f'grid{x}_materials_{y}' for x in ids_grids_train for y in materials_id]
            lines_train = [f'line{x}_materials_{y}' for x in ids_lines_train for y in materials_id]
            scenes_train = grids_train + lines_train

            grids_test = [f'grid{x}_materials_{y}' for x in ids_grids_test for y in materials_id]
            lines_test = [f'line{x}_materials_{y}' for x in ids_lines_test for y in materials_id]
            scenes_test = grids_test + lines_test

            for bad_scene in scenes_with_errors:
                if bad_scene in scenes_train: scenes_train.remove(bad_scene)
                if bad_scene in scenes_test: scenes_test.remove(bad_scene)
        elif args['fold'] == 'fixed_2':
            ids_grids_test = [4, 5, 14, 15, 17, 18, 20, 26, 29, 31, 39, 41, 42, 43, 49, 55, 56, 59, 60, 63, 77, 78, 79, 80, 81, 82, 93, 94, 96, 99]
            ids_lines_test = [2, 3, 6, 8, 10, 12, 15, 26, 30, 31, 36, 37, 38, 43, 45, 48, 49, 54, 55, 57, 58, 64, 65, 74, 78, 79, 80, 84, 93, 96]
            ids_grids_train = [18]
            ids_lines_train = []
            assert ids_grids_train not in ids_grids_test, "Wrong ids"
            assert ids_lines_train not in ids_lines_test, "Wrong ids"
            ids_grids_test = [18]  # Manually select this to get nice plots
            ids_lines_test = []
            materials_id = [1]

            grids_train = [f'grid{x}_materials_{y}' for x in ids_grids_train for y in materials_id]
            lines_train = [f'line{x}_materials_{y}' for x in ids_lines_train for y in materials_id]
            scenes_train = grids_train + lines_train

            grids_test = [f'grid{x}_materials_{y}' for x in ids_grids_test for y in materials_id]
            lines_test = [f'line{x}_materials_{y}' for x in ids_lines_test for y in materials_id]
            scenes_test = grids_test + lines_test

            for bad_scene in scenes_with_errors:
                if bad_scene in scenes_train: scenes_train.remove(bad_scene)
                if bad_scene in scenes_test: scenes_test.remove(bad_scene)
        elif args['fold'] == 'fixed_3':
            ids_grids_test = [1, 5, 7, 8, 13, 14, 17, 19, 25, 31, 32, 45, 47, 51, 52, 59, 64, 67, 69, 70, 72, 76, 78, 85, 86, 89, 91, 92, 94, 97]
            ids_lines_test = [2, 3, 5, 10, 11, 13, 14, 17, 21, 29, 31, 35, 36, 38, 45, 48, 50, 53, 54, 55, 63, 65, 69, 71, 74, 79, 82, 86, 94, 99]
            ids_grids_train = [0]
            ids_lines_train = [0]
            assert ids_grids_train not in ids_grids_test, "Wrong ids"
            assert ids_lines_train not in ids_lines_test, "Wrong ids"
            ids_grids_test = [1]  # Manually select this to get nice plots
            ids_lines_test = [2]

            grids_train = [f'grid{x}_materials_{y}' for x in ids_grids_train for y in range(5)]
            lines_train = [f'line{x}_materials_{y}' for x in ids_lines_train for y in range(5)]
            scenes_train = grids_train + lines_train

            grids_test = [f'grid{x}_materials_{y}' for x in ids_grids_test for y in range(5)]
            lines_test = [f'line{x}_materials_{y}' for x in ids_lines_test for y in range(5)]
            scenes_test = grids_test + lines_test

            for bad_scene in scenes_with_errors:
                if bad_scene in scenes_train: scenes_train.remove(bad_scene)
                if bad_scene in scenes_test: scenes_test.remove(bad_scene)
        else:
            raise ValueError(f'ERROR: Wrong dataset fold {args["fold"]}')

    # For DGX only
    if args['use_dgx']:
        params['datapath'] = '/mnt/audio/home/ricfalcon/data/soundspaces/data/'
        directory_jsons = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/mnt/audio/home/ricfalcon/data/replica/{:s}/mesh.ply'
        directory_geometry_matterport ='/mnt/audio/home/ricfalcon/data/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/mnt/audio/home/ricfalcon/data/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/outputs/'  # scene_name
    elif args['use_triton']:
        params['datapath'] = '/scratch/cs/sequentialml/datasets/soundspaces/data/'
        directory_jsons = '/scratch/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/scratch/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport ='/scratch/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/scratch/cs/sequentialml/datasets/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/scratch/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/scratch/cs/sequentialml/datasets/multiroom3/{:s}/outputs/'  # scene_name
    elif args['use_vrgpu']:
        params['datapath'] = '/m/triton/cs/sequentialml/datasets/soundspaces/data'
        directory_jsons = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport ='/m/triton/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/m/triton/cs/sequentialml/datasets/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/m/triton/cs/sequentialml/datasets/multiroom3/{:s}/outputs/'  # scene_name
    else:
        directory_jsons = '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
        directory_geometry = '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
        directory_lmdb = '/home/ricfalcon/00data/soundspaces_processed/lmdb/'
    
    if dataset_name == 'soundspaces':
        dset_train = SoundspacesDataset(datapath=params['datapath'],
                                        directory_geometry=directory_geometry,
                                        directory_geometry_matterport=directory_geometry_matterport,
                                        directory_geometry_mras=directory_geometry_mras,
                                        directory_jsons=directory_jsons,
                                        directory_jsons_matterport=directory_jsons_matterport,
                                        directory_jsons_mras=directory_jsons_mras,
                                        directory_lmdb=directory_lmdb,
                                        directory_lmdb_maps=directory_lmdb_maps,
                                        directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                        directory_rir_mras=directory_rir_mras,
                                        fname_lmdb=params['fname_lmdb'],
                                        fname_lmdb_maps=params['fname_lmdb_maps'],
                                        scenes=scenes_train,
                                        n_files_per_scene=args['n_files_per_scene'],
                                        read_rirs=True,
                                        read_scenes=True,
                                        read_floormaps=True,
                                        read_acumaps=True,
                                        read_lmdb=params['read_lmdb'],
                                        read_lmdb_maps=params['read_lmdb_maps'],
                                        max_length=params['max_length'],
                                        floormap_processor=floormap_processor,
                                        acumap_processor=acumap_processor,
                                        rir_output_channels=params['rir_output_channels'],
                                        multi_story_removal=True,
                                        return_rot_angle=args['fmap_add_pose'])
        dset_test = SoundspacesDataset(datapath=params['datapath'],
                                       directory_geometry=directory_geometry,
                                       directory_geometry_matterport=directory_geometry_matterport,
                                       directory_geometry_mras=directory_geometry_mras,
                                       directory_jsons=directory_jsons,
                                       directory_jsons_matterport=directory_jsons_matterport,
                                       directory_jsons_mras=directory_jsons_mras,
                                       directory_lmdb=directory_lmdb,
                                       directory_lmdb_maps=directory_lmdb_maps,
                                       directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                       directory_rir_mras=directory_rir_mras,
                                       fname_lmdb=params['fname_lmdb'],
                                       fname_lmdb_maps=params['fname_lmdb_maps'],
                                       scenes=scenes_test,
                                       n_files_per_scene=args['n_files_per_scene'],
                                       #n_files_per_scene=2000,  # pnly when debugging the inference only
                                       read_rirs=True,
                                       read_scenes=True,
                                       read_floormaps=True,
                                       read_acumaps=True,
                                       read_lmdb=params['read_lmdb'],
                                       read_lmdb_maps=params['read_lmdb_maps'],
                                       max_length=params['max_length'],
                                       floormap_processor=floormap_processor,
                                       acumap_processor=acumap_processor,
                                       rir_output_channels=params['rir_output_channels'],
                                       multi_story_removal=True,
                                       return_rot_angle=args['fmap_add_pose'])

        if args['do_baseline_spatial_force_omni']:
            # This is a special case, when I want to use the acumaps from the omni dataset,
            # but applied to the spatial targets
            # So we need to load another dataset.
            # I also have to fix the channels, because the spatial case now has only C50, at 5 diretions, 3 freq bands
            # So I need to take the C50 of the main maps and replicate accordingly
            dset_test_omni = SoundspacesDataset(datapath=params['datapath'],
                                       directory_geometry=directory_geometry,
                                       directory_geometry_matterport=directory_geometry_matterport,
                                       directory_geometry_mras=directory_geometry_mras,
                                       directory_jsons=directory_jsons,
                                       directory_jsons_matterport=directory_jsons_matterport,
                                       directory_jsons_mras=directory_jsons_mras,
                                       directory_lmdb=directory_lmdb,
                                       directory_lmdb_maps=directory_lmdb_maps,
                                       directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                       directory_rir_mras=directory_rir_mras,
                                       fname_lmdb=params['fname_lmdb'],
                                       fname_lmdb_maps='maps_relcenter_10x10.lmdb',
                                       scenes=scenes_test,
                                       n_files_per_scene=args['n_files_per_scene'],
                                       read_rirs=False,
                                       read_scenes=True,
                                       read_floormaps=True,
                                       read_acumaps=True,
                                       read_lmdb=params['read_lmdb'],
                                       read_lmdb_maps=params['read_lmdb_maps'],
                                       max_length=params['max_length'],
                                       floormap_processor=floormap_processor,
                                       acumap_processor=acumap_processor,
                                       rir_output_channels=[0],
                                       multi_story_removal=True,
                                       return_rot_angle=False,
                                       avoid_storing_maps_locally=True)


    else:
        raise ValueError(f'ERROR: Unrecognized dataset {dataset_name}')

    if args['do_baseline_spatial_force_omni']:
        return dset_train, dset_test, dset_test_omni
    else:
        return dset_train, dset_test

def get_dataset_full_generalize_unseen_srcs(dataset_name: str, params: Dict, floormap_processor: object, acumap_processor: object, use_DGX=False, use_triton=False, use_vrgpu=False) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    # Same scenes in both datasets, but different sources
    scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'room_2', 'room_0', 'frl_apartment_1', 'apartment_1', 'office_3', 'frl_apartment_0', 'office_2', 'hotel_0', 'room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
    scenes_test = scenes_train
    #scenes_train = ['frl_apartment_3']
    #scenes_test = scenes_train

    # For DGX only
    if use_DGX:
        params['datapath'] = '/mnt/audio/home/ricfalcon/data/soundspaces/data/'
        directory_jsons = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/mnt/audio/home/ricfalcon/data/replica/{:s}/mesh.ply'
        directory_geometry_matterport ='/mnt/audio/home/ricfalcon/data/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/mnt/audio/home/ricfalcon/data/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/outputs/'  # scene_name
    elif use_triton:
        params['datapath'] = ' /scratch/cs/sequentialml/datasets/soundspaces/data/'
        directory_jsons = ' /scratch/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_geometry = '/scratch/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_lmdb = ' /scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb'
    elif use_vrgpu:
        params['datapath'] = '/m/triton/cs/sequentialml/datasets/soundspaces/data/'
        directory_jsons = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_geometry = '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_lmdb = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb'
    else:
        directory_jsons = '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
        directory_geometry = '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
    
    if dataset_name == 'soundspaces':
        dset_train = SoundspacesDataset(datapath=params['datapath'],
                                        directory_geometry=directory_geometry,
                                        directory_jsons=directory_jsons,
                                        scenes=scenes_train,
                                        n_files_per_scene=params['n_files_per_scene'],
                                        read_rirs=True,
                                        read_scenes=True,
                                        read_floormaps=True,
                                        read_acumaps=True,
                                        max_length=params['max_length'],
                                        floormap_processor=floormap_processor,
                                        acumap_processor=acumap_processor,
                                        rir_output_channels=params['rir_output_channels'])
        dset_test = SoundspacesDataset(datapath=params['datapath'],
                                       directory_geometry=directory_geometry,
                                       directory_jsons=directory_jsons,
                                       scenes=scenes_test,
                                       n_files_per_scene=params['n_files_per_scene'],
                                       read_rirs=True,
                                       read_scenes=True,
                                       read_floormaps=True,
                                       read_acumaps=True,
                                       max_length=params['max_length'],
                                       floormap_processor=floormap_processor,
                                       acumap_processor=acumap_processor,
                                       rir_output_channels=params['rir_output_channels'])
    else:
        raise ValueError(f'ERROR: Unrecognized dataset {dataset_name}')
    

    # Split by sources
    dset_train, dset_test = split_by_srcs(dataset_train=dset_train, dataset_test=dset_test, proportion_srcs_test=0.3, sampling='uniform')

    print("After split")
    print(dset_train)
    print(dset_test)
    print(dset_train.data_src_ids[0:20])
    print(dset_test.data_src_ids[0:20])
    return dset_train, dset_test

def get_dataset_full_generalize_unseen_recs(dataset_name: str, params: Dict, floormap_processor: object, acumap_processor: object,  args: object) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    # Same scenes in both datasets, but different sources
    scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'room_2', 'room_0', 'frl_apartment_1', 'apartment_1', 'office_3', 'frl_apartment_0', 'office_2', 'hotel_0', 'room_1', 'frl_apartment_2', 'apartment_2', 'office_1', 'frl_apartment_4']
    scenes_test = scenes_train
    #scenes_train = ['frl_apartment_3']
    #scenes_test = scenes_train

    # INRAS scenes
    scenes_train = ['frl_apartment_3', 'frl_apartment_5', 'apartment_2', 'apartment_1', 'office_4', 'office_3']
    scenes_test = scenes_train

    # For DGX only
    if args['use_dgx']:
        params['datapath'] = '/mnt/audio/home/ricfalcon/data/soundspaces/data/'
        directory_jsons = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/mnt/audio/home/ricfalcon/data/replica/{:s}/mesh.ply'
        directory_geometry_matterport ='/mnt/audio/home/ricfalcon/data/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/mnt/audio/home/ricfalcon/data/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/outputs/'  # scene_name
    elif args['use_triton']:
        params['datapath'] = '/scratch/cs/sequentialml/datasets/soundspaces/data/'
        directory_jsons = '/scratch/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/scratch/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport ='/scratch/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/scratch/cs/sequentialml/datasets/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/scratch/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/scratch/cs/sequentialml/datasets/multiroom3/{:s}/outputs/'  # scene_name
    elif args['use_vrgpu']:
        params['datapath'] = '/m/triton/cs/sequentialml/datasets/soundspaces/data'
        directory_jsons = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport ='/m/triton/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/m/triton/cs/sequentialml/datasets/multiroom3/{:s}/combined.obj' # scene_name, scene_name
        directory_lmdb = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/m/triton/cs/sequentialml/datasets/multiroom3/{:s}/outputs/'  # scene_name
    else:
        directory_jsons = '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
        directory_geometry = '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
        directory_lmdb = '/home/ricfalcon/00data/soundspaces_processed/lmdb/'
    
    if dataset_name == 'soundspaces':
        dset_train = SoundspacesDataset(datapath=params['datapath'],
                                        directory_geometry=directory_geometry,
                                        directory_geometry_matterport=directory_geometry_matterport,
                                        directory_geometry_mras=directory_geometry_mras,
                                        directory_jsons=directory_jsons,
                                        directory_jsons_matterport=directory_jsons_matterport,
                                        directory_jsons_mras=directory_jsons_mras,
                                        directory_lmdb=directory_lmdb,
                                        directory_lmdb_maps=directory_lmdb_maps,
                                        directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                        directory_rir_mras=directory_rir_mras,
                                        fname_lmdb=params['fname_lmdb'],
                                        fname_lmdb_maps=params['fname_lmdb_maps'],
                                        scenes=scenes_train,
                                        n_files_per_scene=args['n_files_per_scene'],
                                        read_rirs=True,
                                        read_scenes=True,
                                        read_floormaps=True,
                                        read_acumaps=True,
                                        read_lmdb=params['read_lmdb'],
                                        read_lmdb_maps=params['read_lmdb_maps'],
                                        max_length=params['max_length'],
                                        floormap_processor=floormap_processor,
                                        acumap_processor=acumap_processor,
                                        rir_output_channels=params['rir_output_channels'],
                                        multi_story_removal=True,
                                        return_rot_angle=args['fmap_add_pose'])
        dset_test = SoundspacesDataset(datapath=params['datapath'],
                                       directory_geometry=directory_geometry,
                                       directory_geometry_matterport=directory_geometry_matterport,
                                       directory_geometry_mras=directory_geometry_mras,
                                       directory_jsons=directory_jsons,
                                       directory_jsons_matterport=directory_jsons_matterport,
                                       directory_jsons_mras=directory_jsons_mras,
                                       directory_lmdb=directory_lmdb,
                                       directory_lmdb_maps=directory_lmdb_maps,
                                       directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                       directory_rir_mras=directory_rir_mras,
                                       fname_lmdb=params['fname_lmdb'],
                                       fname_lmdb_maps=params['fname_lmdb_maps'],
                                       scenes=scenes_test,
                                       n_files_per_scene=args['n_files_per_scene'],
                                       read_rirs=True,
                                       read_scenes=True,
                                       read_floormaps=True,
                                       read_acumaps=True,
                                       read_lmdb=params['read_lmdb'],
                                       read_lmdb_maps=params['read_lmdb_maps'],
                                       max_length=params['max_length'],
                                       floormap_processor=floormap_processor,
                                       acumap_processor=acumap_processor,
                                       rir_output_channels=params['rir_output_channels'],
                                       multi_story_removal=True,
                                       return_rot_angle=args['fmap_add_pose'])
    else:
        raise ValueError(f'ERROR: Unrecognized dataset {dataset_name}')
    

    # Split by recs
    # This is done with masks, computed in the training loop
    return dset_train, dset_test
    
def split_dataset_by_src(dataset: torch.utils.data.Dataset, n_sources=200):
    """ Purges a dataset by keeping only one instance for each src
    This is useful because the floormap and acumap do not change for differnt receivers.
    Only works when there are n sources and 1 scene."""
    sources_per_file = np.arange(start=0, stop=len(dataset))
    unique_srcs = np.unique(dataset.data_src_ids)
    n_sources = math.floor(np.minimum(len(unique_srcs), n_sources))
    selected_srcs = np.random.choice(unique_srcs, n_sources, replace=False)  # select N random sources
    
    subsets = []
    masks_per_src = []
    for src_id in selected_srcs:
        mask = dataset.data_src_ids == src_id
        first_true_index = mask.tolist().index(True) if True in mask else None  # keep only the first id with this source
        new_mask = [i == first_true_index for i in range(len(mask))]
        masks_per_src.append(new_mask)
    mask = [any(mask) for mask in zip(*masks_per_src)]
    mask = np.array(mask)
    assert sum(mask) == n_sources, 'ERROR, this method only supports n sources now'
    
    print(f'len(nmask) : {len(mask)}')
    print(f'sum(nmask) : {sum(mask)}')
    print(f'len(sources_per_file) : {len(sources_per_file)}')
    print(f'len(sources_per_file[mask]) : {len(sources_per_file[mask])}')
    print(f'len(sources_per_file[~mask]) : {len(sources_per_file[~mask])}')

    dataset._remove(sources_per_file[~mask])
    return 
    
def prune_dataset_by_src_and_scene(dataset: torch.utils.data.Dataset, n_sources=1000, n_recs=500, debug=False):
    """ Prunes a dataset. This iterates all scens in the dataset, then selects n_sources
    for each scene, and then selects n_recs for each source-scene.
    
    The main idea is that for each scene, we might want to keep only some sources, and for each source
    only a handful of receivers. This is because each receiver does not change the floormap or acumaps,
    only the RIRs. And because in soundspaces we dont have that many geometries with different acoustics,
    then the contribution of the RIR is limited. """

    print(f'Purging dataset with n_sources={n_sources}, n_recs={n_recs}')
    if debug:
        n_sources = 2  # per scene
        n_recs = 2  # per source, per scene
        data_scenes = np.array([0,0,0,0, 1,1,1,1])
        data_src_ids = np.array([0,1,0,3, 0,2,0,2])
        dataset_ids = np.arange(start=0, stop=len(data_scenes))
    else:
        data_scenes = dataset.data_scenes
        data_src_ids = dataset.data_src_ids
        dataset_ids = np.arange(start=0, stop=len(data_scenes))
    
    candidate_ids = []
    candidate_remove_ids = []
    for scene in tqdm(dataset.scenes, desc='\t\t Processing scenes'):
        mask_scene = data_scenes == scene  # array([False, False, False, False,  True,  True,  True,  True])
        srcs_this_scene = data_src_ids[mask_scene]  # array([0, 2, 0, 2])
        unique_srcs_this_scene = np.unique(srcs_this_scene)  # array([0, 2])
        this_n_sources = np.minimum(n_sources, len(unique_srcs_this_scene))
        selected_srcs = np.random.choice(unique_srcs_this_scene, this_n_sources, replace=False)  # array([2])   , sample n_sources ids
        masks_per_src = []
        
        for src_id in selected_srcs:
            mask_src = np.logical_and(mask_scene , data_src_ids == src_id)
            #first_true_index = mask_src.tolist().index(True) if True in mask_src else None  # keep only the first id with this source
            indices = np.where(mask_src)[0]
            first_n_indices = indices[:n_recs]
            new_mask = [i in first_n_indices for i in range(len(mask_src))]
            masks_per_src.append(new_mask)
            
        mask = [any(mask) for mask in zip(*masks_per_src)]  # Final mask, of n_recs ids, for this scene, for the n_sources unique
        mask = np.array(mask)
        assert mask.sum() <= len(selected_srcs) * n_recs, f'ERROR, wrong number of elements in mask'
        tmp = dataset_ids[mask]  # filter by scene, and selected src id, this keeps the ids that I want to keep
        candidate_ids.extend(tmp.tolist())
        tmp = dataset_ids[~mask] # negate ids, to get ides that I want to remove
        candidate_remove_ids.extend(tmp.tolist())
    
    mask = np.isin(candidate_remove_ids, candidate_ids, invert=True)
    final_ids_to_remove = np.unique(np.array(candidate_remove_ids)[mask]).tolist()

    print(f'\t dataset(len) before removal : {len(dataset)}')
    dataset._remove(final_ids_to_remove)
    print(f'\t dataset(len) after removal : {len(dataset)}')
    assert len(dataset) <= len(dataset.scenes) * n_sources * n_recs,  f'ERROR, wrong number of elements in dataset after purging'
    return dataset  # We modified the original dataset passed by reference, but we return it for easier reading

def get_split_mask_acuamp_by_recs(acumap: torch.Tensor, method='regions', n_neighbors=32, region_size=(48, 48), uniform_threshold=0.9) -> (torch.Tensor, torch.Tensor):
    """ Here we split an acump into 2 complementary maps, by using the receivers.
    This is applied directly to the pixels of the acumap, taking into account those pixels that 
    are not NaN."""

    if method == 'uniform':
        mask = torch.randint(0, 2, acumap.shape, dtype=torch.bool)
    elif method == 'uniform_inras':  # 90% train, 10% test
        #mask = torch.rand(acumap.shape)
        #mask = mask < 0.3
        if len(acumap.shape) > 3:
            raise ValueError('I was expecting no batches when computing masks for receivers')
        c,h,w = acumap.shape  # Here we share the mask across channels
        mask = torch.rand((1,h,w))
        mask = mask < uniform_threshold  # 0.9 --> 90% for train
        mask = torch.tile(mask, (c,1,1))
    elif method == 'neighbors':
        # Create a mask by selecting a random pixel and its neighboring pixels
        height, width = acumap.shape[-2:]
        x, y = np.random.randint(0, height), np.random.randint(0, width)
        mask = torch.zeros_like(acumap, dtype=torch.bool)
        #n_neighbors = np.random.randint(1, 10)  # Choose a number of neighbors
        for dx in range(-n_neighbors, n_neighbors + 1):
            for dy in range(-n_neighbors, n_neighbors + 1):
                if 0 <= x + dx < height and 0 <= y + dy < width:
                    mask[..., x + dx, y + dy] = True
    elif method == 'regions':
        non_nan_indices = torch.where(~torch.isnan(acumap))
        random_index = torch.randint(0, non_nan_indices[0].shape[0], (1,)).item()
        h_center, w_center = non_nan_indices[-2][random_index].item(), non_nan_indices[-1][random_index].item()
        h_slice = slice(max(0, h_center - region_size[0] // 2), min(acumap.shape[-2], h_center + region_size[0] // 2))
        w_slice = slice(max(0, w_center - region_size[1] // 2), min(acumap.shape[-1], w_center + region_size[1] // 2))
    
        mask = torch.zeros_like(acumap, dtype=torch.bool)
        mask[..., h_slice, w_slice] = True

    mask_train = mask
    mask_test = ~mask

    return mask_train, mask_test 
        
def get_processors(params_processor: Dict, params_acu_processor: Dict, fixed_channel_names=False, device='cpu') -> Tuple[object, object]:
    floormap_processor = features.FloorMapProcessor(resolution=params_processor['resolution'], 
                                                    height_selection=params_processor['height_selection'], 
                                                    slice_coord=params_processor['slice_coord'],
                                                    use_slices_variance=params_processor['use_slices_variance'],
                                                    pos_enc_d=params_processor['pos_enc_d'],
                                                    xlim=params_processor['xlim'],
                                                    use_soft_position=params_processor['use_soft_sources'],
                                                    device=device)
    
    acumap_processor = features.AcuMapProcessor(parameters=params_acu_processor['parameters'], 
                                                freq_bands=params_acu_processor['freq_bands'],
                                                distances=params_acu_processor['distances'] ,
                                                resolution=params_acu_processor['resolution'],
                                                use_pooling=True, use_lowpass=True,
                                                pooling_kernel=params_acu_processor['k'], 
                                                pooling_stride=params_acu_processor['s'], 
                                                pooling_padding=params_acu_processor['p'], 
                                                lowpass_std=params_acu_processor['std'],
                                                xlim=params_acu_processor['xlim'])
    
    if False:
        # Single parameter
        acumap_processor = features.AcuMapProcessor(parameters=['c50'], freq_bands=[250, 1000, 4000], 
                                                    distances=['l1'],
                                                    resolution=params_acu_processor['resolution'],
                                                    use_pooling=True, use_lowpass=True,
                                                    pooling_kernel=params_acu_processor['k'], 
                                                    pooling_stride=params_acu_processor['s'], 
                                                    pooling_padding=params_acu_processor['p'], 
                                                    lowpass_std=params_acu_processor['std'],
                                                    xlim=params_acu_processor['xlim'])
        
    if fixed_channel_names:  # This is used when loading maps from lmdb
        floormap_processor.channel_names = ['fixed_slice', 'mask',]
        #acumap_processor.channel_names = []
    
    return floormap_processor, acumap_processor

def get_normalizers(params: Dict) -> Tuple[nn.Module, nn.Module]:
    normalizer = features.MinMaxScalerFixed(vmin_per_channel=[-20, -20, -20, 0.125, 0.125, 0.125, -10, -10, -10],
                                            vmax_per_channel=[20, 20, 20, 4.0, 4.0, 4.0, 10, 10, 10],
                                            log_channels=[3,4,5])

    denormalizer = features.MinMaxDeScalerFixed(vmin_per_channel=[-20, -20, -20, 0.125, 0.125, 0.125, -10, -10, -10],
                                                vmax_per_channel=[20, 20, 20, 4.0, 4.0, 4.0, 10, 10, 10],
                                                log_channels=[3,4,5])

    if True:
    # New normalizers with more adequate ranges 
        normalizer = features.MinMaxScalerFixed(vmin_per_channel=[-5, -5, -5, 0.031, 0.031, 0.031, -15, -15, -15],
                                                vmax_per_channel=[35, 35, 35, 1.0, 1.0, 1.0, 15, 15, 15],
                                                log_channels=[3,4,5])

        denormalizer = features.MinMaxDeScalerFixed(vmin_per_channel=[-5, -5, -5, 0.031, 0.031, 0.031, -15, -15, -15],
                                                    vmax_per_channel=[35, 35, 35, 1.0, 1.0, 1.0, 15, 15, 15],
                                                    log_channels=[3,4,5])
        
        # Standard, but no log
        normalizer = features.MinMaxScalerFixed(vmin_per_channel=[-20, -20, -20, 0.125, 0.125, 0.125, -10, -10, -10],
                                                vmax_per_channel=[20, 20, 20, 4.0, 4.0, 4.0, 10, 10, 10],
                                                )

        denormalizer = features.MinMaxDeScalerFixed(vmin_per_channel=[-20, -20, -20, 0.125, 0.125, 0.125, -10, -10, -10],
                                                    vmax_per_channel=[20, 20, 20, 4.0, 4.0, 4.0, 10, 10, 10],
                                                    )
        
        # very large range, kinda works for MRAS, but throws some NaNs
        normalizer = features.MinMaxScalerFixed(vmin_per_channel=[-40, -40, -40, 0.01, 0.01, 0.01, -20, -20, -20],
                                                vmax_per_channel=[40, 40, 40, 2.0, 2.0, 2.0, 20, 20, 20],
                                                log_channels=[3,4,5])

        denormalizer = features.MinMaxDeScalerFixed(vmin_per_channel=[-40, -40, -40, 0.01, 0.01, 0.01, -20, -20, -20],
                                                    vmax_per_channel=[40, 40, 40, 2.0, 2.0, 2.0, 20, 20, 20],
                                                    log_channels=[3,4,5])
        
        # very large range, but no log, works for MRAS better
        # This is not as good for replica
        normalizer = features.MinMaxScalerFixed(vmin_per_channel=[-40, -40, -40, 0.01, 0.01, 0.01, -20, -20, -20],
                                                vmax_per_channel=[40, 40, 40, 4.0, 4.0, 4.0, 20, 20, 20],
                                                )

        denormalizer = features.MinMaxDeScalerFixed(vmin_per_channel=[-40, -40, -40, 0.01, 0.01, 0.01, -20, -20, -20],
                                                    vmax_per_channel=[40, 40, 40, 4.0, 4.0, 4.0, 20, 20, 20],
                                                    )
    if False:
        # No normalization
        normalizer = IdentityModule()
        denormalizer = IdentityModule()
    
    # Normalizers for single acoustic param
    if False:
        normalizer = features.MinMaxScalerFixed(vmin_per_channel=[-20, -20, -20],
                                                vmax_per_channel=[20, 20, 20],
                                                )

        denormalizer = features.MinMaxDeScalerFixed(vmin_per_channel=[-20, -20, -20],
                                                    vmax_per_channel=[20, 20, 20],
                                                    )
    if False:
        # No normalization
        normalizer = IdentityModule()
        denormalizer = IdentityModule()

    if True:
        # Now we set the normalizers with the paramerters in the config
        normalizer = features.MinMaxScalerFixed(vmin_per_channel=params['normalizer_vmins'],
                                                vmax_per_channel=params['normalizer_vmaxs'],
                                                log_channels=params['normalizer_logchannels'])

        denormalizer = features.MinMaxDeScalerFixed(vmin_per_channel=params['normalizer_vmins'],
                                                    vmax_per_channel=params['normalizer_vmaxs'],
                                                    log_channels=params['normalizer_logchannels'])
    
    return normalizer, denormalizer


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="magma")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def matpotlib_imshow_grid(tensor: torch.Tensor):
    fig = plt.figure(figsize=(14,14))
    plt.imshow(tensor[0, :, :].detach().cpu().numpy(), cmap='magma', origin='lower')
    plt.grid(visible=False)
    plt.tight_layout()
    plt.show()
    return fig

def matplotlib_imshow_floormap(floormap: torch.Tensor, floormap_processor):
    # Plot floormap
    fig, axs = plt.subplots(1, len(floormap_processor.channel_names), figsize=(16,4))
    axs = axs.flatten()
    ctr = 0
    all_imgs = []
    for c in range(floormap.shape[-3]):
        if 'stft' in floormap_processor.channel_names[c]:
            im = axs[ctr].imshow(floormap[c].numpy(), origin='lower', aspect=1, interpolation='antialiased', interpolation_stage='data')     
        else:
            im = axs[ctr].imshow(floormap[c].numpy(), origin='lower', extent=[floormap_processor.xlim[0], floormap_processor.xlim[1], floormap_processor.xlim[0], floormap_processor.xlim[1]], aspect=1, interpolation='antialiased', interpolation_stage='data') 
        all_imgs.append(im)
        axs[ctr].grid(visible=False)
        axs[ctr].set_title(f'{floormap_processor.channel_names[c]}')
        ctr += 1
    plt.tight_layout()
    plt.show()
    return fig

def matplotlib_imshow_acumap_denorm(acumap: torch.Tensor, src: torch.Tensor, acumap_processor: object,):
    """Plot the acuamps, but without normalization. So it shows the details of each scene more clearly.
    However, comparing between images is more difficult."""

    # Remove batch if needed
    if len(acumap.shape) > 3:
        acumap = acumap[0]
    acumap = acumap.detach().cpu()

    # Plot acumap
    fig, axs = plt.subplots(len(acumap_processor.parameters), len(acumap_processor.freq_bands), figsize=(16,14), sharex=True, sharey=True)
    axs = axs.flatten()

    ctr = 0
    all_imgs = []
    for c in range(acumap.shape[-3]):
        im = axs[ctr].imshow(acumap[c].numpy(), origin='lower', 
                             extent=[acumap_processor.xlim[0], acumap_processor.xlim[1], acumap_processor.xlim[0], acumap_processor.xlim[1]], 
                             aspect=1, interpolation='antialiased', interpolation_stage='data') 
        all_imgs.append(im)
        if src is not None:
            axs[ctr].scatter(-1 * src[..., 1].numpy() - 0, src[..., 0].numpy() - 0, s=100, marker='x')
        axs[ctr].grid(visible=False)
        try:
            axs[ctr].set_title(f'{acumap_processor.channel_names[c]}')
        except Exception as e:
            print(e)
            print(acumap_processor.channel_names)
            pass  # I dont know why sometimes this fails
        ctr += 1

    # Colorbars
    tmp_id = len(acumap_processor.freq_bands)

    all_ims_per_row = []
    all_vmins_per_row = []
    all_vmaxs_per_row = []
    for c in range(len(acumap_processor.parameters)):
        this_ims_row = all_imgs[tmp_id * c : tmp_id * c+1]    
        all_ims_per_row.append(this_ims_row)
        all_vmins_per_row.append(min([im.get_clim()[0] for im in this_ims_row]))
        all_vmaxs_per_row.append(min([im.get_clim()[1] for im in this_ims_row]))

    # manual limits
    #vmax_row0 = 30
    #vmax_row1 = 4
    #vmax_row2 = 30
    for c,row in enumerate(all_ims_per_row):
        for im in row:
            im.set_clim(all_vmins_per_row[c], all_vmaxs_per_row[c])

    fig.subplots_adjust(bottom=.75, right=0.8)
    plt.tight_layout(w_pad=10)
    for c,im in enumerate(all_ims_per_row):
        fig.colorbar(im[0], ax=axs[tmp_id*c:tmp_id*(c+1)], use_gridspec=True)
    plt.show()
    return fig

def get_projector_mlp(input_dim, resolution, inner_dim=512):
    model = nn.Sequential(nn.Linear(input_dim, inner_dim),
                          nn.ReLU(),
                          nn.Linear(inner_dim, resolution**2),
                          nn.Unflatten(1, (1, resolution,resolution)))
    return model

def get_projector(input_dim, resolution, inner_dim=1024):
    model = nn.Sequential(nn.Linear(input_dim, 256),
                          nn.ReLU(),
                          nn.Linear(256, inner_dim),
                          nn.Unflatten(1, (1, 32, 32))
                          )
    return model

def compute_ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=3, stride=1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=3, stride=1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.nanmean()

def compute_loss_gradients(estimate, target):
    estimate_dx, estimate_dy = image_gradients(estimate)
    target_dx, target_dy = image_gradients(target)

    criterion = nn.L1Loss(reduction='none')
    loss = criterion(estimate_dx, target_dx) + criterion(estimate_dy, target_dy)
    return loss.nanmean()

def compute_loss_scc(estimate, target):
    loss = SCC(estimate, target, reduction='none')
    return loss.nanmean()

def compute_2d_wass_distance(estimate, target):
    """ Computes the 2d wassertein distance between images.
    This is using random 1d projections, for the sliced wass method."""
    
    batch, chan = target.shape[0], target.shape[1]
    distance = []
    mask = torch.isnan(target)
    for b in range(batch):
        for c in range(chan):
            this_output = estimate[b,c].clone()
            this_output[mask[b,c]] = 0.0
            this_target = target[b,c].clone()
            this_target[mask[b,c]] = 0.0
            ws_sliced_pot_2 = ot.sliced_wasserstein_distance(this_output, this_target)
            distance.append(ws_sliced_pot_2)

    distance = torch.tensor(distance)
    return distance.mean()

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()
    def forward(self, x):
        return x

def analyze_dataset(dataloader: torch.utils.data.DataLoader, device='cpu'):
    print('Computing min and max of acumaps in validation dataset')
    batch_mins = []
    batch_maxs = []
    for _, (fname, rir, src, rec, scene, floormap, acumap) in enumerate(tqdm(dataloader)):
        acumap = acumap.to(device)
        mask = torch.isnan(acumap)

        acumap_min = acumap.clone()
        acumap_min[mask] = 99999.
        tmp_min, _ = acumap_min.min(dim=-1, keepdim=True)
        tmp_min, _ = tmp_min.min(dim=-2, keepdim=True)
        tmp_min, _ = tmp_min.min(dim=0, keepdim=True)
        #print(tmp_min.shape)
        batch_mins.append(tmp_min.squeeze())

        acumap_max = acumap.clone()
        acumap_max[mask] = -99999.
        tmp_max, _ = acumap_max.max(dim=-1, keepdim=True)
        tmp_max, _ = tmp_max.max(dim=-2, keepdim=True)
        tmp_max, _ = tmp_max.max(dim=0, keepdim=True)
        batch_maxs.append(tmp_max.squeeze())
    
    real_min, _ = torch.stack(batch_mins, dim=0).min(dim=0)
    real_max, _ = torch.stack(batch_maxs, dim=0).max(dim=0)

    print(f'Mins:   {real_min.shape}')
    print(real_min)

    print(f'Maxes:   {real_max.shape}')
    print(real_max)

def create_acumap_animation(dataset: torch.utils.data.Dataset, channel_acumap=1, device='cpu'):
    """ Creates a simple animation of acumaps"""
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    n_maps = 500
    vmin = -10  # -5
    vmax = 15  # 25
    #vmin = 0.0  # For t30, edt
    #vmax = 2.0  # For t30, edt

    print(dataset)
    all_acumaps = []
    all_srcs = []
    all_src_ids = []
    ids = np.random.choice(range(len(dataset)), size=n_maps, replace=False)
    ids = np.sort(ids)

    for id in ids:
        (fname, rir, src, rec, scene, floormap, acumap) = dataset[id]
        print(fname)
        _, src_id = fname.split('/')
        src_id, _ = src_id.split('_')
        if src_id not in all_src_ids:
            all_src_ids.append(src_id)
            all_acumaps.append(acumap.clone()[channel_acumap:channel_acumap+1, ...])
            all_srcs.append(src.clone())

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    im = ax.imshow(all_acumaps[0][0].numpy(), origin='lower', extent=[-10, 10, -10, 10], aspect=1, interpolation='antialiased', interpolation_stage='data') 
    im.set_clim(vmin, vmax)
    scatter = ax.scatter(-1 * all_srcs[0][..., 1].numpy() - 0, all_srcs[0][..., 0].numpy() - 0, s=100, marker='x')
    ax.grid(visible=False)
    ax.set_title(f'{dataset.acumap_processor.channel_names[channel_acumap]}')
    fig.colorbar(im, ax=ax, use_gridspec=True)

    def update(i):
        im.set_array(all_acumaps[i][0])
        tmp_src = np.array([-1 * all_srcs[i][..., 1], all_srcs[i][..., 0]])
        scatter.set_offsets(tmp_src)
        #scatter.set_offsets(-1 * all_srcs[i][..., 1].numpy() - 0, all_srcs[i][..., 0].numpy() - 0) 
        return im, scatter

    # Create animation
    print(f'Creating animation with {len(all_src_ids)} frames')
    ani = animation.FuncAnimation(fig, update, frames=range(len(all_acumaps)), interval=500, blit=False)
    writer = FFMpegWriter(fps=1)
    if not os.path.exists('./animations'):
        os.makedirs('./animations')
    ani.save(f'animations/animation_{scene}_channel{channel_acumap}.gif', writer=writer)
    plt.show()

def get_fixed_ids_for_validation(dataloader: torch.utils.data.DataLoader, sample_size=20) -> List[int]:
    """ This returns a random sample of ids for the given dataset. 
    The idea is that for some metrics, evaluating the whole dataset is super slow.
    So here I can get a fixed set of ids, as a random sample of the dataset. And then
    evaluate those metrics only on this subset.
    This works with ids for batches, not samples."""

    sample_size = np.minimum(sample_size, len(dataloader))
    candidates = np.arange(len(dataloader))
    selected_ids = np.random.choice(candidates, sample_size, replace=False)

    return selected_ids

def get_loss_function(loss_function: str = 'l1'):
    if loss_function == 'l1':    
        loss_f = nn.L1Loss()
    elif loss_function == 'sloped_l1':
        loss_f = losses.SlopedL1Loss()
    elif loss_function == 'filterbank2d':
        loss_f = losses.Filterbank2dLoss(criterion='l1')
    elif loss_function == 'wavelet':
        loss_f = losses.WaveletTransformLoss(criterion='l1')
    elif loss_function == 'l2':
        loss_f = nn.MSELoss()
    else:
        raise ValueError(f'ERROR, wrong loss function: {loss_function}')
    return loss_f

def expand_datum(datum, fmap_add_pose=False):
    if not fmap_add_pose:
        fname, rir, src, rec, scene, floormap, acumap = datum
        pose = None  
    else:
        fname, rir, src, rec, scene, floormap, acumap, pose = datum
    return fname, rir, src, rec, scene, floormap, acumap, pose

def train_overfit(args):
    """ Quick test for the UNET model.
    This overfits 2 examples of real data"""
    
    learning_rate, weight_decay = 0.001, 0.0
    batch = 32 if not (args['use_dgx'] or args['use_triton']) else 128
    if 'spatial' in args['exp_name']:
        batch = 8  # 96
    epochs = 100000
    validation_interval = 2500 if not args['dataset'] == 'mras' else 6000
    num_workers = args['num_workers'] if args['use_dgx'] or args['use_triton'] else 4
    num_workers_valid = args['num_workers'] if args['use_dgx'] or args['use_triton'] else 1
    use_projector = False  # for source postion
    if args['net_stft_encoder_concat']:
        use_stft_input = False
    elif args['net_use_multiple_rirs']:
        use_stft_input = False
    else: 
        use_stft_input = args['net_use_stft_input']   # True
    use_split_by_src = True
    use_split_by_recs = args['use_within_room_generalization_method'] != 'none' and args['use_within_room_generalization_method'] is not None
    use_rotations = False
    use_translations = False
    use_translations_centered = False
    use_rotations_centered = False
    use_augmentation_getitem = args['use_augmentation_getitem']
    use_floormap_augmentation = args['use_floormap_augmentation']
    use_dropout = args['net_use_dropout']

    overfit = False
    do_baseline = args['do_baseline']
    do_inference_only = args['do_validation']
    use_film = args['net_use_film']  # default = False
    remove_src_rec = args['net_remove_src_rec']  # default = False
    use_pos_enc_src = args['net_use_pos_enc']  # default = False
    use_pos_enc_rec = args['net_use_pos_enc']  # default = False
    use_multiple_rirs = args['net_use_multiple_rirs']  # default = False
    use_wrong_src = False
    use_concat_stft = args['net_stft_encoder_concat']    # False
    use_custom_linear = False  # Adds a custom linear fucntion to the decoder
    use_multislope_thresholding = args['net_use_threshold']

    do_analyze_dataset = False
    do_create_map_animation = args['do_create_map_animation']
    do_continue_training = args['do_continue_training']  # Resume training from a previous checkpoint
    do_precompute_lmdb = args['do_precompute_rirs_lmdb']  # Only for the train dataset, replica
    do_precompute_lmdb_maps = args['do_precompute_maps_lmdb']  # Only for the train dataset, replica or mattertport

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    debug = False
    profiling = False
    batch_id_tensorboard = 0
    logdir = 'logging'
    run = 'debug'
    run = f"{args['job_id']}_{args['exp_name']}_{args['dataset']}_{args['fold']}"

    if do_baseline:
        run = f"{run}_{'base'}_{args['baseline_method']}"

    if args['comment'] is not None and  args['comment'] != '':
        run = f"{run}_{args['comment']}"

    args['run'] = run
    args['logdir'] = f'{logdir}/{run}'
    print('RUN IS')
    print(run)
    print(logdir)
    parameters.save_config_to_disk(args)

    if args['wandb']:
        if not parameters.setup_wandb(args):
            warnings.warn(f'WARNING, cannot start wandb.')

    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
        device = torch.device("cuda")
        print(device)
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")

    # For MLFlow
    params = {
        "solver": "ranger",
        "epochs": 1000,
        "weight_decay": "weight_decay",
        "learning_rate": learning_rate,
    }


    utils.seed_everything(args['seed'], 'balanced')
    if not debug:
        writer = SummaryWriter(args['logdir'])
    else:
        writer = None

    params, params_processor, params_acumap_processor = get_params(args, 'soundspaces')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Set data, model, modules, loss, optimzer

    if False and params['read_lmdb_maps']: # I think this is not needed now, because the processors are loaded INSIDE the dataset if needed
        import pickle
        # Load processors from disk
        if args['use_vrgpu']:
            this_directory_lmdb_maps = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        elif args['use_triton']:
            this_directory_lmdb_maps = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
            
        this_directory_lmdb_maps = os.path.join(this_directory_lmdb_maps, params['fname_lmdb_maps'])
        with open(os.path.join(this_directory_lmdb_maps, 'floormap_processor.pkl'), 'rb') as handle:
            floormap_processor = pickle.load(handle)
            if not hasattr(floormap_processor, "use_soft_position"):
                setattr(floormap_processor, "use_soft_position", params_processor['use_soft_sources'])
                floormap_processor.init_smoother(device)
            #print(floormap_processor)

        with open(os.path.join(this_directory_lmdb_maps, 'acumap_processor.pkl'), 'rb') as handle:
            acumap_processor = pickle.load(handle)
            print(acumap_processor)
    else:
        # create processors from params
        floormap_processor, acumap_processor = get_processors(params_processor, params_acumap_processor, params['read_lmdb_maps'], device=device)

    normalizer, denormalizer = get_normalizers(params)
    normalizer = normalizer.to(device)
    #normalizer = IdentityModule().to(device)
    denormalizer = denormalizer.to(device)
    #denormalizer = IdentityModule().to(device)
    projector = get_projector(3, params_processor['resolution']).to(device)
    stft_tform = features.Feature_MelPlusPhase(n_mels=128, hop=188, nfft=1024).to(device)
    rotator = augmentation.RandomRotation(device=device).to(device)
    if use_rotations_centered:
        rotator = augmentation.RandomCenteredRotationWithBoundingBox(bbox_channel=params_processor['bbox_channel']).to(device)
    translator = augmentation.RandomTranslation().to(device)
    if use_translations_centered:
        translator = augmentation.RandomTranslationWithBoundingBox(bbox_channel=params_processor['bbox_channel']).to(device)

    if overfit:
        dataset = get_dataset('soundspaces', params, floormap_processor, acumap_processor)
        if use_split_by_src:
            print('Removing stuff from the dataset .... \n')
            split_dataset_by_src(dataset)
            print(dataset)
        dataset_valid = dataset
    else:
        
        if do_inference_only:  
            if args['do_baseline_spatial_force_omni']:  # Only for the baseliens using omni acumaps for spatial targets
                dataset, dataset_valid, dataset_valid_omni = get_dataset_selected_examples('soundspaces', params, floormap_processor, acumap_processor, args)   
            else:
                #dataset, dataset_valid = get_dataset_selected_examples('soundspaces', params, floormap_processor, acumap_processor, args)   # only for visualzation
                dataset, dataset_valid = get_dataset_full('soundspaces', params, floormap_processor, acumap_processor, args)  # for cross dataset validaiton
                dataset_valid_omni = None

            # for baselines, we need the training dataset too
            # UPDATE: we get the train set now from get_dataset_selected_examples
            if False and args['do_baseline']:  
                if args['do_baseline_spatial_force_omni']:  # Only for the baseliens using omni acumaps for spatial targets
                    dataset, _, dataset_valid_omni = get_dataset_full('soundspaces', params, floormap_processor, acumap_processor, args)   
                else:
                    dataset, _ = get_dataset_full('soundspaces', params, floormap_processor, acumap_processor, args)   
                    dataset_valid_omni = None
        else:
            # Only for the baseliens using omni acumaps for spatial targets
            if args['do_baseline_spatial_force_omni']:
                dataset, dataset_valid, dataset_valid_omni = get_dataset_full('soundspaces', params, floormap_processor, acumap_processor, args)   
            else:
                dataset, dataset_valid = get_dataset_full('soundspaces', params, floormap_processor, acumap_processor, args)   
                dataset_valid_omni = None
            if False:
                if use_split_by_recs:
                    dataset, dataset_valid = get_dataset_full_generalize_unseen_recs('soundspaces', params, floormap_processor, acumap_processor, use_DGX=args['use_dgx'], use_triton=args['use_triton'], use_vrgpu=args['use_vrgpu'])    
                else:
                    dataset, dataset_valid = get_dataset_full_generalize_unseen_srcs('soundspaces', params, floormap_processor, acumap_processor, use_DGX=args['use_dgx'], use_triton=args['use_triton'], use_vrgpu=args['use_vrgpu'])
        
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('right after loading datasets')

    # If loading maps from lmdb, load the processors too
    if params['read_lmdb_maps']:
        floormap_processor = dataset.floormap_processor

        # Update the distances if needed:
        dataset.acumap_processor.distances = acumap_processor.distances
        acumap_processor = dataset.acumap_processor

    # Update floormap processor if using soft sources
    if params_processor['use_soft_sources']:
        print('YOLOOOOOOOOOO setting soft source to true')
        floormap_processor.use_soft_position = True
        floormap_processor.init_smoother(device)
        dataset.floormap_processor = floormap_processor
        dataset_valid.floormap_processor = floormap_processor


    print()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print()
    print()
    print(floormap_processor)
    print(acumap_processor)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Precompute dataset 
    if do_precompute_lmdb:
        dataset.max_length = -1
        dataset.avoid_storing_rirs_locally = True
        dataset_valid = None
        precomputeLMDB(dataset)
        return 0
    
    if do_precompute_lmdb_maps:
        dataset.max_length = -1
        dataset_valid = None
        precomputeLMDB_maps(dataset)
        return 0
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Conttinue main code

    # Experimental, apply augmentation in get_item()
    if use_augmentation_getitem:
        translator_tform = augmentation.CombinedAugmentation(bbox_channel=params_processor['bbox_channel'],
                                                             xlim=params_processor['xlim'],
                                                             return_rot_angles=args['fmap_add_pose'])  # Neede for maps with range not in [-10, 10]
        dataset.augmentation_transform = translator_tform
    else:
        translator_tform = None
    assert ~(translator_tform is not None and (use_rotations_centered or use_translations_centered)), f'ERROR, we should not use double augmentations'

    if use_floormap_augmentation:
        augmentation_floormaps = augmentation.FloormapAugmentation(active_displacement=1, 
                                                                   kernel_size = args['aug_kernel_size'], num_kernels=args['aug_kernel_num'],  device=device)
    else:
        augmentation_floormaps = None

    if 'spatial' in args['exp_name']:
        raise NotImplementedError()
        # This is not ready
        # Or rather, this still fails in DGX when loading 2nd order
        dataloader = DataLoader(dataset, batch_size=batch, num_workers=num_workers, shuffle=True,
                                pin_memory=False,
                                prefetch_factor=1)
        dataloader_valid = DataLoader(dataset_valid, batch_size=batch, num_workers=num_workers_valid, drop_last=False,
                                      pin_memory=False,
                                      prefetch_factor=1) 
    else:
        dataloader = DataLoader(dataset, batch_size=batch, num_workers=num_workers, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=batch, num_workers=num_workers_valid, drop_last=False, shuffle=True)  # Zero workers because we dont use augmentation in validation
    
    datum = dataset[0]
    fname, rir, src, rec, scene, floormap, acumap, pose = expand_datum(datum, args['fmap_add_pose'])

    floormap = floormap.to(device)
    src, rec = src.to(device), rec.to(device)
    if not remove_src_rec:
        floormap = floormap_processor.add_src_rec(floormap, src, rec)
    if use_pos_enc_src:
        floormap = floormap_processor.add_positional_encoding(floormap, src, 'src')
    if use_pos_enc_rec:
        floormap = floormap_processor.add_positional_encoding(floormap, rec, 'rec')
    if args['fmap_add_pose']:
        floormap = floormap_processor.add_pose_vector(floormap, pose)
    if use_stft_input:
        stft = stft_tform(rir.to(device))
        floormap = torch.concat([floormap.to(device), stft], dim=-3)
    if use_film:
        stft = stft_tform(rir.to(device))
    if use_multiple_rirs:
        stft = stft_tform(rir.to(device))
        if use_pos_enc_src:
            stft = floormap_processor.add_positional_encoding(stft, src.to(device), 'src')
            stft = floormap_processor.add_positional_encoding(stft, rec.to(device), 'rec')
        else:
            stft = floormap_processor.add_src_rec(stft, src, rec)
    if use_concat_stft:
        stft = stft_tform(rir.to(device))
        #stft = floormap_processor.add_positional_encoding(stft, src.to(device), 'src')
        #stft = floormap_processor.add_positional_encoding(stft, rec.to(device), 'rec')

    print('')
    for datum in [rir, src, rec, floormap, acumap, pose]:
        if datum is not None:
            datum = datum.to(device)
            print(f'Data: {datum.shape}')
            print(f'Data: {datum.dtype}')
    print('')


    # Models stuff
    if not use_film and not use_multiple_rirs and not use_concat_stft:
        if args['net_unet_model'] == 'basic':
            model = UnetBasic(input_channels=floormap.shape[-3], output_channels=acumap.shape[-3],  
                              use_dropout=use_dropout, use_custom_linear=use_custom_linear, block_type=args['net_block'], debug=debug).to(device)
        elif args['net_unet_model']  == 'bottleneck':
            model = UnetBottleNeck(input_channels=floormap.shape[-3], output_channels=acumap.shape[-3],  
                          use_dropout=use_dropout, use_custom_linear=use_custom_linear, block_type=args['net_block'], debug=debug).to(device)
        else:
            raise ValueError(f'Unrecongnized Unet model: {args["net_unet_model"]}')
        
        stft_encoder = None
    if use_multiple_rirs:
        model = UnetFilm(input_channels=floormap.shape[-3], output_channels=acumap.shape[-3],  use_dropout=use_dropout, debug=debug).to(device)
        # RIR channels + src_x, src_y, rec_x, rec_y (positional encodings, not single pixel)
        n_chans = stft.shape[-3]
        stft_encoder = Encoder(input_channels=len(params['rir_output_channels']) + n_chans - 1, use_coordconv=False, use_dropout=use_dropout, 
                               channels=[16, 32, 64, 128, 256, 512, 256],  # channels so that we can concat 2 rirs embeddings
                               return_skip_connections=False, debug=debug).to(device)
    elif use_film:
        model = UnetFilm(input_channels=floormap.shape[-3], output_channels=acumap.shape[-3], use_film_decoder=True,
                          use_dropout=use_dropout, debug=debug).to(device)
        stft_encoder = Encoder(input_channels=len(params['rir_output_channels']), use_coordconv=False, use_dropout=use_dropout, 
                               return_skip_connections=False, debug=debug).to(device)
    elif use_concat_stft:
        conditioning_channels = 512  # Default = 512
        model = UnetConcatBottleNeck(input_channels=floormap.shape[-3], output_channels=acumap.shape[-3],
                                     bottleneck_channels=512, conditioning_channels=conditioning_channels,  
                                     use_bias_linear_projection=False, use_dropout=use_dropout, debug=debug).to(device)
        # DEfult input_channels=len(params['rir_output_channels']) + 4
        stft_encoder = Encoder(input_channels=len(params['rir_output_channels']) + 0, use_coordconv=False, use_dropout=use_dropout, 
                               channels=[16, 32, 64, 128, 256, 512, conditioning_channels],
                               return_skip_connections=False, debug=debug).to(device)
        
    # For the multi-slope case, we add soem thresholding layers
    if use_multislope_thresholding:
        chans = []
        for j in range(len(acumap_processor.freq_bands)):
            for iii in range(3):
                chans.append(iii + (j*3))
        model_threshold_rt = CustomThreshold(min_threshold=0.1, max_threshold=None, channels=chans).to(device)
        chans = []
        for j in range(len(acumap_processor.freq_bands)):
            for iii in range(3,6):
                chans.append(iii + (j*3))
        model_threshold_amp = CustomThreshold(min_threshold=-50, max_threshold=None, channels=chans).to(device)
    else:
        model_threshold_rt = None
        model_threshold_amp = None
        
    #model = Encoder(input_channels=floormap.shape[-3], inner_channels=acumap.shape[-3], debug=debug).to(device)

    # Loss function
    loss_f = get_loss_function(args['loss_function'])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #from ranger import Ranger
    from ranger21 import Ranger21 as Ranger

    if not use_projector:
        optimizer = Ranger(model.parameters(), lr=learning_rate, use_warmup=False, warmdown_active=False, num_epochs=100, num_batches_per_epoch=10)
    else:
        optimizer = Ranger(list(model.parameters()) + list(projector.parameters()), lr=learning_rate, weight_decay=weight_decay)

    if not use_film and not use_multiple_rirs and not use_concat_stft:
        optimizer = Ranger(model.parameters(), lr=learning_rate, use_warmup=False, warmdown_active=False, num_epochs=100, num_batches_per_epoch=10)
    else:
        optimizer = Ranger(list(model.parameters()) + list(stft_encoder.parameters()), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2500, gamma=0.9, last_epoch=- 1, verbose=False)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Show model and profile if needed
    torch_summary_depth = 5  # default = 3
    if use_projector:
        summary(projector, input_data=torch.rand(1,3).to(device),
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=3, dtypes=[dtype])
        
    if not use_film and not use_multiple_rirs and not use_concat_stft:
        summary(model, input_data=floormap[None, ...].to(device),
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=torch_summary_depth, dtypes=[dtype])
    elif use_multiple_rirs:
        summary(stft_encoder, input_data=stft[None, ...].to(device),
        col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
        row_settings=['depth'], depth=2, dtypes=[dtype])

        conditioning = stft_encoder(stft[None, ...])
        conditioning_aux = stft_encoder(stft[None, ...])
        conditioning = torch.concat([conditioning, conditioning_aux], dim=-3)
        summary(model, input_data=[floormap[None, ...].to(device), conditioning],
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=3, dtypes=[dtype])
    elif use_film:
        summary(stft_encoder, input_data=stft[None, ...].to(device),
        col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
        row_settings=['depth'], depth=2, dtypes=[dtype])

        conditioning = stft_encoder(stft[None, ...])
        summary(model, input_data=[floormap[None, ...].to(device), conditioning],
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=3, dtypes=[dtype])
    elif use_concat_stft:
        summary(stft_encoder, input_data=stft[None, ...].to(device),
        col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
        row_settings=['depth'], depth=2, dtypes=[dtype])

        conditioning = stft_encoder(stft[None, ...])
        summary(model, input_data=[floormap[None, ...].to(device), conditioning],
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=3, dtypes=[dtype])


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Profiling setup
    # activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    if profiling:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=1, active=50, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logging/profile2'),
            record_shapes=True,
            with_stack=False)  # False to avoid Security problems when running remote Tensorboard
        prof.start()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Analysis of dataset
    if do_analyze_dataset:
        print('Analyzing dataset')
        analyze_dataset(dataloader_valid, device='cpu')
        return 0
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Animation of some acumap
    if do_create_map_animation:
        print('Computing animation')
        create_acumap_animation(dataset_valid, channel_acumap=args['do_animation_channel'], device='cpu')
        return 0

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Baselines only
    if do_baseline: 
        if do_inference_only:  # this is using a small number of samples, mostly for plotting
            dataset_inference_only = dataset_valid
            dataset_for_valid = dataset
            writer = None
        else:
            dataset_inference_only = None
            dataset_for_valid = dataset_valid
        dataset_for_valid.avoid_storing_maps_locally = True  # Do not store maps in memory to avoid OOM issues

        dataset_for_valid.augmentation_transform = None
        if dataset_inference_only is not None:
            dataset_inference_only.augmentation_transform = None
        
        # TODO REMOVE THIS
        #dataset_for_valid = dataset_inference_only
        compute_baselines(dataset_for_valid, device=device, normalizer=normalizer, denormalizer=denormalizer,
                          writer=writer, floormap_processor=floormap_processor, acumap_processor=acumap_processor,
                          use_multislope_thresholding=use_multislope_thresholding, model_threshold_rt=model_threshold_rt, model_threshold_amp=model_threshold_amp,
                          method=args['baseline_method'], dataset_name=args['dataset'],
                          random_sample_dataset=args['validation_random_sample'],
                          logdir=f'{logdir}/{run}',
                          do_directional=args['do_baseline_spatial'], do_directional_omni=args['do_baseline_spatial_force_omni'],
                          use_decay_fitnet=args['do_baseline_multislope'],
                          dataset_valid_omni=dataset_valid_omni, dataset_inference_only=dataset_inference_only)
        if writer is not None:
            writer.close()
        return 0
    

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Inference only
    if do_inference_only:
        checkpoint = '73345_exp05_dropout_rot_tran_maskrecs_2222'
        checkpoint = '73327_exp05_dropout_rot_tran_maskrecs'
        checkpoint = args['validation_checkpoint']
        #checkpoint = 'debug_debug'
        writer = None
        validation_only(checkpoint, dataloader=dataloader, dataloader_valid=dataloader_valid, net=model, device=device,
                        floormap_processor=floormap_processor, acumap_processor=acumap_processor, normalizer=normalizer, denormalizer=denormalizer,
                        stft_tform=stft_tform, projector=projector, stft_encoder=stft_encoder,
                        use_multislope_thresholding=use_multislope_thresholding, model_threshold_rt=model_threshold_rt, model_threshold_amp=model_threshold_amp,
                        remove_src_rec=remove_src_rec, use_stft_input=use_stft_input, use_projector=use_projector,
                        use_film=use_film, use_split_by_recs=use_split_by_recs,
                        writer=writer, logdir=logdir, run=run, batch_id_tensorboard=batch_id_tensorboard,
                        use_floormap_augmentation=use_floormap_augmentation, augmentation_floormaps=augmentation_floormaps)
        if writer is not None:
            writer.close()
        return 0
    

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Continue training a previous model
    if do_continue_training:
        checkpoint = args['validation_checkpoint']
        checkpoint_path = os.path.join('./logging', checkpoint)
        model = load_model(model, checkpoint_path, logdir, device)
        return 0
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Train
    
    masks_recs_train = {}
    masks_recs_test = {}
    best_loss_valid = 1e10  # keep track of validation loss to save the best model so far
    ids_for_validaiton = get_fixed_ids_for_validation(dataloader_valid)  # these are batch ids

    ctr = 0
    time_start = time.time()
    print(f'Device: {device}')
    for epoch in range(epochs):
        if profiling:
            prof.step()
            if ctr >= 5 + 1 + 50 * 1:
                break    
        for _, (datum) in enumerate(dataloader):
            fname, rir, src, rec, scene, floormap, acumap, pose = expand_datum(datum, args['fmap_add_pose'])
            model.train()
            src, rec = src.to(device), rec.to(device)
            floormap = floormap.to(device)
            acumap = acumap.to(device)
            if pose is not None:
                pose = pose.to(device)

            if use_floormap_augmentation:
                floormap = augmentation_floormaps(floormap)

            if use_wrong_src:
                # Sample another src and rec position
                ids_aux = []
                for ii in range(len(scene)):
                    ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(dataloader.dataset, scene=scene[ii]))

                subset = torch.utils.data.Subset(dataloader.dataset, ids_aux)
                subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch)
                fname_aux, _, src_aux, rec_aux, scene_aux, _, _ = next(iter(subset_loader))
                src = src_aux.to(device)
                rec = rec_aux.to(device)
                assert scene_aux == scene, f'ERROR: The scene from the auxiliary rir is different'

            if use_multiple_rirs:
                if False:
                    src_aux = src.clone().to(device)
                    rec_aux = rec.clone().to(device)
                    rir_aux = rir.clone().to(device)
                else:
                    ids_aux = []
                    for ii in range(len(scene)):
                        ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(dataloader.dataset, scene=scene[ii]))

                    subset = torch.utils.data.Subset(dataloader.dataset, ids_aux)
                    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch)
                    fname_aux, rir_aux, src_aux, rec_aux, scene_aux, _, _ = next(iter(subset_loader))
                    src_aux = src_aux.to(device)
                    rec_aux = rec_aux.to(device)
                    assert scene_aux == scene, f'ERROR: The scene from the auxiliary rir is different'

                    #print(f'src_aux.shape {src_aux.shape}')

            if debug:
                print(f'TRAIN rir: {rir.shape}')
                print(f'TRAIN src: {src.shape}')
                print(f'TRAIN rec: {rec.shape}')
                print(f'TRAIN floormap: {floormap.shape}')
                print(f'TRAIN acumap: {acumap.shape}')

            # Get relevant masks (that can be rotated / translated later)
            if not use_split_by_recs:
                mask = torch.logical_not(torch.isnan(acumap)).clone()
            else:
                keys = []
                for iii, k in enumerate(fname):  #fname is tuple of fnames, due to collate
                    tmp = k.split('/')  # 'office_0/10_39.wav' -> office_0, 10_39.wav
                    tmp2 = tmp[-1].split('_')  # 10_39.wav -> 10, 39.wav
                    k = f'{tmp[0]}/{tmp2[0]}'  # office_0/10
                    keys.append(k)
                if epoch == 0:
                    for iii, k in enumerate(keys):  #fname is tuple of fnames, due to collate
                        if k not in masks_recs_test:
                            tmp_mask_train, tmp_mask_test = get_split_mask_acuamp_by_recs(acumap=acumap[iii], 
                                                                                          method=args['use_within_room_generalization_method'], 
                                                                                          uniform_threshold=args['use_within_room_generalization_threshold'])
                            masks_recs_train[k] = tmp_mask_train
                            masks_recs_test[k] = tmp_mask_test

                            print(f'fname: {k}')
                            print(f'mask_train.sum {tmp_mask_train.nansum()}')
                            print(f'mask_test.sum {tmp_mask_test.nansum()}')

                pre_mask = [masks_recs_train[k] for k in keys]
                pre_mask = torch.stack(pre_mask, dim=0).to(device)
                a = torch.logical_not(torch.isnan(acumap))
                #print(f'a.shape {a.shape}')
                #print(f'pre_mask.shape {pre_mask.shape}')
                mask = torch.logical_not(torch.isnan(acumap)) * pre_mask
                
                #print(f'a.nansum() {a.nansum()}')
                #print(f'pre_mask.nansum {pre_mask.nansum()}')
                #print(f'mask.nansum {mask.nansum()}')

            # Data augmentation
            if (use_rotations or use_rotations_centered):
                if use_multiple_rirs:
                    # Cocnat multiple srcs and recs before rotation
                    srcs = [src, src_aux]
                    recs = [rec, rec_aux]
                    srcs, recs, floormap, acumap, mask = rotator(srcs, recs, floormap, acumap, mask)

                    src, src_aux = srcs[0], srcs[1]
                    rec, rec_aux = recs[0], recs[1]
                else:
                    src, rec, floormap, acumap, mask = rotator(src, rec, floormap, acumap, mask)
            if (use_translations or use_translations_centered):
                ##print(f'yoloooooooooooo: mask.nansum {mask.nansum()}')
                ##print(f'yoloooooooooooo: mask.shape {mask.shape}')
                if use_multiple_rirs:
                    # Cocnat multiple srcs and recs before rotation
                    srcs = [src, src_aux]
                    recs = [rec, rec_aux]
                    srcs, recs, floormap, acumap, mask = translator(srcs, recs, floormap, acumap, mask)

                    src, src_aux = srcs[0], srcs[1]
                    rec, rec_aux = recs[0], recs[1]
                else:
                    src, rec, floormap, acumap, mask = translator(src, rec, floormap, acumap, mask)
                ##print(f'yoloooooooooooo: mask.nansum {mask.nansum()}')
                ##print(f'yoloooooooooooo: mask.shape {mask.shape}')

            # Normalization and adding sources
            with torch.no_grad():    
                if not remove_src_rec: 
                    floormap = floormap_processor.add_src_rec(floormap, src, rec)
                if use_pos_enc_src:
                    floormap = floormap_processor.add_positional_encoding(floormap, src, 'src')
                if use_pos_enc_rec:
                    floormap = floormap_processor.add_positional_encoding(floormap, rec, 'rec')
                if args['fmap_add_pose']:
                    floormap = floormap_processor.add_pose_vector(floormap, pose)

                acumap = normalizer(acumap).to(torch.float32)
                if debug:
                    print(f'TRAIN-post rir: {rir.shape}')
                    print(f'TRAIN-post src: {src.shape}')
                    print(f'TRAIN-post rec: {rec.shape}')
                    print(f'TRAIN-post floormap: {floormap.shape}')
                    print(f'TRAIN-post acumap: {acumap.shape}')

                # TODO Debugging this part, because I think there is a bug when normalizing
                if not use_split_by_recs:
                    post_mask = torch.logical_not(torch.isnan(acumap)) 

            # RIR encoding and spectrograms
            if use_stft_input:
                stft = stft_tform(rir.to(device))
                floormap = torch.concat([floormap, stft], dim=-3)
                if len(floormap_processor.channel_names) < floormap.shape[-3]:
                    for j in range(stft.shape[-3]):
                        floormap_processor.channel_names.append(f'stft{j}')

            # RIR encodfing for multiptle RIRs
            if use_multiple_rirs:
                stft = stft_tform(rir.to(device))
                stft_aux = stft_tform(rir_aux.to(device))

                # Concat src and rec pos to RIR encoder
                #floormap = floormap_processor.add_src_rec(floormap, src_aux, rec_aux)  # floormap goes to main Unet, src here is target

                if use_pos_enc_src:
                    stft = floormap_processor.add_positional_encoding(stft, src, 'src')
                    stft = floormap_processor.add_positional_encoding(stft, rec, 'rec')
                    stft_aux = floormap_processor.add_positional_encoding(stft_aux, src_aux, 'src')
                    stft_aux = floormap_processor.add_positional_encoding(stft_aux, rec_aux, 'rec')
                else:
                    stft = floormap_processor.add_src_rec(stft, src, rec)
                    stft_aux = floormap_processor.add_src_rec(stft_aux, src_aux, rec_aux)

                #print(f'stft shape {stft.shape}')  # yoloooooooo
                #print(f'stft_aux shape {stft_aux.shape}')  # yoloooooooo

                if use_stft_input:
                    floormap = torch.concat([floormap, stft_aux], dim=-3)
                    if len(floormap_processor.channel_names) < floormap.shape[-3]:
                        for j in range(stft.shape[-3]):
                            floormap_processor.channel_names.append(f'stft_aux{j}')
                conditioning = stft_encoder(stft)
                conditioning_aux = stft_encoder(stft_aux)
                conditioning = torch.concat([conditioning, conditioning_aux], dim=-3)

            if use_film:
                stft = stft_tform(rir.to(device))
                conditioning = stft_encoder(stft)

            if use_concat_stft:
                stft = stft_tform(rir.to(device))
                #stft = floormap_processor.add_positional_encoding(stft, src.to(device), 'src')
                #stft = floormap_processor.add_positional_encoding(stft, rec.to(device), 'rec')
                conditioning = stft_encoder(stft)

            # Projection / encoding of source position
            if use_projector:
                # Replace the src with a 2d linear projection
                src_projected = projector(src)
                src_projected = src_projected.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)  # Only if we need to reshape stuff
                id = floormap_processor.channel_names.index('source')
                floormap[..., id, :, :] = src_projected[..., 0, :, :]
                
                #floormap = torch.concat([floormap, src_projected], dim=-3)
                #if 'src_proj' not in floormap_processor.channel_names:
                #    floormap_processor.channel_names.append('src_proj')
            
            if debug:
                print(mask)
                print(f'mask.shape {mask.shape}')
                print(f'out[mask].shape {out[mask].shape}')
                print(f'acumap.shape {acumap.shape}')
    
                nan_count_1 = torch.sum(torch.isnan(out)).item()
                nan_count_2 = torch.sum(torch.isnan(acumap)).item()
                print(f'NaNs out: {nan_count_1}')
                print(f'NaNs out: {nan_count_2}')
    
                print(f'out.dtype {out.dtype}')
                print(f'acumap.dtype {acumap.dtype}')
                print(f'mask.dtype {mask.dtype}')
    
                print(f'out[mask].dtype {out[mask]}')
                print(f'acumap[mask].dtype {acumap[mask]}')

            # Validate floormap
            non_zero_channels = (floormap > 0.0).any(dim=0, keepdims=True).any(dim=2, keepdims=True).any(dim=3, keepdims=True)
            assert non_zero_channels.all(), 'ERROR, some channels in the floormap as all 0.0. Is the src and rec properly added?'

            model.zero_grad()
            if not use_film and not use_multiple_rirs and not use_concat_stft:
                out = model(floormap)
            else:
                #print(f'floormap.shape {floormap.shape}')   # yolooooooo
                #print(f'conditioning.shape {conditioning.shape}')  # yolooooooo
                #print(f'fmap processor {floormap_processor.channel_names}') # yolooooooo
                out = model(floormap, conditioning)

            # Multi-slope mode, we add threshold to some outputs
            if use_multislope_thresholding:
                out = model_threshold_rt(out)
                out = model_threshold_amp(out)

            if use_split_by_recs:
                loss = loss_f(out[mask], acumap[mask])
                post_mask = None
            else:
                mask = post_mask
                loss = loss_f(out[mask], acumap[mask])

            if torch.isnan(loss):
                print('WARNING: NaN detected in loss. ')
                print(fname)
                print(f'out[mask].shape {out[mask].shape}')
                print(f'acumap[mask].shape {acumap[mask].shape}')

                nan_count_1 = torch.sum(torch.isnan(out[mask])).item()
                nan_count_2 = torch.sum(torch.isnan(acumap[mask])).item()
                print(f'NaNs out: {nan_count_1}')
                print(f'NaNs acumap: {nan_count_2}')
                if post_mask is not None:
                    nan_count_1 = torch.sum(torch.isnan(out[post_mask])).item()
                    nan_count_2 = torch.sum(torch.isnan(acumap[post_mask])).item()
                    print(f'NaNs out: {nan_count_1}')
                    print(f'NaNs acumap: {nan_count_2}')

                out[~mask] = 0.0
                where_nans = torch.where(torch.any(torch.isnan(out),dim=[-3,-2,-1]))
                print(f'Batch ids where nans : {where_nans}')

                raise ValueError('ERROR, there are anans in the out. Giving up')
            loss.backward()
            optimizer.step()
    
            # Loggin stuff
            if ctr % 100 == 0: 
                if writer is not None:
                    writer.add_scalar('Loss/train', loss.item(), ctr)
            if ctr % 500 == 0:
                time_now = time.time()
                elapsed = time_now - time_start
                print('Epoch: {} / {} , loss {:.8f} \t\t {}'.format(epoch, epochs, loss.item(), time.strftime("%H:%M:%S", time.gmtime(elapsed))))   # '04h13m06s'))
                if not use_multiple_rirs:
                    log_output_examples(ctr, writer, out, floormap, acumap, mask, floormap_processor, subset='train', batch_id_tensorboard=batch_id_tensorboard, 
                                        n_freq_bands=len(acumap_processor.freq_bands), acumap_processor=acumap_processor, denormalizer=denormalizer, use_split_unseen_recs=use_split_by_recs)
                else:
                    log_output_examples(ctr, writer, out, floormap, acumap, mask, floormap_processor, subset='train', batch_id_tensorboard=batch_id_tensorboard, 
                                        stft=stft, n_freq_bands=len(acumap_processor.freq_bands), acumap_processor=acumap_processor, denormalizer=denormalizer, use_split_unseen_recs=use_split_by_recs)

                
            if ctr % validation_interval == 0:
                # Save checkpoint
                save_model(model, optimizer, logdir=f'{logdir}/{run}', iteration=ctr, model_name='last')
                
                # Validiton loop
                model.eval()
                
                loss_f_valid = nn.L1Loss(reduction='none')
                loss_mssim = MSSIM(kernel_size=9, betas=(0.0448, 0.2856, 0.3001, 0.2363), reduction='none').to(device)
                #betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
                #loss_SCC = SCC(reduction='none').to(device)

                print(f'Validation {ctr}')
                # When using augmentaitonin get_item, we remove it to get the metrics
                # Then enable it again
                if dataloader.dataset.augmentation_transform is not None:
                    dataloader.dataset.augmentation_transform = None

                # For MRAS, we use a random sample of the training set, because its huge
                if args['dataset'] == 'mras':
                    ids = np.random.choice(len(dataloader.dataset), np.minimum(500, len(dataloader.dataset)), replace=False)
                    datatset_train_subset = torch.utils.data.Subset(dataloader.dataset, ids)
                    dataloader_train_subset = DataLoader(datatset_train_subset, batch_size=batch, num_workers=num_workers, shuffle=True)
                else:
                    dataloader_train_subset = dataloader

                for loader, subset in zip([dataloader_train_subset, dataloader_valid], ['train', 'valid']):
                    all_metrics = {}
                    with torch.no_grad():
                        for batch_id, (datum) in enumerate(loader):
                            fname, rir, src, rec, scene, floormap, acumap, pose = expand_datum(datum, args['fmap_add_pose'])
                            src, rec = src.to(device), rec.to(device)
                            floormap = floormap.to(device)
                            acumap = acumap.to(device)
                            if pose is not None:
                                pose = pose.to(device)

                            if use_wrong_src:
                                # Sample another src and rec position
                                ids_aux = []
                                for ii in range(len(scene)):
                                        ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(loader.dataset, scene=scene[ii]))

                                tmp_subset = torch.utils.data.Subset(loader.dataset, ids_aux)
                                tmp_subset_loader = torch.utils.data.DataLoader(tmp_subset, batch_size=batch)
                                fname_aux, _, src_aux, rec_aux, scene_aux, _, _ = next(iter(tmp_subset_loader))
                                src = src_aux.to(device)
                                rec = rec_aux.to(device)
                                assert scene_aux == scene, f'ERROR: The scene from the auxiliary rir is different'

                            if use_multiple_rirs:
                                ids_aux = []
                                for ii in range(len(scene)):
                                    if args['dataset'] == 'mras':
                                        ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(loader.dataset.dataset, scene=scene[ii]))
                                    else:    
                                        ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(loader.dataset, scene=scene[ii]))

                                tmp_subset = torch.utils.data.Subset(loader.dataset, ids_aux)
                                tmp_subset_loader = torch.utils.data.DataLoader(tmp_subset, batch_size=batch)
                                fname_aux, rir_aux, src_aux, rec_aux, scene_aux, _, _ = next(iter(tmp_subset_loader))
                                src_aux = src_aux.to(device)
                                rec_aux = rec_aux.to(device)
                                assert scene_aux == scene, f'ERROR: The scene from the auxiliary rir is different'

                            if not remove_src_rec:
                                floormap = floormap_processor.add_src_rec(floormap, src, rec)
                            if use_pos_enc_src:
                                floormap = floormap_processor.add_positional_encoding(floormap, src, 'src')
                            if use_pos_enc_rec:
                                floormap = floormap_processor.add_positional_encoding(floormap, rec, 'rec')
                            if args['fmap_add_pose']:
                                floormap = floormap_processor.add_pose_vector(floormap, pose)
                            acumap = normalizer(acumap).to(torch.float32)
             
                            if use_stft_input:
                                stft = stft_tform(rir.to(device))
                                floormap = torch.concat([floormap, stft], dim=-3)
                                if len(floormap_processor.channel_names) < floormap.shape[-3]:
                                    for j in range(stft.shape[-3]):
                                        floormap_processor.channel_names.append(f'stft{j}')
                                
                            if use_projector:
                                # Replace the src with a 2d linear projection
                                src_projected = projector(src)
                                src_projected = src_projected.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)  # Only if we need to reshape stuff
                                id = floormap_processor.channel_names.index('source')
                                floormap[..., id, :, :] = src_projected[..., 0, :, :]

                            # RIR encodfing for multiptle RIRs
                            if use_multiple_rirs:
                                stft = stft_tform(rir.to(device))
                                stft_aux = stft_tform(rir_aux.to(device))

                                # Concat src and rec pos to RIR encoder
                                #floormap = floormap_processor.add_src_rec(floormap, src_aux, rec_aux)
                                if use_pos_enc_src:
                                    stft = floormap_processor.add_positional_encoding(stft, src, 'src')
                                    stft = floormap_processor.add_positional_encoding(stft, rec, 'rec')
                                    stft_aux = floormap_processor.add_positional_encoding(stft_aux, src_aux, 'src')
                                    stft_aux = floormap_processor.add_positional_encoding(stft_aux, rec_aux, 'rec')
                                else:
                                    stft = floormap_processor.add_src_rec(stft, src, rec)
                                    stft_aux = floormap_processor.add_src_rec(stft_aux, src_aux, rec_aux)

                                if use_stft_input:
                                    floormap = torch.concat([floormap, stft_aux], dim=-3)
                                    if len(floormap_processor.channel_names) < floormap.shape[-3]:
                                        for j in range(stft.shape[-3]):
                                            floormap_processor.channel_names.append(f'stft_aux{j}')

                                conditioning = stft_encoder(stft)
                                conditioning_aux = stft_encoder(stft_aux)
                                conditioning = torch.concat([conditioning, conditioning_aux], dim=-3)

                            if use_film:
                                stft = stft_tform(rir.to(device))
                                conditioning = stft_encoder(stft)

                            if use_concat_stft:
                                stft = stft_tform(rir.to(device))
                                #stft = floormap_processor.add_positional_encoding(stft, src.to(device), 'src')
                                #stft = floormap_processor.add_positional_encoding(stft, rec.to(device), 'rec')
                                conditioning = stft_encoder(stft)
                                
                            if not use_film and not use_multiple_rirs and not use_concat_stft:
                                out = model(floormap)
                            else:
                                out = model(floormap, conditioning)

                            # Multi-slope mode, we add threshold to some outputs
                            if use_multislope_thresholding:
                                out = model_threshold_rt(out)
                                out = model_threshold_amp(out)

                            if not use_split_by_recs:
                                mask = torch.logical_not(torch.isnan(acumap)) 
                            else:
                                keys = []
                                for iii, k in enumerate(fname):  #fname is tuple of fnames, due to collate
                                    tmp = k.split('/')  # 'office_0/10_39.wav' -> office_0, 10_39.wav
                                    tmp2 = tmp[-1].split('_')  # 10_39.wav -> 10, 39.wav
                                    k = f'{tmp[0]}/{tmp2[0]}'  # office_0/10
                                    keys.append(k)
                                if epoch == 0:
                                    for iii, k in enumerate(keys):  #fname is tuple of fnames, due to collate
                                        tmp = k.split('/')  # 'office_0/10_39.wav' -> office_0, 10_39.wav
                                        tmp2 = tmp[-1].split('_')  # 10_39.wav -> 10, 39.wav
                                        k = f'{tmp[0]}/{tmp2[0]}'  # office_0/10
                                        if k not in masks_recs_test:
                                            tmp_mask_train, tmp_mask_test = get_split_mask_acuamp_by_recs(acumap=acumap[iii], 
                                                                                                          method=args['use_within_room_generalization_method'],
                                                                                                          uniform_threshold=args['use_within_room_generalization_threshold'])
                                            masks_recs_train[k] = tmp_mask_train
                                            masks_recs_test[k] = tmp_mask_test
                                if subset == 'train':
                                    pre_mask = [masks_recs_train[k] for k in keys]
                                elif subset == 'valid':
                                    pre_mask = [masks_recs_test[k] for k in keys]
                                pre_mask = torch.stack(pre_mask, dim=0).to(device)
                                mask = torch.logical_not(torch.isnan(acumap)) * pre_mask

                            
                            #loss = loss_f_valid(out[mask], acumap[mask])   # this removes the shape of the tensors
                            ############## print(f'maskkkkkk {mask.shape}')
                            loss = loss_f_valid(out * mask, acumap * mask)
                            ############## print(f'loss {loss.shape}')
                            denom = mask.nansum(dim=(-2,-1))
                            loss = loss.nansum(dim=(-2,-1)) / denom  # pixelwise mean, for valid pixels only, [batch, channels], there should not be any nans here


                            if torch.any(torch.isnan(loss)):
                                print('WARNING: NaN detected in loss. ')
                                print(fname)
                                print(f'out[mask].shape {out[mask].shape}')
                                print(f'acumap[mask].shape {acumap[mask].shape}')

                                nan_count_1 = torch.sum(torch.isnan(out[mask])).item()
                                nan_count_2 = torch.sum(torch.isnan(acumap[mask])).item()
                                print(f'NaNs out: {nan_count_1}')
                                print(f'NaNs acumap: {nan_count_2}')
                                if post_mask is not None:
                                    nan_count_1 = torch.sum(torch.isnan(out[post_mask])).item()
                                    nan_count_2 = torch.sum(torch.isnan(acumap[post_mask])).item()
                                    print(f'NaNs out: {nan_count_1}')
                                    print(f'NaNs acumap: {nan_count_2}')

                                out[~mask] = 0.0
                                where_nans = torch.where(torch.any(torch.isnan(out),dim=[-3,-2,-1]))
                                print(f'Batch ids where nans : {where_nans}')

                                raise ValueError('ERROR, there are anans in the out. Giving up')

                            ### loss = loss.nanmean(dim=-2).nanmean(-1)   # Pixelwise mean, [batch, channels], ignore NaNs, but uses wrong denominator (all pixels)
                            ############## print(f'loss.shape {loss.shape}')
                            ############## print(f'loss nay nan {torch.any(torch.isnan(loss))}')
                            ############## loss = loss.view(out.shape[0], out.shape[1], -1).mean(dim=-1)  # Pixelwise mean, [batch, channels]
                            out_denormalized = denormalizer(out)

                            # Loss with other masking
                            loss_second = loss_f_valid(out[mask], acumap[mask])   # this removes the shape of the tensors
                            ##############print(f'loss_second.shape   {loss_second.shape}')
                            loss_second = loss_second.mean()  # float
                            ##############print(f'loss_second.mean.shape   {loss_second}')


                            # NEW SSIM metric for similarity between image-like data
                            ssim = compute_ssim(out * mask, acumap * mask) 

                            # New MSSIM
                            tmp_acumap = acumap.clone()
                            tmp_acumap[~mask] = 0.0  # hack to avoid problems with nans
                            mssim = loss_mssim(out * mask, tmp_acumap * mask)
                            mssim = mssim.nanmean()

                            # New SCC
                            scc = compute_loss_scc(out * mask, tmp_acumap * mask)
                            #scc = compute_loss_scc(out[mask], acumap[mask])

                            # New ImageGradients
                            imag_grad = compute_loss_gradients(out * mask, tmp_acumap * mask)

                            # New metrics for image simalarity
                            # 2d wassertein
                            if batch_id in ids_for_validaiton:
                                tmp_out = out.clone()
                                tmp_out[~mask] = 0.0 # hack to avoid problems with nans
                                tmp_acumap = acumap.clone()
                                tmp_acumap[~mask] = 0.0  # hack to avoid problems with nans
                                #wass_2d = compute_2d_wass_distance(out, tmp_acumap)
                                loss_fn = losses.Wasserstein2dLoss()
                                wass_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
                                loss_fn = losses.WaveletTransformLoss(return_breakdown=True)
                                wavelet_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
                                loss_fn = losses.Filterbank2dLoss(return_breakdown=True)
                                filterbank_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
                            else:
                                wass_2d = None
                                wavelet_2d = None
                                filterbank_2d = None

                            # Additional labels to analyze errors
                            src_dist = torch.sqrt(torch.sum((src - rec) ** 2, dim=1))  # Eucldean distance between src and reference receiver
                            scene_area = torch.sum(floormap[..., 1, :, :], dim=(-2,-1))  # Scene area 

                            error_metrics = evaluation(out_denormalized, denormalizer(acumap), mask, acumap_processor=acumap_processor)
                            for k,v in error_metrics.items():
                                #print(f'k, v: {k} : {v}')

                                # Parse keys for spatial metrics only
                                if '.' in k:
                                    tmp_dir, tmp_param = k.split('/')  # 0.0/c50 --> 0.0, c50
                                    k = f'{tmp_param}/{tmp_dir}'  # c50/0.0
                                if k not in all_metrics:
                                    all_metrics[k] = [v]
                                else:
                                    all_metrics[k].append(v)    
                            
                            # append fname so I can find some good examples later
                            if 'fname' not in all_metrics:
                                all_metrics['fname'] = []
                                all_metrics['fname'].append(fname)    
                            else:
                                all_metrics['fname'].append(fname)    
                            if 'rec_dist' not in all_metrics:
                                all_metrics['rec_dist'] = [src_dist]    
                            else:
                                all_metrics['rec_dist'].append(src_dist)   
                            if 'scene_area' not in all_metrics:
                                all_metrics['scene_area'] = [scene_area]
                            else:
                                all_metrics['scene_area'].append(scene_area)   
                            if 'loss' not in all_metrics:
                                all_metrics['loss'] = [loss]
                            else:
                                all_metrics['loss'].append(loss)

                            if 'loss_reduced' not in all_metrics:
                                all_metrics['loss_reduced'] = [loss_second[None]]
                            else:
                                all_metrics['loss_reduced'].append(loss_second[None])
                            if 'ssim' not in all_metrics:
                                all_metrics['ssim'] = [ssim[None]]
                            else:
                                all_metrics['ssim'].append(ssim[None])    
                            if 'mssim' not in all_metrics:
                                all_metrics['mssim'] = [mssim[None]]
                            else:
                                all_metrics['mssim'].append(mssim[None])
                            if 'scc' not in all_metrics:
                                all_metrics['scc'] = [scc[None]]
                            else:
                                all_metrics['scc'].append(scc[None])
                            if 'imag_grad' not in all_metrics:
                                all_metrics['imag_grad'] = [imag_grad[None]]
                            else:
                                all_metrics['imag_grad'].append(imag_grad[None])
                            if wass_2d is not None:
                                if 'wass_2d' not in all_metrics:
                                    all_metrics['wass_2d'] = [wass_2d[None]]
                                else:
                                    all_metrics['wass_2d'].append(wass_2d[None])
                            if wavelet_2d is not None:
                                #print(f'wavelet_2d is : {wavelet_2d}')
                                if isinstance(wavelet_2d, List):  # Breakdown by levels
                                    for jjj, this_wavelet_2d in enumerate(wavelet_2d):
                                        label = f'wavelet_2d/{jjj}'
                                        if label not in all_metrics:
                                            all_metrics[label] = [this_wavelet_2d[None]]
                                        else:
                                            all_metrics[label].append(this_wavelet_2d[None])
                                else:
                                    if 'wavelet_2d' not in all_metrics:
                                        all_metrics['wavelet_2d'] = [wavelet_2d[None]]
                                    else:
                                        all_metrics['wavelet_2d'].append(wavelet_2d[None])
                            if filterbank_2d is not None:
                                #print(f'filterbank_2d is : {wavelet_2d}')
                                if isinstance(filterbank_2d, List):  # Breakdown by levels
                                    for jjj, this_filterbank_2d in enumerate(filterbank_2d):
                                        label = f'filterbank_2d/{jjj}'
                                        if label not in all_metrics:
                                            all_metrics[label] = [this_filterbank_2d[None]]
                                        else:
                                            all_metrics[label].append(this_filterbank_2d[None])
                                else:
                                    if 'filterbank_2d' not in all_metrics:
                                        all_metrics['filterbank_2d'] = [filterbank_2d[None]]
                                    else:
                                        all_metrics['filterbank_2d'].append(filterbank_2d[None])
                            
                    # Reshape as batchless tensor, then log
                    # Shape is [n, n_freqs] for each parameter
                    for k,v in all_metrics.items():
                        if k in ['fname']:
                            continue
                        this_v = torch.concat(v, dim=0)
                        #print(f'this_v.sahpe {this_v.shape}')
                        all_metrics[k] = this_v
                        if k in ['scene_area', 'rec_dist']:
                            continue
                        if writer is not None and ('loss' not in k and '.' in k):  # spatial acoustical parameters
                            writer.add_scalar(f'Spatial_{subset}/{k}', this_v.mean().item(), ctr)
                        elif writer is not None and ('loss' not in k and utils.is_multislope_tag(k)):  # multislope parameters
                            writer.add_scalar(f'MultiSlope_{subset}/{k}', this_v.mean().item(), ctr)
                        elif writer is not None and ('loss' not in k and '/' not in k): # regular acoustical parameters
                            writer.add_scalar(f'Metrics_{subset}/{k}', this_v.mean().item(), ctr)
                        elif writer is not None and '/' in k:  # breakdown of image sim metrics
                            writer.add_scalar(f'ImageSim_{subset}/{k}', this_v.mean().item(), ctr)
                    if subset == 'valid':
                        writer.add_scalar(f'Loss/Valid', loss.mean().item(), ctr)
                        writer.add_scalar(f'Loss/Valid_reduced', loss_second.item(), ctr)
                        log_output_examples(ctr, writer, out, floormap, acumap, mask, floormap_processor, subset='valid', batch_id_tensorboard=batch_id_tensorboard, 
                                            n_freq_bands=len(acumap_processor.freq_bands), acumap_processor=acumap_processor, denormalizer=denormalizer, use_split_unseen_recs=use_split_by_recs)
                        if loss.mean().item() < best_loss_valid:
                            best_loss_valid = loss.mean().item()
                            print(f'>>>>>>>>> Best model detected at {ctr}')
                            save_model(model, optimizer, logdir=f'{logdir}/{run}', iteration=ctr, model_name='best')
                            print_metrics(all_metrics, logdir=f'{logdir}/{run}', iter=ctr, subset=subset)   
                            save_metrics(all_metrics, logdir=f'{logdir}/{run}', subset=subset, model_name='best')

                            # Save one batch of examples
                            save_output_examples(ctr, acuamp_output=out_denormalized.detach().cpu(), 
                                                 floormap=floormap.detach().cpu(), 
                                                 acumap_target=denormalizer(acumap).detach().cpu(), 
                                                 logdir=f'{logdir}/{run}')
                    save_metrics(all_metrics, logdir=f'{logdir}/{run}', subset=subset, model_name='last')
                    print(all_metrics.keys())

                if translator_tform is not None:
                    dataloader.dataset.augmentation_transform = translator_tform
            ctr += 1  # counter of iterations
            
    if profiling:                
        prof.stop()
    if writer is not None:
        writer.close()
    if args['wandb']:
        wandb.finish()

    model.eval()
    out = model(floormap)
    print('')
    print('outputs : {}'.format(out.detach().cpu().numpy()))
    a = out[mask].detach().cpu().numpy()
    b = acumap[mask].detach().cpu().numpy()
    print('target : {}'.format(b))
    assert np.allclose(a, b, atol=1e-4), 'Wrong outputs'

    print('Unit test completed.')

def validation_only(checkpoint_path, dataloader, dataloader_valid, net, device, 
                    floormap_processor, acumap_processor, normalizer, denormalizer,
                    stft_tform, projector, stft_encoder, 
                    use_multislope_thresholding, model_threshold_rt, model_threshold_amp,
                    remove_src_rec, use_stft_input, use_projector, use_film, use_split_by_recs,
                    writer, logdir, run, batch_id_tensorboard, 
                    augmentation_floormaps=None, use_floormap_augmentation=False, save_outputs=True):
    if save_outputs:
        ids_for_images = [0, 500, 1000, 1500]
        
    logdir = 'logging'
    model = load_model(net, checkpoint_path, logdir, device)

    use_wrong_src = False
    use_multiple_rirs = False
    use_pos_enc_src = False
    use_pos_enc_rec = use_pos_enc_src
    use_concat_stft = False

    # Validiton loop
    model.eval()
    ids_for_validaiton = get_fixed_ids_for_validation(dataloader_valid)  # these are batch ids

    if dataloader_valid.dataset.augmentation_transform is not None:
        dataloader_valid.dataset.augmentation_transform = None

    loss_f_valid = nn.L1Loss(reduction='none')
    loss_mssim = MSSIM(kernel_size=9, betas=(0.0448, 0.2856, 0.3001, 0.2363), reduction='none').to(device)
    print(f'Validation Only')
    for loader, subset in zip([dataloader_valid], ['test']):
        all_metrics = {}
        with torch.no_grad():
            for ctr, (datum) in enumerate(loader):
                batch_id = ctr
                fname, rir, src, rec, scene, floormap, acumap, pose = expand_datum(datum, args['fmap_add_pose'])
                src, rec = src.to(device), rec.to(device)
                floormap = floormap.to(device)
                acumap = acumap.to(device)
                if pose is not None:
                    pose = pose.to(device)

                if use_floormap_augmentation:  # NOTE: test time augmentaiton, just to evaluation the robustness to floormap noise
                    floormap = augmentation_floormaps(floormap)

                if use_wrong_src:
                    # Sample another src and rec position
                    ids_aux = []
                    for ii in range(len(scene)):
                            ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(loader.dataset, scene=scene[ii]))

                    tmp_subset = torch.utils.data.Subset(loader.dataset, ids_aux)
                    tmp_subset_loader = torch.utils.data.DataLoader(tmp_subset, batch_size=batch)
                    fname_aux, _, src_aux, rec_aux, scene_aux, _, _ = next(iter(tmp_subset_loader))
                    src = src_aux.to(device)
                    rec = rec_aux.to(device)
                    assert scene_aux == scene, f'ERROR: The scene from the auxiliary rir is different'

                if use_multiple_rirs:
                    ids_aux = []
                    for ii in range(len(scene)):
                        if args['dataset'] == 'mras':
                            ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(loader.dataset.dataset, scene=scene[ii]))
                        else:    
                            ids_aux.append(GWA_sampling_analysis.uniform_sampling_same_scene(loader.dataset, scene=scene[ii]))

                    tmp_subset = torch.utils.data.Subset(loader.dataset, ids_aux)
                    tmp_subset_loader = torch.utils.data.DataLoader(tmp_subset, batch_size=batch)
                    fname_aux, rir_aux, src_aux, rec_aux, scene_aux, _, _ = next(iter(tmp_subset_loader))
                    src_aux = src_aux.to(device)
                    rec_aux = rec_aux.to(device)
                    assert scene_aux == scene, f'ERROR: The scene from the auxiliary rir is different'

                if not remove_src_rec:
                    floormap = floormap_processor.add_src_rec(floormap, src, rec)
                if use_pos_enc_src:
                    floormap = floormap_processor.add_positional_encoding(floormap, src, 'src')
                if use_pos_enc_rec:
                    floormap = floormap_processor.add_positional_encoding(floormap, rec, 'rec')
                if args['fmap_add_pose']:
                    floormap = floormap_processor.add_pose_vector(floormap, pose)
                acumap = normalizer(acumap).to(torch.float32)
    
                if use_stft_input:
                    stft = stft_tform(rir.to(device))
                    floormap = torch.concat([floormap, stft], dim=-3)
                    if len(floormap_processor.channel_names) < floormap.shape[-3]:
                        for j in range(stft.shape[-3]):
                            floormap_processor.channel_names.append(f'stft{j}')
                    
                if use_projector:
                    # Replace the src with a 2d linear projection
                    src_projected = projector(src)
                    src_projected = src_projected.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)  # Only if we need to reshape stuff
                    id = floormap_processor.channel_names.index('source')
                    floormap[..., id, :, :] = src_projected[..., 0, :, :]

                # RIR encodfing for multiptle RIRs
                if use_multiple_rirs:
                    stft = stft_tform(rir.to(device))
                    stft_aux = stft_tform(rir_aux.to(device))

                    # Concat src and rec pos to RIR encoder
                    #floormap = floormap_processor.add_src_rec(floormap, src_aux, rec_aux)
                    if use_pos_enc_src:
                        stft = floormap_processor.add_positional_encoding(stft, src, 'src')
                        stft = floormap_processor.add_positional_encoding(stft, rec, 'rec')
                        stft_aux = floormap_processor.add_positional_encoding(stft_aux, src_aux, 'src')
                        stft_aux = floormap_processor.add_positional_encoding(stft_aux, rec_aux, 'rec')
                    else:
                        stft = floormap_processor.add_src_rec(stft, src, rec)
                        stft_aux = floormap_processor.add_src_rec(stft_aux, src_aux, rec_aux)

                    if use_stft_input:
                        floormap = torch.concat([floormap, stft_aux], dim=-3)
                        if len(floormap_processor.channel_names) < floormap.shape[-3]:
                            for j in range(stft.shape[-3]):
                                floormap_processor.channel_names.append(f'stft_aux{j}')

                    conditioning = stft_encoder(stft)
                    conditioning_aux = stft_encoder(stft_aux)
                    conditioning = torch.concat([conditioning, conditioning_aux], dim=-3)

                if use_film:
                    stft = stft_tform(rir.to(device))
                    conditioning = stft_encoder(stft)

                if use_concat_stft:
                    stft = stft_tform(rir.to(device))
                    #stft = floormap_processor.add_positional_encoding(stft, src.to(device), 'src')
                    #stft = floormap_processor.add_positional_encoding(stft, rec.to(device), 'rec')
                    conditioning = stft_encoder(stft)
                    
                if not use_film and not use_multiple_rirs and not use_concat_stft:
                    out = model(floormap)
                else:
                    out = model(floormap, conditioning)

                # Multi-slope mode, we add threshold to some outputs
                if use_multislope_thresholding:
                    out = model_threshold_rt(out)
                    out = model_threshold_amp(out)

                if not use_split_by_recs:
                    mask = torch.logical_not(torch.isnan(acumap)) 
                else:
                    keys = []
                    for iii, k in enumerate(fname):  #fname is tuple of fnames, due to collate
                        tmp = k.split('/')  # 'office_0/10_39.wav' -> office_0, 10_39.wav
                        tmp2 = tmp[-1].split('_')  # 10_39.wav -> 10, 39.wav
                        k = f'{tmp[0]}/{tmp2[0]}'  # office_0/10
                        keys.append(k)
                    if epoch == 0:
                        for iii, k in enumerate(keys):  #fname is tuple of fnames, due to collate
                            tmp = k.split('/')  # 'office_0/10_39.wav' -> office_0, 10_39.wav
                            tmp2 = tmp[-1].split('_')  # 10_39.wav -> 10, 39.wav
                            k = f'{tmp[0]}/{tmp2[0]}'  # office_0/10
                            if k not in masks_recs_test:
                                tmp_mask_train, tmp_mask_test = get_split_mask_acuamp_by_recs(acumap=acumap[iii], 
                                                                                              method=args['use_within_room_generalization_method'],
                                                                                              uniform_threshold=args['use_within_room_generalization_threshold'])
                                masks_recs_train[k] = tmp_mask_train
                                masks_recs_test[k] = tmp_mask_test
                    if subset == 'train':
                        pre_mask = [masks_recs_train[k] for k in keys]
                    elif subset == 'valid':
                        pre_mask = [masks_recs_test[k] for k in keys]
                    pre_mask = torch.stack(pre_mask, dim=0).to(device)
                    mask = torch.logical_not(torch.isnan(acumap)) * pre_mask

                #loss = loss_f_valid(out[mask], acumap[mask])   # this removes the shape of the tensors
                ############## print(f'maskkkkkk {mask.shape}')
                loss = loss_f_valid(out * mask, acumap * mask)
                ############## print(f'loss {loss.shape}')
                denom = mask.nansum(dim=(-2,-1))
                loss = loss.nansum(dim=(-2,-1)) / denom  # pixelwise mean, for valid pixels only, [batch, channels], there should not be any nans here

                if torch.any(torch.isnan(loss)):
                    print('WARNING: NaN detected in loss. ')
                    print(fname)
                    print(f'out[mask].shape {out[mask].shape}')
                    print(f'acumap[mask].shape {acumap[mask].shape}')

                    nan_count_1 = torch.sum(torch.isnan(out[mask])).item()
                    nan_count_2 = torch.sum(torch.isnan(acumap[mask])).item()
                    print(f'NaNs out: {nan_count_1}')
                    print(f'NaNs acumap: {nan_count_2}')
                    if post_mask is not None:
                        nan_count_1 = torch.sum(torch.isnan(out[post_mask])).item()
                        nan_count_2 = torch.sum(torch.isnan(acumap[post_mask])).item()
                        print(f'NaNs out: {nan_count_1}')
                        print(f'NaNs acumap: {nan_count_2}')

                    out[~mask] = 0.0
                    where_nans = torch.where(torch.any(torch.isnan(out),dim=[-3,-2,-1]))
                    print(f'Batch ids where nans : {where_nans}')

                    raise ValueError('ERROR, there are anans in the out. Giving up')

                ### loss = loss.nanmean(dim=-2).nanmean(-1)   # Pixelwise mean, [batch, channels], ignore NaNs, but uses wrong denominator (all pixels)
                ############## print(f'loss.shape {loss.shape}')
                ############## print(f'loss nay nan {torch.any(torch.isnan(loss))}')
                ############## loss = loss.view(out.shape[0], out.shape[1], -1).mean(dim=-1)  # Pixelwise mean, [batch, channels]
                out_denormalized = denormalizer(out)   # UPDATE Oct 01, paper submission. I am not sure if these should be her or not. DOUBLE CHECk
                # UPDATE June 16 2025: now I have reenable this, because i am getting large errors

                # Loss with other masking
                loss_second = loss_f_valid(out[mask], acumap[mask])   # this removes the shape of the tensors
                ##############print(f'loss_second.shape   {loss_second.shape}')
                loss_second = loss_second.mean()  # float
                ##############print(f'loss_second.mean.shape   {loss_second}')


                # NEW SSIM metric for similarity between image-like data
                ssim = compute_ssim(out * mask, acumap * mask) 

                # New MSSIM
                tmp_acumap = acumap.clone()
                tmp_acumap[~mask] = 0.0  # hack to avoid problems with nans
                mssim = loss_mssim(out * mask, tmp_acumap * mask)
                mssim = mssim.nanmean()

                # New SCC
                scc = compute_loss_scc(out * mask, tmp_acumap * mask)
                #scc = compute_loss_scc(out[mask], acumap[mask])

                # New ImageGradients
                imag_grad = compute_loss_gradients(out * mask, tmp_acumap * mask)

                # New metrics for image simalarity
                # 2d wassertein
                if batch_id in ids_for_validaiton:
                    tmp_out = out.clone()
                    tmp_out[~mask] = 0.0 # hack to avoid problems with nans
                    tmp_acumap = acumap.clone()
                    tmp_acumap[~mask] = 0.0  # hack to avoid problems with nans
                    #wass_2d = compute_2d_wass_distance(out, tmp_acumap)
                    loss_fn = losses.Wasserstein2dLoss()
                    wass_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
                    loss_fn = losses.WaveletTransformLoss(return_breakdown=True)
                    wavelet_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
                    loss_fn = losses.Filterbank2dLoss(return_breakdown=True)
                    filterbank_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
                else:
                    wass_2d = None
                    wavelet_2d = None
                    filterbank_2d = None

                # Additional labels to analyze errors
                src_dist = torch.sqrt(torch.sum((src - rec) ** 2, dim=1))  # Eucldean distance between src and reference receiver
                scene_area = torch.sum(floormap[..., 1, :, :], dim=(-2,-1))  # Scene area 

                # UPDATE June 16 2025: disabled this
                #out_denormalized = out # UPDATE Oct 01, paper submission. I am not sure if these should be her or not. DOUBLE CHECk
                # Becase I was gettinng very bad errors even with train data, probably due to double denormalization
                error_metrics = evaluation(out_denormalized, denormalizer(acumap), mask, acumap_processor=acumap_processor)
                for k,v in error_metrics.items():
                    #print(f'k, v: {k} : {v}')

                    # Parse keys for spatial metrics only
                    if '.' in k:
                        tmp_dir, tmp_param = k.split('/')  # 0.0/c50 --> 0.0, c50
                        k = f'{tmp_param}/{tmp_dir}'  # c50/0.0
                    if k not in all_metrics:
                        all_metrics[k] = [v]
                    else:
                        all_metrics[k].append(v)    
                
                # append fname so I can find some good examples later
                if 'fname' not in all_metrics:
                    all_metrics['fname'] = []
                    all_metrics['fname'].append(fname)    
                else:
                    all_metrics['fname'].append(fname)    
                if 'rec_dist' not in all_metrics:
                    all_metrics['rec_dist'] = [src_dist]    
                else:
                    all_metrics['rec_dist'].append(src_dist)   
                if 'scene_area' not in all_metrics:
                    all_metrics['scene_area'] = [scene_area]
                else:
                    all_metrics['scene_area'].append(scene_area)   
                if 'loss' not in all_metrics:
                    all_metrics['loss'] = [loss]
                else:
                    all_metrics['loss'].append(loss)

                if 'loss_reduced' not in all_metrics:
                    all_metrics['loss_reduced'] = [loss_second[None]]
                else:
                    all_metrics['loss_reduced'].append(loss_second[None])
                if 'ssim' not in all_metrics:
                    all_metrics['ssim'] = [ssim[None]]
                else:
                    all_metrics['ssim'].append(ssim[None])    
                if 'mssim' not in all_metrics:
                    all_metrics['mssim'] = [mssim[None]]
                else:
                    all_metrics['mssim'].append(mssim[None])
                if 'scc' not in all_metrics:
                    all_metrics['scc'] = [scc[None]]
                else:
                    all_metrics['scc'].append(scc[None])
                if 'imag_grad' not in all_metrics:
                    all_metrics['imag_grad'] = [imag_grad[None]]
                else:
                    all_metrics['imag_grad'].append(imag_grad[None])
                if wass_2d is not None:
                    if 'wass_2d' not in all_metrics:
                        all_metrics['wass_2d'] = [wass_2d[None]]
                    else:
                        all_metrics['wass_2d'].append(wass_2d[None])
                if wavelet_2d is not None:
                    #print(f'wavelet_2d is : {wavelet_2d}')
                    if isinstance(wavelet_2d, List):  # Breakdown by levels
                        for jjj, this_wavelet_2d in enumerate(wavelet_2d):
                            label = f'wavelet_2d/{jjj}'
                            if label not in all_metrics:
                                all_metrics[label] = [this_wavelet_2d[None]]
                            else:
                                all_metrics[label].append(this_wavelet_2d[None])
                    else:
                        if 'wavelet_2d' not in all_metrics:
                            all_metrics['wavelet_2d'] = [wavelet_2d[None]]
                        else:
                            all_metrics['wavelet_2d'].append(wavelet_2d[None])
                if filterbank_2d is not None:
                    #print(f'filterbank_2d is : {wavelet_2d}')
                    if isinstance(filterbank_2d, List):  # Breakdown by levels
                        for jjj, this_filterbank_2d in enumerate(filterbank_2d):
                            label = f'filterbank_2d/{jjj}'
                            if label not in all_metrics:
                                all_metrics[label] = [this_filterbank_2d[None]]
                            else:
                                all_metrics[label].append(this_filterbank_2d[None])
                    else:
                        if 'filterbank_2d' not in all_metrics:
                            all_metrics['filterbank_2d'] = [filterbank_2d[None]]
                        else:
                            all_metrics['filterbank_2d'].append(filterbank_2d[None])
                
        # Reshape as batchless tensor, then log
        # Shape is [n, n_freqs] for each parameter
        for k,v in all_metrics.items():
            if k in ['fname']:
                continue
            this_v = torch.concat(v, dim=0)
            #print(f'this_v.sahpe {this_v.shape}')
            all_metrics[k] = this_v
            if k in ['scene_area', 'rec_dist']:
                continue
            if writer is not None and ('loss' not in k and '.' in k):  # spatial acoustical parameters
                writer.add_scalar(f'Spatial_{subset}/{k}', this_v.mean().item(), ctr)
            elif writer is not None and ('loss' not in k and utils.is_multislope_tag(k)):  # multislope parameters
                writer.add_scalar(f'MultiSlope_{subset}/{k}', this_v.mean().item(), ctr)
            elif writer is not None and ('loss' not in k and '/' not in k): # regular acoustical parameters
                writer.add_scalar(f'Metrics_{subset}/{k}', this_v.mean().item(), ctr)
            elif writer is not None and '/' in k:  # breakdown of image sim metrics
                writer.add_scalar(f'ImageSim_{subset}/{k}', this_v.mean().item(), ctr)
        if subset == 'test':
            if writer is not None:
                writer.add_scalar(f'Loss/Test', loss.mean().item(), ctr)
                writer.add_scalar(f'Loss/Test_reduced', loss_second.item(), ctr)
                log_output_examples(ctr, writer, out, floormap, acumap, mask, floormap_processor, subset='valid', batch_id_tensorboard=batch_id_tensorboard, 
                                    n_freq_bands=len(acumap_processor.freq_bands), acumap_processor=acumap_processor, denormalizer=denormalizer, use_split_unseen_recs=use_split_by_recs)
        #save_metrics(all_metrics, logdir=f'{logdir}/{run}', subset=subset, model_name='test')

        id_for_plot = 0
        # more params
        cbar_labels = ['C50 [dB]', 'T30 [s]', 'DRR [dB]', 'EDT [s]']
        freqs_toplot = [1,3,5]  

        # spatial
        id_for_plot = 4
        cbar_labels = ['C50 [dB]', 'C50 [dB]', 'C50 [dB]', 'C50 [dB]', 'C50 [dB]']
        freqs_toplot = [0,1,2]  

        print(f'HEYYYY, I am about to plot {fname}')    
        print(f'acumap {acumap.shape}')    
        plot_and_save_acumap(denormalizer(acumap)[id_for_plot].detach().cpu(), src[id_for_plot].detach().cpu(), acumap_processor, freqs_toplot=freqs_toplot, cbar_labels=cbar_labels, fname='targets', scene_name=f'{fname[id_for_plot]}_targets')
        tmp_map = out_denormalized  # * mask[id_for_plot].detach().cpu()
        tmp_map[~mask] = np.nan

        error_map = (denormalizer(acumap) - out_denormalized).abs()  # this works
        #error_map /= denormalizer(acumap).abs()  # not needed


        #error_map = denormalizer(acumap - out).abs()
        #error_map[~mask] = np.nan
        plot_and_save_acumap(tmp_map[id_for_plot].detach().cpu() , src[id_for_plot].detach().cpu(), acumap_processor, freqs_toplot=freqs_toplot, cbar_labels=cbar_labels, fname='out', scene_name=f'{fname[id_for_plot]}_out')
        plot_and_save_acumap(error_map[id_for_plot].detach().cpu() , src[id_for_plot].detach().cpu(), acumap_processor, freqs_toplot=freqs_toplot, cbar_labels=cbar_labels, fname='error', scene_name=f'{fname[id_for_plot]}_modelerror')
        print(all_metrics.keys())
        logdir = 'logging'
        print_metrics(all_metrics, logdir=f'{logdir}/{run}', iter=ctr, subset=subset)   
        save_metrics(all_metrics, logdir=f'{logdir}/{run}', subset=subset, model_name='last')
        save_output_examples(ctr, acuamp_output=out_denormalized.detach().cpu(), 
                             floormap=floormap.detach().cpu(), 
                             acumap_target=denormalizer(acumap).detach().cpu(), 
                             logdir=f'{logdir}/{run}')

def evaluation(estimate: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, acumap_processor: nn.Module = None) -> Dict:
    """ Evaluates the errors for metrics. The tensors should NOT be normalized. """
    if acumap_processor is None:
        tags = ['c50', 't30', 'drr']
        dist = ['l1', 'mape', 'l1']
        freqs = [250, 1000, 4000]
    else:
        tags = acumap_processor.parameters
        dist = acumap_processor.distances
        freqs = acumap_processor.freq_bands
    if len(estimate.shape) > 3:
        batch = estimate.shape[0]
    else:
        batch - None

    assert estimate.shape == target.shape, f'ERROR, the shape of estimate {estimate.shape} and target {target.shape} should match.'
    assert len(estimate.shape) >= 3, f'ERROR, I was expecting 3d of 4d tensors (image-like with or without batch), but got {estimate.shape}'
    assert estimate.shape[-3] == len(tags) * len(freqs) and estimate.shape[-3] == len(freqs) * len(dist), f'ERROR, the number ofr channels in estimate {estimate.shape} should match the selected evalaiton metrics {tags} at frequenies {freqs} '
    assert estimate.shape == mask.shape, f'ERROR, the shape of estimate {estimate.shape} and mask {mask.shape} should match'
    assert len(tags) == len(dist), f'ERROR, tags and distances do not match. tags:{tags} \t dist:{dist}'

    error = {}  
    n_freqs = len(freqs)
    for ii, tag in enumerate(tags):
        ch = ii * n_freqs
        #this_estimate = estimate[..., ch:ch+n_freqs, :, :][mask[..., ch:ch+n_freqs, :, :]]
        #this_target = target[..., ch:ch+n_freqs, :, :][mask[..., ch:ch+n_freqs, :, :]]
        this_estimate = estimate[..., ch:ch+n_freqs, :, :] * mask[..., ch:ch+n_freqs, :, :]
        this_target = target[..., ch:ch+n_freqs, :, :] * mask[..., ch:ch+n_freqs, :, :]

        # Reshape with batch for better analysis
        if False and batch is not None:
            this_estimate = this_estimate.view(batch, n_freqs, -1)
            this_target = this_target.view(batch, n_freqs, -1)

        if dist[ii] == 'mape':
            tmp = pytorch_acoustics.mean_absolute_proportional_error(this_estimate, this_target, reduction=False)
        elif dist[ii] == 'mape_nozero':
            tmp = pytorch_acoustics.mean_absolute_proportional_error_no_zero(this_estimate, this_target, reduction=False)
            if tmp is None: continue  # this means there are no non zero targets in this batch
        elif dist[ii] == 'mape_threshold':
            tmp = pytorch_acoustics.mean_absolute_proportional_error_threshold(this_estimate, this_target, reduction=False)
        elif dist[ii] == 'male':
            tmp = pytorch_acoustics.mean_absolute_log_error(this_estimate, this_target, reduction=False)
        elif dist[ii] == 'l1':
            fn = torch.nn.L1Loss(reduction='none')
            tmp = fn(this_estimate, this_target)
        else:
            raise ValueError(f'ERROR: Unrecognized distance fuction {dist[ii]}')
        ######print(f'yoloooo {tmp.shape}')
        try:
            if dist[ii] == 'mape_nozero':
                denom = (tmp != 0).sum()
                error[tag] = tmp / denom
            else:
                error[tag] = tmp.nanmean(dim=-2).nanmean(dim=-1)  # pizelwise mean
        except Exception as e:
            print(f'tag {tag}')
            print(f'tmp.shape {tmp.shape}')
            raise e
        ######print(f'yolooo after mean {error[tag].shape}')

    return error

def get_directional_metrics(dataset_name: str, rir: torch.Tensor, dataset_fs: int, metrics: Dict, use_decayfitnet=False, device='cpu'):
    # Computes the directional metrics, by first beamforming the rir to the same angles as during the dataset preprocessing
    # Directional metrics
    if dataset_name in ['replica', 'mp3d']:
        rotate_alternative_convention = False
    else:
        rotate_alternative_convention = True
    W, angles = pytorch_acoustics.get_beamforming_matrix(w_pattern='maxre', sph_order=2, polygon_sides=6, 
                                                         rotate_alternative_convention=rotate_alternative_convention)
    angles = torch.rad2deg(angles)
    rir_processed = torch.einsum('it,di->dt', rir.to(torch.float64).to(device), W.to(device)).cpu()

    for jj, _ in enumerate(range(rir_processed.shape[0])):
        if use_decayfitnet:
            this_metrics = pytorch_acoustics.get_metrics_decayfitnet(rir_processed[jj,:], dataset_fs, 
                                                                    remove_direct_sound=True, fadeout_length=0.5)
        else:
            this_metrics = pytorch_acoustics.get_metrics(rir_processed[jj,:], dataset_fs)
        metrics[f'metrics_{angles[jj]}'] = this_metrics
    return metrics

def get_directional_metrics_from_omni(dataset_name: str, rir: torch.Tensor, dataset_fs: int, metrics: Dict, use_decayfitnet=False):
    # Computes the directional metrics, from the omni channel only, so no beamforming
    if dataset_name in ['replica', 'mp3d']:
        rotate_alternative_convention = False
        fadeout_length = 0.5
    else:
        rotate_alternative_convention = True
        fadeout_length = 0.1
    W, angles = pytorch_acoustics.get_beamforming_matrix(w_pattern='maxre', sph_order=2, polygon_sides=6, 
                                                         rotate_alternative_convention=rotate_alternative_convention)
    angles = torch.rad2deg(angles)

    for jj, _ in enumerate(range(len(angles))):
        if use_decayfitnet:
            this_metrics = pytorch_acoustics.get_metrics_decayfitnet(rir[0,:], dataset_fs, 
                                                                    remove_direct_sound=True, fadeout_length=fadeout_length)
        else:
            this_metrics = pytorch_acoustics.get_metrics(rir[0,:], dataset_fs)
        metrics[f'metrics_{angles[jj]}'] = this_metrics
    return metrics

@torch.no_grad()
def compute_baselines(dataset, device, normalizer, denormalizer, method='sample_acumaps_scene', dataset_name='replica',
                      use_split_by_recs=False, floormap_processor=None, acumap_processor=None, random_sample_dataset=False,
                      use_multislope_thresholding=False, model_threshold_rt=None, model_threshold_amp=False,
                      writer=None, subset='valid', logdir='trash', use_decay_fitnet=False, do_directional=False, do_directional_omni=True,
                      dataset_valid_omni=None, dataset_inference_only=None):
    baseline_methods = ['average_rir_all', 'average_rir_same_scene', 'average_acumap_scene', 'same_rir_NN',  'average_rir_same_scene_and_source', 'sample_acumaps_scene']
    ids_for_validaiton = get_fixed_ids_for_validation(dataset, sample_size=1000)  # these are dataset ids, not batch ids
    assert (do_directional and not do_directional_omni) or (do_directional_omni and not do_directional) or (not do_directional and not do_directional_omni), f'ERROR, cannot do directional and omni at the same time'
    
    if method not in baseline_methods:
        raise ValueError(f'ERROR, the baseline method {method} is not supported.')

    fnames_with_errors = []
    n_average_rirs = np.minimum(100, len(dataset))
    tags = acumap_processor.parameters  # ['c50', 't30', 'drr']
    freqs = acumap_processor.freq_bands  # [250, 1000, 4000]
    base_freqs = [125, 250, 500, 1000, 2000, 4000, 8000]
    freqs_ind = [base_freqs.index(value) for value in freqs]

    # Precompute targets if needed
    if method == 'average_rir_all':
        average_rir = GWA_sampling_analysis.average_rir_all(dataset, n_rirs=n_average_rirs*5)
    elif method == 'average_rir_same_scene':
        hat_rirs_per_scene = {}
        for scene in dataset.scenes:  # Precompute RIRs for each scene
            tmp_rir = GWA_sampling_analysis.average_rir_same_scene(dataset, scene, n_rirs=n_average_rirs)
            hat_rirs_per_scene[scene] = tmp_rir
    elif method == 'average_acumap_scene':
        hat_acumap_per_scene = {}
        for scene in dataset.scenes:  # Precompute mean acumap for each scene
            if dataset_valid_omni is None:
                tmp_acumap = GWA_sampling_analysis.sample_acumap_same_scene(dataset, scene, n_srcs=n_average_rirs, reduction='mean')    
            else:
                tmp_acumap = GWA_sampling_analysis.sample_acumap_same_scene(dataset_valid_omni, scene, n_srcs=n_average_rirs, reduction='mean')    
                tmp_acumap = tmp_acumap[..., 0:3, :, :]
                tmp_acumap = torch.cat([tmp_acumap]*5, dim=-3)  # we repeat the c50 channels to cover all directions
            hat_acumap_per_scene[scene] = tmp_acumap
    elif method == 'sample_acumaps_scene':
        hat_acumap_per_scene = {}
        for scene in dataset.scenes:  # Get a sample of acumaps for each scene
            if dataset_valid_omni is None:
                tmp_acumap = GWA_sampling_analysis.sample_acumap_same_scene(dataset, scene, n_srcs=n_average_rirs, reduction='none')
            else:
                tmp_acumap = GWA_sampling_analysis.sample_acumap_same_scene(dataset_valid_omni, scene, n_srcs=n_average_rirs, reduction='none')
                tmp_acumap = tmp_acumap[..., 0:3, :, :]
                tmp_acumap = torch.cat([tmp_acumap]*5, dim=-3)  # we repeat the c50 channels to cover all directions
            hat_acumap_per_scene[scene] = tmp_acumap

    # Compute metrics
    loss_f_valid = nn.L1Loss(reduction='none')
    print(f'Baseline {method}')
    all_metrics = {}

    if dataset_inference_only is not None:  # Only for inference
        a = len(dataset_inference_only) - 1
        id_for_plot = [int(a)]
        # Note, here we do not have batche, so we are plotting is the last instance in the dataset
        dataset_subset_sample = torch.utils.data.Subset(dataset_inference_only, id_for_plot)
        print('Inference ONLY')
    elif random_sample_dataset:   # Take a small sample of the valdiation set, just to compute baselines faster
        n_samples = 80000 if dataset_name == 'replica' else 200000
        ids = np.random.choice(len(dataset), n_samples, replace=False)
        dataset_subset_sample = torch.utils.data.Subset(dataset, ids)
        print('=========================================================================')
        print(f'Taking a random sample of the validaiton dataset, of size {n_samples}')
        print()
    else:
        dataset_subset_sample = dataset

    print(f'Beggining baseline for loops for: {method}')
    print(f'Dataset len is: {len(dataset_subset_sample)}')

    # Save some files to dump the full results at the end
    # ths is mostly useful to get plots 
    acumaps_to_save = []
    outputs_to_save = []
    errors_to_save = []
    n_to_save = 100
    ids_to_save = np.arange(len(dataset_subset_sample) - n_to_save, len(dataset_subset_sample))

    if dataset_inference_only is not None:
        print(f'dataset_inference_only = {len(dataset_inference_only)}')
    for ctr, (datum) in enumerate(tqdm(dataset_subset_sample)):
        fname, rir, src, rec, scene, floormap, acumap, pose = expand_datum(datum, args['fmap_add_pose'])
        assert acumap.shape[-3] == len(tags) * len(freqs), f'ERROR, the shape of acumaps does not match the target metrics {acumap.shape} tags={tags}, freqs={freqs}'
        src, rec = src.to(device), rec.to(device)
        floormap = floormap.to(device)
        acumap = acumap.to(device)
        acumap = normalizer(acumap).to(torch.float32)
        if pose is not None:
            pose = pose.to(device)
            
        # Compute estimate acumap with selected baseline
        if method == 'average_rir_all':
            hat_rir = average_rir
            hat_acumap = torch.ones_like(acumap)
            if use_decay_fitnet:  # This crashes in triton login3, due to a missing library, so we disable it for now
                metrics_estimate = pytorch_acoustics.get_metrics_decayfitnet(hat_rir[0,:], dataset.fs, device=device)
            else:
                metrics_estimate = pytorch_acoustics.get_metrics(hat_rir[0,:], dataset.fs)

            # Append directional metrics if needed
            if do_directional:
                metrics_estimate = get_directional_metrics(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    
            elif do_directional_omni:
                metrics_estimate = get_directional_metrics_from_omni(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    
            
            ctr_inner = 0
            for tag in tags:  
                for freq in freqs_ind:
                    if '/' in tag:  # directional metrics
                        # f'metrics_{angles[jj]}
                        #print(metrics.keys())
                        this_acu_param_dir, this_acu_param_label = tag.split('/')  # 0.0/c50 --> 0.0, c50
                        #print(this_acu_param_dir)
                        #print(this_acu_param_label)
                        this_metric = metrics_estimate[f'metrics_{this_acu_param_dir}'][this_acu_param_label][freq]  
                    else:
                        this_metric = metrics_estimate[tag][freq]
                    this_metric = torch.tensor(this_metric).view(1,1).to(device)
                    hat_acumap[ctr_inner, :, :] *= this_metric
                    ctr_inner += 1
            hat_acumap = normalizer(hat_acumap)
        elif method == 'average_rir_same_scene':
            hat_rir = hat_rirs_per_scene[scene]  
            hat_acumap = torch.ones_like(acumap)
            if use_decay_fitnet:  # This crashes in triton login3, due to a missing library, so disable
                metrics_estimate = pytorch_acoustics.get_metrics_decayfitnet(hat_rir[0,:], dataset.fs)
            else:
                metrics_estimate = pytorch_acoustics.get_metrics(hat_rir[0,:], dataset.fs)

            # Append directional metrics if needed
            if do_directional:
                metrics_estimate = get_directional_metrics(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    
            elif do_directional_omni:
                metrics_estimate = get_directional_metrics_from_omni(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    

            ctr_inner = 0
            for tag in tags:
                for freq in freqs_ind:
                    if '/' in tag:  # directional metrics
                        # f'metrics_{angles[jj]}
                        #print(metrics.keys())
                        this_acu_param_dir, this_acu_param_label = tag.split('/')  # 0.0/c50 --> 0.0, c50
                        #print(this_acu_param_dir)
                        #print(this_acu_param_label)
                        this_metric = metrics_estimate[f'metrics_{this_acu_param_dir}'][this_acu_param_label][freq]  
                    else:
                        this_metric = metrics_estimate[tag][freq]
                    this_metric = torch.tensor(this_metric).view(1,1).to(device)
                    hat_acumap[ctr_inner, :, :] *= this_metric
                    ctr_inner += 1
            hat_acumap = normalizer(hat_acumap)
        elif method == 'average_acumap_scene':
            hat_acumap = hat_acumap_per_scene[scene].to(device)
            hat_acumap = normalizer(hat_acumap)
        elif method == 'sample_acumaps_scene':
            acumaps = hat_acumap_per_scene[scene]
            rand_id = torch.randint(low=0, high=acumaps.shape[0], size=(1,)).item()  # Sample 1 map from the candidates
            hat_acumap = normalizer(acumaps[rand_id].to(device))
        elif method == 'same_rir_NN':
            hat_rir = rir
            while not utils.validate_audio(hat_rir):
                # This rir is wrong, so lets grab any other rir
                id = np.random.choice(np.arange(dataset), size=1, replace=False)
                _, rir, _, _, _, _, _ = dataset[id]
                hat_rir = rir

            hat_acumap = torch.ones_like(acumap)
            try:
                if use_decay_fitnet:  # This crashes in triton login3, due to a missing library, so disable
                    metrics_estimate = pytorch_acoustics.get_metrics_decayfitnet(hat_rir[0,:], dataset.fs)
                else:
                    metrics_estimate = pytorch_acoustics.get_metrics(hat_rir[0,:], dataset.fs)

                # Append directional metrics if needed
                if do_directional:
                    metrics_estimate = get_directional_metrics(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    
                elif do_directional_omni:
                    metrics_estimate = get_directional_metrics_from_omni(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    
            except Exception as e:
                # Some RIRs are wrong, e.g. pure silence. So we ignore those.
                # In the end, I think its not a huge problem, as long as they are not too many
                fnames_with_errors.append(fname)
                print(f'ERROR in {fname}')
                print()
                continue

            ctr_inner = 0
            for tag in tags:
                for freq in freqs_ind:
                    if '/' in tag:  # directional metrics
                        # f'metrics_{angles[jj]}
                        #print(metrics_estimate.keys())
                        this_acu_param_dir, this_acu_param_label = tag.split('/')  # 0.0/c50 --> 0.0, c50
                        #print(this_acu_param_dir)
                        #print(this_acu_param_label)
                        this_metric = metrics_estimate[f'metrics_{this_acu_param_dir}'][this_acu_param_label][freq]  
                    else:
                        this_metric = metrics_estimate[tag][freq]
                    this_metric = torch.tensor(this_metric).view(1,1).to(device)
                    hat_acumap[ctr_inner, :, :] *= this_metric
                    ctr_inner += 1
            try:
                hat_acumap = normalizer(hat_acumap)
            except Exception as e:
                # Some RIRs are wrong, e.g. pure silence. So we ignore those.
                # In the end, I think its not a huge problem, as long as they are not too many
                fnames_with_errors.append(fname)
                print(f'ERROR in {fname}')
                print()
                continue
                # raise e
        elif method == 'average_rir_same_scene_and_source':
            src_id, rec_id = dataset.parse_fname(fname)
            hat_rir = GWA_sampling_analysis.average_rir_same_scene_and_source(dataset, scene, src_id=src_id, n_recs=n_average_rirs)
            hat_acumap = torch.ones_like(acumap)
            if use_decay_fitnet:  # This crashes in triton login3, due to a missing library, so disable
                metrics_estimate = pytorch_acoustics.get_metrics_decayfitnet(hat_rir[0,:], dataset.fs)
            else:
                metrics_estimate = pytorch_acoustics.get_metrics(hat_rir[0,:], dataset.fs)
            
            # Append directional metrics if needed
            if do_directional:
                metrics_estimate = get_directional_metrics(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    
            elif do_directional_omni:
                metrics_estimate = get_directional_metrics_from_omni(dataset_name, hat_rir, dataset.fs, metrics=metrics_estimate)    
            
            ctr_inner = 0
            for tag in tags:
                for freq in freqs_ind:
                    if '/' in tag:  # directional metrics
                        # f'metrics_{angles[jj]}
                        #print(metrics.keys())
                        this_acu_param_dir, this_acu_param_label = tag.split('/')  # 0.0/c50 --> 0.0, c50
                        #print(this_acu_param_dir)
                        #print(this_acu_param_label)
                        this_metric = metrics_estimate[f'metrics_{this_acu_param_dir}'][this_acu_param_label][freq]  
                    else:
                        this_metric = metrics_estimate[tag][freq]
                    this_metric = torch.tensor(this_metric).view(1,1).to(device)
                    hat_acumap[ctr_inner, :, :] *= this_metric
                    ctr_inner += 1
            hat_acumap = normalizer(hat_acumap)
        
        if not use_split_by_recs:
            mask = torch.logical_not(torch.isnan(acumap)) 
        else:
            k = fname
            tmp = k.split('/')  # 'office_0/10_39.wav' -> office_0, 10_39.wav
            tmp2 = tmp[-1].split('_')  # 10_39.wav -> 10, 39.wav
            k = f'{tmp[0]}/{tmp2[0]}'  # office_0/10
            keys = [k]
            if subset == 'train':
                pre_mask = [masks_recs_train[k] for k in keys]
            elif subset == 'valid':
                pre_mask = [masks_recs_test[k] for k in keys]
            pre_mask = torch.stack(pre_mask, dim=0).to(device)
            mask = torch.logical_not(torch.isnan(acumap)) * pre_mask

        # Multi-slope mode, we add threshold to some outputs
        if use_multislope_thresholding:
            hat_acumap = model_threshold_rt(hat_acumap)
            hat_acumap = model_threshold_amp(hat_acumap)

        loss = loss_f_valid(hat_acumap * mask, acumap * mask)
        denom = mask.nansum(dim=(-2,-1))
        loss = loss.nansum(dim=(-2,-1)) / denom  # pixelwise mean, for valid pixels only, [batch, channels], there should not be any nans here
        out_denormalized = denormalizer(hat_acumap)

        if False:
            for i in range(out_denormalized.shape[0]):
                tmp = torch.any(~torch.isnan(out_denormalized[i, :, :]))
                print(f'YOLOOOOOOO {i}    {tmp}')
            return 1

        # Loss with other masking
        loss_second = loss_f_valid(hat_acumap[mask], acumap[mask])   # this removes the shape of the tensors
        loss_second = loss_second.mean()

        # NEW SSIM metric for similarity between image-like data
        ssim = compute_ssim(hat_acumap * mask, acumap * mask) 

        # New metrics for image simalarity
        # 2d wassertein
        if ctr in ids_for_validaiton:
            this_mask = torch.logical_or(torch.isnan(acumap), torch.isnan(hat_acumap))
            tmp_out = hat_acumap.clone()
            tmp_out[this_mask] = 0.0 # hack to avoid problems with nans
            tmp_acumap = acumap.clone()
            tmp_acumap[this_mask] = 0.0  # hack to avoid problems with nans
            loss_fn = losses.Wasserstein2dLoss()
            wass_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
            loss_fn = losses.WaveletTransformLoss(return_breakdown=True)
            wavelet_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())
            loss_fn = losses.Filterbank2dLoss(return_breakdown=True)
            filterbank_2d = loss_fn(tmp_out.clone(), tmp_acumap.clone())

            if torch.any(torch.isnan(tmp_out)):
                print('NaNs detected in tmp_out')
            if torch.any(torch.isnan(tmp_acumap)):
                print('NaNs detected in tmp_acumap')
            if torch.any(torch.isnan(wass_2d)):
                print('NaNs detected in wass_2d')
        else:
            wass_2d = None
            wavelet_2d = None
            filterbank_2d = None

        # Save some tensors to dump to disk and visualize later
        if ctr in ids_to_save:
            acumaps_to_save.append(denormalizer(acumap).detach().cpu())
            outputs_to_save.append(out_denormalized.detach().cpu())

        # Additional labels to analyze errors
        if len(src.shape) < 2: src = src[None, ...]  # Add batch
        if len(rec.shape) < 2: rec = rec[None, ...]  # Add batch
        src_dist = torch.sqrt(torch.sum((src - rec) ** 2, dim=1))  # Eucldean distance between src and reference receiver
        scene_area = torch.sum(floormap[..., 1:2, :, :], dim=(-2,-1))  # Scene area 

        error_metrics = evaluation(out_denormalized[None, ...], denormalizer(acumap)[None, ...], mask[None, ...], acumap_processor=acumap_processor)
        for k,v in error_metrics.items():
            #print(f'k, v: {k} : {v}')
            # Parse keys for spatial metrics only
            if '.' in k:
                tmp_dir, tmp_param = k.split('/')  # 0.0/c50 --> 0.0, c50
                k = f'{tmp_param}/{tmp_dir}'  # c50/0.0
            if k not in all_metrics:
                all_metrics[k] = [v]
            else:
                all_metrics[k].append(v)     
        if 'fname' not in all_metrics:
            all_metrics['fname'] = []
            all_metrics['fname'].append(fname)    
        else:
            all_metrics['fname'].append(fname)   
        if 'rec_dist' not in all_metrics:
            all_metrics['rec_dist'] = [src_dist]    
        else:
            all_metrics['rec_dist'].append(src_dist)   
        if 'scene_area' not in all_metrics:
            all_metrics['scene_area'] = [scene_area]
        else:
            all_metrics['scene_area'].append(scene_area)   

        if 'loss' not in all_metrics:
            all_metrics['loss'] = [loss]
        else:
            all_metrics['loss'].append(loss)

        if 'loss_reduced' not in all_metrics:
            all_metrics['loss_reduced'] = [loss_second[None]]
        else:
            all_metrics['loss_reduced'].append(loss_second[None])
        if 'ssim' not in all_metrics:
            all_metrics['ssim'] = [ssim[None]]
        else:
            all_metrics['ssim'].append(ssim[None])

        if wass_2d is not None:
            if 'wass_2d' not in all_metrics:
                all_metrics['wass_2d'] = [wass_2d[None]]
            else:
                all_metrics['wass_2d'].append(wass_2d[None])
        if wavelet_2d is not None:
            #print(f'wavelet_2d is : {wavelet_2d}')
            if isinstance(wavelet_2d, List):  # Breakdown by levels
                for jjj, this_wavelet_2d in enumerate(wavelet_2d):
                    label = f'wavelet_2d/{jjj}'
                    if label not in all_metrics:
                        all_metrics[label] = [this_wavelet_2d[None]]
                    else:
                        all_metrics[label].append(this_wavelet_2d[None])
            else:
                if 'wavelet_2d' not in all_metrics:
                    all_metrics['wavelet_2d'] = [wavelet_2d[None]]
                else:
                    all_metrics['wavelet_2d'].append(wavelet_2d[None])
        if filterbank_2d is not None:
            #print(f'filterbank_2d is : {wavelet_2d}')
            if isinstance(filterbank_2d, List):  # Breakdown by levels
                for jjj, this_filterbank_2d in enumerate(filterbank_2d):
                    label = f'filterbank_2d/{jjj}'
                    if label not in all_metrics:
                        all_metrics[label] = [this_filterbank_2d[None]]
                    else:
                        all_metrics[label].append(this_filterbank_2d[None])
            else:
                if 'filterbank_2d' not in all_metrics:
                    all_metrics['filterbank_2d'] = [filterbank_2d[None]]
                else:
                    all_metrics['filterbank_2d'].append(filterbank_2d[None])
            
    # Reshape as batchless tensor, then log
    # Shape is [n, n_freqs] for each parameter
    for k,v in all_metrics.items():
        if k in ['fname']:
            continue
        this_v = torch.concat(v, dim=0)
        #print(f'this_v.sahpe {this_v.shape}')
        all_metrics[k] = this_v
        if k in ['scene_area', 'rec_dist']:
            continue
        if writer is not None and ('loss' not in k and '.' in k):  # spatial acoustical parameters
            writer.add_scalar(f'Spatial_{subset}/{k}', this_v.mean().item(), 0)
        elif writer is not None and ('loss' not in k and utils.is_multislope_tag(k)):  # multislope parameters
            writer.add_scalar(f'MultiSlope_{subset}/{k}', this_v.mean().item(), 0)
        elif writer is not None and ('loss' not in k and '/' not in k): # regular acoustical parameters
            writer.add_scalar(f'Metrics_{subset}/{k}', this_v.mean().item(), 0)
        elif writer is not None and '/' in k:  # breakdown of image sim metrics
            writer.add_scalar(f'ImageSim_{subset}/{k}', this_v.mean().item(), 0)
    
    print_metrics(all_metrics, logdir=logdir, iter=ctr, subset=subset)   
    save_metrics(all_metrics, logdir=logdir, subset=subset, model_name=f'baseline_{method}')
    if subset == 'valid':
        if writer is not None:
            writer.add_scalar(f'Loss/Valid', loss.mean().item(), 0)
            writer.add_scalar(f'Loss/Valid_reduced', loss_second.item(), 0)
            if floormap_processor is not None:
                log_output_examples(0, writer, hat_acumap[None, ...], floormap[None, ...], acumap[None, ...], mask[None, ...], floormap_processor,
                                    subset='valid', batch_id_tensorboard=0, n_freq_bands=len(acumap_processor.freq_bands),
                                    acumap_processor=acumap_processor, denormalizer=denormalizer, use_split_unseen_recs=use_split_by_recs)
        # Save one batch of examples
        if False:
            save_output_examples(ctr, acuamp_output=out_denormalized.detach().cpu(), 
                                floormap=floormap.detach().cpu(), 
                                acumap_target=denormalizer(acumap).detach().cpu(), 
                                logdir=logdir)
        save_output_examples(ctr, acuamp_output=torch.stack(outputs_to_save, dim=0), 
                             floormap=None, 
                             acumap_target=torch.stack(acumaps_to_save, dim=0), 
                             logdir=logdir)
    
    cbar_labels = ['C50 [dB]', 'T30 [s]', 'DRR [dB]', 'EDT [s]']
    print(f'acumap {acumap.shape}')

    id_for_plot = 0
    # more params
    cbar_labels = ['C50 [dB]', 'T30 [s]', 'DRR [dB]', 'EDT [s]']
    freqs_toplot = [1,3,5]  

    # spatial
    id_for_plot = 10
    cbar_labels = ['C50 [dB]', 'C50 [dB]', 'C50 [dB]', 'C50 [dB]', 'C50 [dB]']
    freqs_toplot = [0,1,2]  

    print(f'HEYYYY, I am about to plot {fname}')    
    print(f'acumap {acumap.shape}')   
    
    plot_and_save_acumap(denormalizer(acumap).detach().cpu(), src.detach().cpu(), acumap_processor, freqs_toplot=freqs_toplot, cbar_labels=cbar_labels, fname='baseline_targets', scene_name=f'{fname}_{method}')
    tmp_map = out_denormalized  # * mask[id_for_plot].detach().cpu()
    tmp_map[~mask] = np.nan

    error_map = (denormalizer(acumap) - out_denormalized).abs()  # this works
    #error_map /= denormalizer(acumap).abs()  # not needed


    #error_map = denormalizer(acumap - out).abs()
    #error_map[~mask] = np.nan
    plot_and_save_acumap(tmp_map.detach().cpu() , src.detach().cpu(), acumap_processor, freqs_toplot=freqs_toplot, cbar_labels=cbar_labels, fname='baseline_out', scene_name=f'{fname}_{method}')
    plot_and_save_acumap(error_map.detach().cpu() , src.detach().cpu(), acumap_processor, freqs_toplot=freqs_toplot, cbar_labels=cbar_labels, fname='baseline_error', scene_name=f'{fname}_{method}')
    
    print('End of baseline')
    print(f'I found errors in : {len(fnames_with_errors)} files.')
    if len(fnames_with_errors) > 1:
        print(f'Files with errors:')
        print(f'{fnames_with_errors[0:5]}')

def save_output_examples(ctr, acuamp_output, floormap, acumap_target, logdir):
    """ Here we save a full batch of floormaps, target acuamps, and output acuamps.
    I need these so that I can make nice plots later.
    We save a full batch.
    
    I assume the acumaps are denormlizerd"""
    output_dir = './outputs'
    output_dir = os.path.join(logdir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask = ~torch.isnan(acumap_target)  # keeps valid pixels
    tmp_map = acuamp_output.clone() 
    tmp_map[~mask] = np.nan

    error_map = (acumap_target - acuamp_output).abs()  
    #error_map /= denormalizer(acumap).abs()  # not needed

    # Dump tensors, for further plotting later
    if floormap is not None:
        torch.save(floormap, f'{output_dir}/iter_{ctr}_floormap.pth')
    torch.save(acumap_target, f'{output_dir}/iter_{ctr}_acummap.pth')
    torch.save(acuamp_output, f'{output_dir}/iter_{ctr}_output.pth')
    torch.save(error_map, f'{output_dir}/iter_{ctr}_errors.pth')

def log_output_examples(ctr, writer, out, floormap, acumap, mask, floormap_processor, subset='train', batch_id_tensorboard=0, 
                        stft=None, n_freq_bands=3, acumap_processor=None, denormalizer=None, use_split_unseen_recs=False):
    channels = acumap.shape[-3]
    
    tmp_output = out[batch_id_tensorboard:batch_id_tensorboard+1].clone()
    tmp_output[~mask[batch_id_tensorboard:batch_id_tensorboard+1]] = np.nan
    
    tmp_output_floormasked = out[batch_id_tensorboard:batch_id_tensorboard+1].clone()
    floormask = floormap[batch_id_tensorboard:batch_id_tensorboard+1, 1:2, ...] > 0
    tmp_output_floormasked[~floormask.expand(-1, channels, -1, -1)] = np.nan

    # For within scene genrealizaiton, we mask the targets too
    if use_split_unseen_recs:
        tmp_acumap = acumap.clone()
        tmp_acumap[~mask] = np.nan
        acumap = tmp_acumap
    
    unmasked_grid = torchvision.utils.make_grid(out[batch_id_tensorboard:batch_id_tensorboard+1, 0:channels, ...].permute([1, 0, 2, 3]), 
                                                nrow=n_freq_bands, normalize=False)
    floormasked_grid = torchvision.utils.make_grid(tmp_output_floormasked[0:1, 0:channels, ...].permute([1, 0, 2, 3]), 
                                                   nrow=n_freq_bands, normalize=False)
    out_grid = torchvision.utils.make_grid(tmp_output[0:1, 0:channels, ...].permute([1, 0, 2, 3]), 
                                           nrow=n_freq_bands, normalize=False)
    target_grid = torchvision.utils.make_grid(acumap[batch_id_tensorboard:batch_id_tensorboard+1, 0:channels, ...].permute([1, 0, 2, 3]), 
                                              nrow=n_freq_bands, normalize=False)

    fig = matpotlib_imshow_grid(unmasked_grid)
    writer.add_figure(f'Unmasked/{subset}', fig, ctr)
    fig = matpotlib_imshow_grid(floormasked_grid)
    writer.add_figure(f'Floormasked/{subset}', fig, ctr)
    fig = matpotlib_imshow_grid(out_grid)
    writer.add_figure(f'Output/{subset}', fig, ctr)
    fig = matpotlib_imshow_grid(target_grid)
    writer.add_figure(f'Target/{subset}', fig, ctr if subset == 'train' else 0)

    if denormalizer is not None:
        fig = matplotlib_imshow_acumap_denorm(denormalizer(tmp_output[batch_id_tensorboard:batch_id_tensorboard+1]), None, acumap_processor)
        writer.add_figure(f'DenormOutput/{subset}', fig, ctr)
        fig = matplotlib_imshow_acumap_denorm(denormalizer(acumap[batch_id_tensorboard:batch_id_tensorboard+1]), None, acumap_processor)
        writer.add_figure(f'DenormTarget/{subset}', fig, ctr if subset == 'train' else 0)

    error = (tmp_output - acumap[batch_id_tensorboard:batch_id_tensorboard+1]).abs()
    error_grid = torchvision.utils.make_grid(error[:, 0:channels, ...].permute([1, 0, 2, 3]), 
                                             nrow=n_freq_bands, normalize=False)
    fig = matpotlib_imshow_grid(error_grid)
    writer.add_figure(f'Error/{subset}', fig, ctr)

    for ii in range(2):
        fig = matplotlib_imshow_floormap(floormap[batch_id_tensorboard+ii*4].detach().cpu(), floormap_processor)
        writer.add_figure(f'Floormap-{subset}/{ii}', fig, ctr)
        if floormap.shape[0] < 2:
            break
    
    if stft is not None:
        fig = matplotlib_imshow_floormap(stft[batch_id_tensorboard].detach().cpu(), floormap_processor)
        writer.add_figure(f'STFT_encoder/{subset}', fig, ctr)
                
def save_model(net, optimizer, logdir='.', iteration=None, model_name='best'):
    model_checkpoint = f'./net_params_{iteration:07}.pth'
    model_checkpoint = f'{logdir}/net_params_{model_name}.pth'

    checkpoint = {'net_state_dict': net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  #'scheduler_state_dict': net.lr_scheduler.state_dict(),
                  'rng_state': torch.get_rng_state(),
                  'cuda_rng_state': torch.cuda.get_rng_state()}
    torch.save(checkpoint, model_checkpoint)
    print('Checkpoint saved to {}.'.format(model_checkpoint))

def load_model(net, checkpoint_path, logdir, device):
    model_checkpoint = f'{logdir}/{checkpoint_path}/net_params_best.pth'

    print(f"Loading model state from: \n{model_checkpoint}")
    checkpoint = torch.load(model_checkpoint, map_location=device)

    [print(f'{x}') for x in checkpoint['net_state_dict'].keys()]
    #print(f"{checkpoint['net_state_dict'].keys()}")


    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #[print(f'{x}') for x in net.state_dict().keys()]
    # Use to restart training or do inference only
    net.load_state_dict(checkpoint['net_state_dict'], strict=False)
    #self.optimizer_predictor.load_state_dict(checkpoint['optimizer_state_dict'])
    #self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return net

def save_metrics(metrics: Dict, logdir: str, subset: str, model_name='best'):
    fname = f'{logdir}/metrics_{subset}_{model_name}.pth'
    torch.save(metrics, fname)
    print('Metrics saved to {}.'.format(fname))

def print_metrics(metrics: Dict, logdir: str, iter: int = 0, subset: str = 'best'):
    print('Best metrics so far:')
    with open(os.path.join(logdir, f'best_metrics_{subset}.txt'), 'w') as file:
        file.write(f'Best metrics at iter {iter}' + '\n')
        for k, v in metrics.items():
            if k in ['fname', 'scene_area', 'rec_dist']:
                continue
            line = f'{k} \t\t\t {v.mean()} \t\t {v.std()}'
            print(line)
            file.write(line + '\n')

def log_mlflow(run_name, params, metrics):
    # Initiate the MLflow run context
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

def log_wandb(params: Dict, metrics: Dict):
    "Logs metrics to wandb"
    raise NotImplementedError
    import wandb
    
    wandb.log({'best_val_step_macro': best_val_step_macro})
    wandb.summary['BestMACRO/SELD'] = best_metrics_macro[4]
    wandb.summary['BestMACRO/ER'] = best_metrics_macro[0]
    wandb.summary['BestMACRO/F'] = best_metrics_macro[1]
    wandb.summary['BestMACRO/LE'] = best_metrics_macro[2]
    wandb.summary['BestMACRO/LR'] = best_metrics_macro[3]
    wandb.summary['Losses/valid'] = best_val_loss
    wandb.summary['best_val_step_macro'] = best_val_step_macro

    wandb.log({'best_val_step_micro': best_val_step_micro})
    wandb.summary['BestMicro/SELD'] = best_metrics_micro[4]
    wandb.summary['BestMicro/ER'] = best_metrics_micro[0]
    wandb.summary['BestMicro/F'] = best_metrics_micro[1]
    wandb.summary['BestMicro/LE'] = best_metrics_micro[2]
    wandb.summary['BestMicro/LR'] = best_metrics_micro[3]
    wandb.summary['Losses/valid'] = best_val_loss
    wandb.summary['best_val_step_micro'] = best_val_step_micro

def precomputeLMDB(dataset):
    """ Precoputes the dataset by dumping all data into an LMDB file.
    The idea is that reading an LMDB file should be faster when the whole dataset does not fit in memory,
    especially when using multiple workers in the dataloader. This is useful when using augmentations 
    in the get item."""

    import lmdb
    import gc
    size = 10**12 # 1,000 GB
    chunk_size = 1000
    num_writes = 0

    # Prepare dataset object to dump lmdb
    dataset.read_lmdb = False
    dataset.env = None
    dataset.read_floormaps = False
    dataset.read_acumaps = False
    dataset.augmentation_transform = None

    fpath = os.path.join(dataset.directory_lmdb, dataset.fname_lmdb)
    if not os.path.exists(dataset.directory_lmdb):
        os.makedirs(fpath)
    env = lmdb.open(fpath, map_size=size, lock=False)

    print('')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Dumping files to LMDB')
    print(f'Dumping to:  ')
    print(f'{fpath}')
    txn = env.begin(write=True)  
    for i, (fname, rir, src, rec, scene) in tqdm(enumerate(dataset)):
        #print(f'rir.shape {rir.shape}')  # TODO REMOVE THIS
        tensor = rir.cpu().numpy().tobytes()
        key = fname.encode(encoding='utf-8')
        txn.put(key, tensor)

        # Commit in chunks to reduce memory usage
        num_writes += 1
        if num_writes % chunk_size == 0:
            print(f'Dumping partial transaction, num_writes = {num_writes}')
            txn.commit()
            gc.collect()
            txn = env.begin(write=True)  # Start a new transaction
    txn.commit()  # with env.begin(write=True) as txn:  automatically commits transaction
    env.close()

    print('')
    print('DONE')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

def precomputeLMDB_maps(dataset):
    """ Precoputes the dataset by dumping all data into an LMDB file.
    This works only on the floormaps and acumaps.
    We have 1 floormap and 1 acumap for eacn scene/src , regardless of receivers.
    So the key is scene/src
    The idea is that reading an LMDB file should be faster when the whole dataset does not fit in memory,
    especially when using multiple workers in the dataloader. This is useful when using augmentations 
    in the get item.
    
    NOTE: These can sometimes run out of memory. So I have to go to the get_dataset_full(...) function and change the splits.
    So I can manually select a subset of the whole split. This is very messy.

    Example: how to call this in vrgpu:
    python train_basic.py -c configs/mras_default.yaml --exp_name exp_test --use_vrgpu --job_id 0002 --fold all --n_files_per_scene 10000000 --dataset mras --max_length 24000 --do_precompute_maps_lmdb
    python train_basic.py -c configs/mras_more_parameters.yaml --exp_name exp_test --use_vrgpu --job_id 0002 --fold all_grids --n_files_per_scene 10000000 --dataset mras --max_length 24000 --do_precompute_maps_lmdb
    python train_basic.py -c configs/mras_multislope.yaml --exp_name exp_test --use_vrgpu --job_id 0002 --fold all_grids --n_files_per_scene 10000000 --dataset mras --max_length 24000 --do_precompute_maps_lmdb
    
    python train_basic.py -c configs/train_multislope.yaml --exp_name exp_test --use_vrgpu --job_id 0002 --fold all_no_zero --n_files_per_scene 10000000 --dataset replica --max_length 24000 --do_precompute_maps_lmdb
    """

    import lmdb
    import pickle
    import gc
    size = int(100e9) # 100 GB
    chunk_size = 50000
    num_writes = 0

    # Prepare dataset object to dump lmdb
    dataset.read_lmdb_maps = False
    dataset.env_maps = None
    dataset.read_rirs = False
    dataset.augmentation_transform = None
    dataset.avoid_storing_maps_locally = False  # Set to true, to not store maps in memory to avoid OOM issues, but then this is super slow
    dataset.avoid_storing_rirs_locally = True

    print(dataset)

    fpath = os.path.join(dataset.directory_lmdb_maps, dataset.fname_lmdb_maps)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    env = lmdb.open(fpath, map_size=size, lock=False)

    print('')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Dumping maps files to LMDB')
    print(f'Dumping to:  ')
    print(f'{fpath}')
    txn = env.begin(write=True)  
    for i, (fname, _, src, rec, scene, floormap, acumap) in tqdm(enumerate(dataset)):
        key = fname.rsplit('_', 1)[0] # frl_apartment_1/22_13  --> frl_apartment_1/22
        key = key.encode(encoding='utf-8')
        data = [floormap.cpu().numpy().tobytes(), acumap.cpu().numpy().tobytes()]
        data = pickle.dumps(data)
        txn.put(key, data)
        
        # Commit in chunks to reduce memory usage
        num_writes += 1
        if num_writes % chunk_size == 0:
            #print(f'Dumping partial transaction, num_writes = {num_writes}')
            dataset.data_floormaps = {}  # WARNING flush maps stored in memory, just to avoid OOM errors
            dataset.data_acumaps = {}  # WARNING flush maps stored in memory, just to avoid OOM errors
            #dataset.data_rirs = {}
            #dataset.data_src = {}
            #dataset.data_rec = {}
            txn.commit()
            gc.collect()
            txn = env.begin(write=True)  # Start a new transaction
    txn.commit()  # with env.begin(write=True) as txn:  automatically commits transaction
    env.close()

    # Serializing processors and saving to files
    print('')
    print('Dumping processors objects')
    with open(os.path.join(dataset.directory_lmdb_maps, dataset.fname_lmdb_maps, f'floormap_processor.pkl'), 'wb') as handle:
        pickle.dump(dataset.floormap_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dataset.directory_lmdb_maps, dataset.fname_lmdb_maps, f'acumap_processor.pkl'), 'wb') as handle:
        pickle.dump(dataset.acumap_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('')
    print('DONE')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

def precomputeLMDB_maps_DEBUG(dataset):
    """ Precoputes the dataset by dumping all data into an LMDB file.
    This works only on the floormaps and acumaps.
    We have 1 floormap and 1 acumap for eacn scene/src , regardless of receivers.
    So the key is scene/src
    The idea is that reading an LMDB file should be faster when the whole dataset does not fit in memory,
    especially when using multiple workers in the dataloader. This is useful when using augmentations 
    in the get item."""

    import lmdb
    import pickle
    size = int(100e9) # 100 GB

    # Prepare dataset object to dump lmdb
    dataset.read_lmdb_maps = False
    dataset.env_maps = None
    dataset.read_rirs = False
    dataset.augmentation_transform = None

    print(dataset)

    if not os.path.exists(dataset.directory_lmdb_maps):
        os.makedirs(dataset.directory_lmdb_maps)
    fpath = os.path.join(dataset.directory_lmdb_maps, dataset.fname_lmdb_maps)
    env = lmdb.open(fpath, map_size=size)

    print('')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Dumping maps files to LMDB')
    print(f'Dumping to:  ')
    print(f'{fpath}')
    with env.begin(write=True) as txn:
        for i, (fname, _, src, rec, scene, floormap, acumap) in tqdm(enumerate(dataset)):

            #data = [floormap.cpu().numpy().tobytes(), acumap.cpu().numpy().tobytes()]
            #data = pickle.dumps(data)

            #key = fname.split("_")[-2]  # frl_apartment_1/22_13  --> frl_apartment_1/22
            key = fname.rsplit('_', 1)[0] # frl_apartment_1/22_13  --> frl_apartment_1/22
            print(key)
            key = key.encode(encoding='utf-8')
            tensor = floormap.cpu().numpy().tobytes()
            txn.put(key, tensor)
    env.close()

    print('')
    print('DONE')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

if __name__ == '__main__':
    # DEbugging errors when using more inputs channels
    import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ['TORCH_USE_CUDA_DSA'] = "1"

    if False:
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_files_per_scene", help="Number of files per scens", type=int, default=2)
        parser.add_argument("--scenes", nargs='*', help="List of scenes to process", type=int, default=[0])
        parser.add_argument("--exp_name", help="List of scenes to process", type=str, default='debug')
        parser.add_argument("--use_dgx", help="Flag to change data paths if using DGX", action='store_true')
        parser.add_argument("--use_triton", help="Flag to change data paths if using Aalto Triton cluster", action='store_true')
        parser.add_argument("--use_vrgpu", help="Flag to change data paths if using the deskptop vrgpu in Aalto", action='store_true')
        parser.add_argument("--num_workers", help="Number of workers for the ", type=int, default=2)
        parser.add_argument("--job_id", help="Slurm job id", type=str, default='debug')

        # Messy, but quick params to run jobs from sbatch
        # baseline_methods = ['average_rir_all', 'average_rir_same_scene', 'average_acumap_scene', 'same_rir_NN',  'average_rir_same_scene_and_source', 'sample_acumaps_scene']
        parser.add_argument("--baseline_method", help="baseline method", type=str, default='sample_acumaps_scene')
        parser.add_argument("--fold", help="fold for the dataset", type=int, default=0)
        args = parser.parse_args()
    
    # Read params from config file and command line
    args = parameters.get_parameters()

    if args['use_dgx'] and args['use_triton']:
        raise ValueError('ERROR: We cannot use both DGX and triton at the same time. Pick one cluster.')
    if args['use_dgx'] and args['use_vrgpu']:
        raise ValueError('ERROR: We cannot use both DGX and vrgpu at the same time. Pick one cluster.')
    if args['use_triton'] and args['use_vrgpu']:
        raise ValueError('ERROR: We cannot use both vrgpu and triton at the same time. Pick one cluster.')

    #train_overfit('soundspaces', args['n_files_per_scene'], args.scenes)
    train_overfit(args)
