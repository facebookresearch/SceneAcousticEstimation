# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3'
from torch.utils.data import Dataset
import os
import re
import copy
import lmdb
import numpy as np
import torchaudio
import torch
from tqdm import tqdm, trange
import torch.utils.data.dataloader
import warnings
import math
import pickle
#import h5py
import json
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from typing import List, Union, Tuple
sns.set_theme(style="darkgrid")

from features import fixed_slice_coords_matterport, fixed_slice_replica
import utils

# scene / listener
list_or_files_with_errors = {'line1_materials_0': [15]}

def match_pattern_scenes(s):
    """ For the simultated apartments"""
    pattern = r'(line|grid)\d+_materials_\d+'
    if re.match(pattern, s):
        return True
    return False

class SoundspacesDataset(Dataset):
    """
    Torch dataset that reads the Soundspaces 1.0 dataset that includes Room Impulse Responses and some metadata.
    NOTE: This is work in progress and will fail to read all the dataset because its massive (~800 GB). 
          Eventtually, there will be a preprocessing step that can read save it to a mmap or similar.
    
    Each item returns:
        RIR - (torch.Tensor) - [channels, timesteps]
        fname - (str) - File Name of the RIR (including scene directory)
        pos_src - (torch.Tensor) - [3] - Vector of x,y,z absolute scene coordinates for the source
        pos_rec - (torch.Tensor) - [3] - Vector of x,y,z absolute scene coordinates for the receiver
        scene - [Optional] (str) - The name of the scene
        floormap - [Optional] (torch.Tensor) - [channels, x, y]
    """
    def __init__(self, 
                 scenes: List[Union[float, str]] = [0],
                 datapath: str = '/m/triton/cs/sequentialml/datasets/soundspaces/data/',
                 rir_type: str = 'ambisonic',  # {binaural, ambisonic, raw}
                 directory_rir: str = '{:s}_rirs/replica',  # rir_type/scene_name
                 directory_rir_mras: str = '/m/triton/cs/sequentialml/datasets/scenes_proposed_v3/{:s}/outputs/',  # scene_name
                 directory_rir_lmdb_per_scene: str = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/data/lmdb',  # one lmdb with the rirs per scene, includes points.txt
                 directory_jsons: str = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}',  # scene_name
                 directory_jsons_matterport: str = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}',  # scene_name
                 directory_jsons_mras: str = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}',  # scene_name
                 directory_geometry: str = '/m/triton/cs/sequentialml/datasets/replica/{:s}/mesh.ply', # scene_name
                 directory_geometry_matterport: str = '/m/triton/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply', # scene_name, scene_name
                 directory_geometry_mras: str = '/m/triton/cs/sequentialml/datasets/scenes_proposed_v4/{:s}/combined.obj', # scene_name, scene_name
                 directory_lmdb: str = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb/',  # single file for whole datate
                 directory_lmdb_maps: str = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps/',  # single file for whole datate
                 fname_rir_template: str = '{:s}.wav',  # e.g. 101_10.wav
                 fname_meta_template: str = '{:s}_params.json',  # e.g. 101_10_params.json
                 fname_lmdb: str = 'rirs.lmdb',  # Filename of the LMDB file
                 fname_lmdb_maps: str = 'maps.lmdb',  # Filename of the LMDB file
                 resampler=None,
                 lazy_loading=True,  # Lazy loading of rirs and/or curves files
                 read_lmdb=False,  # Read rirs from lmdb instead of loading into memory
                 read_lmdb_maps=False, # Read floormaps and acumaps from LMDB
                 data_limit=100,
                 rir_output_channels: List[int] = list(range(9)), # Channels to load, useful to reduce ambisonic order
                 read_rirs: bool = False,
                 read_scenes: bool = False,
                 read_floormaps: bool = False,
                 read_acumaps: bool = False,
                 max_length: int = -1,  # in samples, -1 for no cropping or padding
                 n_files_per_scene: int = 1000,
                 floormap_processor = None,  # To extract floormaps, see features.py
                 acumap_processor = None,
                 augmentation_transform=None,  # For augmentation,
                 multi_story_removal=False,  # For multi-story scenes, ignore src,rec locations in other stories, using predetermined slices in Features.py
                 use_fixed_masks=False,# To mask areas of each acumap
                 avoid_storing_rirs_locally=False,
                 avoid_storing_maps_locally=False,
                 return_rot_angle = False,
                 do_precentering = False):  
        super().__init__()
        
        self.scenes = scenes
        self.datapath = datapath
        self.rir_type = rir_type
        self.directory_rir = directory_rir
        self.directory_rir_lmdb_per_scene = directory_rir_lmdb_per_scene
        self.directory_rir_mras = directory_rir_mras
        self.directory_jsons = directory_jsons
        self.directory_jsons_matterport = directory_jsons_matterport
        self.directory_jsons_mras = directory_jsons_mras
        self.directory_geometry = directory_geometry
        self.directory_geometry_matterport = directory_geometry_matterport
        self.directory_geometry_mras = directory_geometry_mras
        self.directory_lmdb = directory_lmdb
        self.directory_lmdb_maps = directory_lmdb_maps
        self.fname_rir_template = fname_rir_template
        self.fname_meta_template = fname_meta_template
        self.fname_lmdb = fname_lmdb
        self.fname_lmdb_maps = fname_lmdb_maps
        self.resampler = resampler
        self.lazy_loading = lazy_loading
        self.data_limit = data_limit
        self.read_rirs = read_rirs
        self.read_lmdb = read_lmdb
        self.read_lmdb_maps = read_lmdb_maps
        self.read_scenes = read_scenes
        self.read_floormaps = read_floormaps
        self.read_acumaps = read_acumaps
        self.max_length = max_length
        self.n_files_per_scene = n_files_per_scene
        self.floormap_processor = floormap_processor
        self.acumap_processor = acumap_processor
        self.rir_output_channels = rir_output_channels
        self.augmentation_transform = augmentation_transform
        self.use_fixed_masks = use_fixed_masks
        self.multi_story_removal = multi_story_removal
        self.max_height_distance_for_multi_story = 2.0
        self.avoid_storing_rirs_locally = avoid_storing_rirs_locally
        self.avoid_storing_maps_locally = avoid_storing_maps_locally
        self.return_rot_angle = return_rot_angle
        self.do_precentering = do_precentering
        self.fs = None

        # Note, this might cause erros, see:
        # https://junyonglee.me/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
        #print('yolooooooo')  # TODO remove this, once lmdb is debugged
        #print(self.read_lmdb) # TODO remove this, once lmdb is debugged
        if self.read_lmdb:
            if not match_pattern_scenes(self.scenes[0]):
                fpath = os.path.join(self.directory_lmdb, self.fname_lmdb)
                #fpath = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/data/lmdb2/office_0'
                #print(fpath)  # TODO remove this, once lmdb is debugged
                #print(os.path.exists(fpath))  # TODO remove this, once lmdb is debugged
                if os.path.exists(fpath):
                    self.env = lmdb.open(fpath, readonly=True, lock=False)
                else:
                    self.env = None
                    print(f'WARNING: Cannot open lmdb in {fpath}')
        else:
            self.env = None


        # Debugging lmdb files
        if False:
            txn = self.env.begin()
            cursor = txn.cursor()

            count = 0
            for key, _ in cursor:
                print(key.decode())  # Assuming the keys are stored as byte strings
                try:
                    pass
                    #print(key.decode('utf-8'))
                    #print(key.decode('utf-8', errors='ignore'))
                    #print(key.decode(errors='ignore'))
                except UnicodeDecodeError:
                    print("Unable to decode key:", key)
                count += 1
                if count == 1:
                    break

        # NEW_ Read lmdb for floormaps and acumaps
        if self.read_lmdb_maps:
            fpath = os.path.join(self.directory_lmdb_maps, self.fname_lmdb_maps)
            #fpath = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/data/lmdb2/office_0'
            #print(fpath)  # TODO remove this, once lmdb is debugged
            #print(os.path.exists(fpath))  # TODO remove this, once lmdb is debugged
            if os.path.exists(fpath):
                self.env_maps = lmdb.open(fpath, readonly=True, lock=False)
            else:
                self.env_maps = None

            # Read proessors if available (processors are dumped to the same directory)
            processor_path = os.path.join(fpath, 'floormap_processor.pkl')
            if os.path.exists(processor_path):
                with open(processor_path, 'rb') as f:
                    self.floormap_processor = pickle.load(f) 
                    #self.floormap_processor = utils.CudaToCpuUnpickler(f)  # Update 05.10.2025, we might need this when there is no gpu, not sure if this will break other stuff
                    if not hasattr(self.floormap_processor, "use_soft_position"):
                        setattr(self.floormap_processor, "use_soft_position", False)
                    if not hasattr(self.floormap_processor, "use_directional_sources"):
                        setattr(self.floormap_processor, "use_directional_sources", False)
            
            processor_path = os.path.join(fpath, 'acumap_processor.pkl')
            if os.path.exists(processor_path):
                with open(processor_path, 'rb') as f:
                    self.acumap_processor = pickle.load(f)
        else:
            self.env_maps = None

        self.scenes_points = {}  # per scene, a dictionary that maps points to coordinates
        self.scenes_jsons = {}  # per scene, a json with all acoustic parameters for all src-rec pairs
        self.scenes_rir_lmdbs = {}  # per scene, we have an lmdb with all the RIRs, instead of reading individual files, mostly for mp3d and mras
        self.scenes_heights = {}  # per scene, to keep track of height of scenes, to find which scenes are multi-story
        
        self.fnames = []
        self.data_scenes = []  # per file, so that I can do filtering for analysis
        self.data_src_ids = []  # per file, so that I can do filtering for analysis
        self.data_rec_ids = []  # per file, so that I can do filtering for analysis
        self.data_rirs = {}
        self.data_src = {}
        self.data_rec = {}
        self.data_floormaps = {}  # keys are scenes, not fnames
        self.data_acumaps = {}  # keys are scene/src_id, not fnames
        self.data_centroids_per_scene = {}

        if self.use_fixed_masks:
            self.data_masks_acuamps = {}
        else:
            self.data_masks_acuamps = None  # keys are scene/src_id, not fnames
        
        # FOr MRAS, load lmdb if needed
        if self.read_lmdb:
            if match_pattern_scenes(self.scenes[0]):
                # hard coded for now
                lmdbs_mras = ['rirs_mono_mras_grids.lmdb',
                              'rirs_mono_mras_lines.lmdb']
                for f_lmdb in lmdbs_mras:
                    fpath = os.path.join(self.directory_lmdb, f_lmdb)
                    if os.path.exists(fpath) and 'line' in fpath:
                        self.scenes_rir_lmdbs['lines'] = lmdb.open(fpath, readonly=True, lock=False)
                    elif os.path.exists(fpath) and 'grid' in fpath:
                        self.scenes_rir_lmdbs['grids'] = lmdb.open(fpath, readonly=True, lock=False)
                    else:
                        print(f'WARNING: Cannot open lmdb in {fpath}')
                self.env = None


        # Some fixed properties of the Replica dataset
        self.replica_scenes_names = ['apartment_0', 'apartment_2', 'frl_apartment_1', 'frl_apartment_3',
                                     'frl_apartment_5', 'office_0', 'office_2', 'office_4',
                                     'room_1', 'apartment_1', 'frl_apartment_0', 'frl_apartment_2',
                                     'frl_apartment_4', 'hotel_0', 'office_1', 'office_3',
                                     'room_0', 'room_2']
        self.matterport_scenes_names = ['pLe4wQe7qrG', 'gTV8FGcVJC9', 'YFuZgdQ5vWj', 'oLBMNvg9in8', 'ZMojNkEp431', 'r47D5H71a5s', 'wc2JMjhGNzB', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'r1Q1Z4BcV1o', 'Vt2qJdWjCF2', 'JeFG25nYj2p', 'gYvKGZ5eRqb', 'x8F5xyUWy9e', 'E9uDoFAP3SH', 'ULsKaCPVFJR', 's8pcmisQ38h', 'q9vSo1VnCiC', 'V2XKFyX4ASd', 'EDJbREhghzL', 'p5wJjkQkbXX', 'kEZ7cmS4wCh', 'b8cTxDM8gDG', 'Z6MFQCViBuw', 'VLzqgDo317F', 'D7N2EKCX4Sj', 'sKLMLpTHeUy', 'pRbA3pwrgk9', 'sT4fr6TAbpF', '2n8kARJN3HM', '17DRP5sb8fy', 'rPc6DW4iMge', 'XcA2TqTSSAj', 'rqfALeAoiTq', '29hnd4uzFmX', 'YmJkqBEsHnH', 'D7G3Y4RVNrH', 'i5noydFURQK', 'qoiz87JEwZ2', 'B6ByNegPMKs', '1pXnuDYAj8r', 'JF19kD82Mey', 'Uxmj2M2itWa', 'mJXqzFtmKg4', '5LpN3gDmAk7', 'JmbYfDe2QKZ', '7y3sRwLe3Va', 'jtcxE69GiFV', 'SN83YJsR3w2', 'EU6Fwq7SyZv', 'X7HyMhZNoso', 'VzqfbhrpDEA', 'HxpKQynjfin', 'WYY7iVyf5p8', 'Vvot9Ly1tCj', 'QUCTc6BB5sX', 'yqstnuAEVhm', 'aayBHfsNo7d', '2azQ1b91cZZ', 'ARNzJeq3xxb', 'zsNo4HB9uLZ', 'uNb9QFRL6hY', '82sE5b5pLXE', '5q7pvUzZiYa', '759xd9YjKW5', 'PX4nDJXEHrG', '1LXtFkjw3qL', '8194nk5LbLH', 'UwV83HsGsw3', 'S9hNv5qa7GM', 'e9zR4mvMWw7', 'jh4fc5c5qoQ', 'Pm6F8kyY3z2', '8WUmhLawc2A', '5ZKStnWn8Zo', 'VVfe2KiqLaN', 'PuKPg4mmafe', 'ac26ZMwG7aT', 'fzynW3qQPVF', 'ur6pFq6Qu1A', 'GdvgFV5R1Z5', 'TbHJrupSAjP', 'gxdoqLR6rwA', 'pa4otMbVnkk', 'vyrNrziPKCB', 'gxdoqLR6rwA']
        self.replica_angles = [0, 90, 180, 270]  # Only relevant for binaural rirs

        # TODO: just for debugging the multi-story scenes
        self.rejected_points = []
        self.tmp_distances_to_centers = []
        self.tmp_distances_to_centers_booleans = []

        # Each scene has its own directory, that looks like 'apartment_0'
        scenes_path = os.path.join(self.datapath, self.directory_rir.format(self.rir_type))

        print('Reading scenes (subdirectories) names')
        if self.scenes is not None and len(self.scenes) > 0:
            if type(self.scenes[0]) is int:  # NOTE: This is only relevant for replica
                # Read all scenes and select some from id
                subdirs = [x[0].split() for x in os.walk(scenes_path) if x[0] != 'collection'] 
                del subdirs[0]  # remove root directory
                self.scenes = [os.path.basename(subdirs[id][0]) for id in self.scenes]    
            if type(self.scenes[0]) is str:
                # Use the scenes given already as they are
                self.scenes = self.scenes 
        else:
            # Read all scenes
            subdirs = [x[0] for x in os.walk(scenes_path) if x[0] != 'collection'] 
            del subdirs[0]  # remove root directory
            self.scenes = subdirs
            self.scenes = [os.path.basename(x) for x in subdirs]
 
        # Read metadata from points.txt of each scene, to get the coordinates mapping
        self.num_srcs_per_scene = {}    
        self.num_recs_per_scene = {}    
        print('Reading metadata per scene')
        read_points_txt = False
        scenes_to_remove = []  # For MRAS, we keep track of invalid scenes, and remove them later
        for scene in tqdm(self.scenes):          
            if match_pattern_scenes(scene):
                scenes_path = self.directory_geometry_mras.format(scene)
            elif scene in self.replica_scenes_names:
                scenes_path = os.path.join(self.datapath, self.directory_rir.format(self.rir_type))
                read_points_txt = True
            elif scene in self.matterport_scenes_names:
                scenes_path = self.directory_rir_lmdb_per_scene
                read_points_txt = True
            else:
                print(f'ERROR: scene: {scene} not found.')
            if read_points_txt:
                with open(os.path.join(scenes_path, scene, 'points.txt'), 'r') as f:
                    data = np.loadtxt(f, dtype=np.float32)
                    # Extract the id, x, y, and z coordinates
                    point_coords = data[:, 0:4]
                self.scenes_points[scene] = point_coords

            ########print(self.scenes_points[scene].shape)
            # Read rir fnames, replica (assuming RIRs stored as individual wav files)
            
            if scene in self.replica_scenes_names:
                files = os.listdir(os.path.join(scenes_path, scene))
                ##print(f'files in {scene} :  {len(files)}')
                ##print(f'{scenes_path}')
                n = 0
                for f in files:
                    if not f.endswith('.wav'):
                        print(f'Ignoring file {f}, n= {n}')
                        continue

                    fname = os.path.join(scene, os.path.basename(f))
                    src, rec = self.parse_fname(fname)  # in string, these are ids for points
                    tmp_src = np.isclose(self.scenes_points[scene][:,0], src, atol=1e-1)
                    src_coords = torch.from_numpy(self.scenes_points[scene][tmp_src, 1:4])
                    tmp_rec = self.scenes_points[scene][:, 0] == rec
                    rec_coords = torch.from_numpy(self.scenes_points[scene][tmp_rec, 1:4])

                    # For multi-story scenes, ignore src/rec positions that are too far way (in height) from the fixed slice
                    if self.multi_story_removal:
                        if scene in fixed_slice_replica.keys():
                            tmp = np.abs(src_coords[0,2] - fixed_slice_replica[scene])
                            self.tmp_distances_to_centers.append(tmp)
                            self.tmp_distances_to_centers_booleans.append(tmp > self.max_height_distance_for_multi_story)
                            if np.abs(src_coords[0,2] - fixed_slice_replica[scene]) > self.max_height_distance_for_multi_story:
                                # src is probably in another story, ignoring
                                self.rejected_points.append(fname)
                                continue
                            if np.abs(rec_coords[0,2] - fixed_slice_replica[scene]) > self.max_height_distance_for_multi_story:
                                # rec is probably in another story, ignoring
                                self.rejected_points.append(fname)
                                continue

                    # TODO Just to debug the adversarial training
                    # We only want to keep a very small amount of sources, but lots of RIRs to have full acumaps
                    if False and (src < 50 or src > 51):
                        #print(src)
                        #print(50 < src < 51)
                        continue

                    self.fnames.append(fname)
                    self.data_scenes.append(scene)
                    self.data_src[fname] = src_coords.squeeze()
                    self.data_rec[fname] = rec_coords.squeeze()
                    self.data_src_ids.append(src)
                    self.data_rec_ids.append(rec)

                    n += 1
                    if self.n_files_per_scene is not None and n >= self.n_files_per_scene:
                        print(f'break at n = {n}')
                        break  # break inner loop
                ##print(f'n= {n} for {scene}')

            # Read rir fnames, as keys for the lmdb of individual scenes
            # NOTE: keys here do NOT have the '.wav'
            elif scene in self.matterport_scenes_names:
                tmp_lmdb_file = os.path.join(scenes_path, scene)  # path to the lmdb file
                tmp_env = lmdb.open(tmp_lmdb_file, readonly=True, lock=False)
                self.scenes_rir_lmdbs[scene] = tmp_env
                with tmp_env.begin() as txn:
                    cursor = txn.cursor()
                    n = 0
                    for key, _ in cursor:
                        fname = key.decode('utf-8')
                        src, rec = self.parse_fname(fname)  # in string, these are ids for points
                        tmp_src = np.isclose(self.scenes_points[scene][:,0], src, atol=1e-1)
                        src_coords = torch.from_numpy(self.scenes_points[scene][tmp_src, 1:4])
                        tmp_rec = self.scenes_points[scene][:, 0] == rec
                        rec_coords = torch.from_numpy(self.scenes_points[scene][tmp_rec, 1:4])

                        # For multi-story scenes, ignore src/rec positions that are too far way (in height) from the fixed slice
                        if self.multi_story_removal:
                            if scene in fixed_slice_coords_matterport.keys():
                                if np.abs(src_coords[0,2] - fixed_slice_coords_matterport[scene]) > self.max_height_distance_for_multi_story:
                                    # src is probably in another story, ignoring
                                    #self.rejected_points.append(fname)  # only for debugging
                                    continue
                                if np.abs(rec_coords[0,2] - fixed_slice_coords_matterport[scene]) > self.max_height_distance_for_multi_story:
                                    # rec is probably in another story, ignoring
                                    #self.rejected_points.append(fname)  # only for debugging
                                    continue
                        
                        self.fnames.append(fname)
                        self.data_scenes.append(scene)
                        self.data_src[fname] = src_coords.squeeze()
                        self.data_rec[fname] = rec_coords.squeeze()
                        self.data_src_ids.append(src)
                        self.data_rec_ids.append(rec)

                        n += 1
                        if self.n_files_per_scene is not None and n >= self.n_files_per_scene:
                            break  # break inner loop
            
            # Scenes from the MRAS scene generator
            # Here we have a sources.txt and listeners.txt with the ids and coordinates
            elif match_pattern_scenes(scene):   
                rot_mat = utils.get_rotation_matrix(0.0, 0.0, -np.pi/2, device='cpu')  # rotate 90 degs due to different convention

                srcs_file = os.path.join(os.path.dirname(scenes_path), 'sources.txt')
                if not os.path.exists(srcs_file):
                    print(f'WARNING: Scene {scene} not found. Ignoring.')
                    print(srcs_file)
                    scenes_to_remove.append(scene)
                    # This is becasuse some scenes had corrupt zips, so there are no rirs
                    continue  # next scene
                tmp_srcs_keys = []
                tmp_srcs_coords = {}
                #print('MRASSSS: Reading sources.txt')
                with open(srcs_file, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        items = line.split()  # e.g.  4 sources_4 8.75 2.75 1.5
                        key = int(items[1].split('_')[1])  # 4 sources_4 8.75 2.75 1.5  --> 4 (from sources_4)
                        values = np.array([float(i) for i in items[2:]])[None, ...]

                        values[:, [0, 1]] = values[:, [1, 0]]  # Different convention, swap x and y
                        values[:, [0, 1]] -= 10  # Corect from xlim [0,20] to [-10, 10]
                        values = torch.from_numpy(values).numpy()

                        #values = torch.einsum('ik,kj->ij', values.to(torch.float64), rot_mat.T).to(torch.float32).numpy()

                        #values = np.array([1.5, 4, 1.5])

                        #values *= np.array([1, -1, 1])  # Different convention
                        ###values = values * np.array([1, 1, 1]) + np.array([-0, -20, 0])  # Different convetion
                        tmp_srcs_keys.append(key)
                        tmp_srcs_coords[key] = torch.from_numpy(values)

                recs_file = os.path.join(os.path.dirname(scenes_path), 'listeners.txt')
                tmp_recs_keys = []
                tmp_recs_coords = {}
                #print('MRASSSS: Reading listeners.txt')
                with open(recs_file, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        items = line.split()  # e.g.  4 listener_4 8.75 2.75 1.5
                        key = int(items[1].split('_')[1])  # 4 listener_4 8.75 2.75 1.5  --> 4 (from listener_4)

                        # Some listeners do not exist in the RIRs, ignoring
                        if scene in list_or_files_with_errors.keys():
                            if key in list_or_files_with_errors[scene]:
                                print(f'WARNING: Rejected listener {key} for scene {scene}')
                                continue
                        values = np.array([float(i) for i in items[2:]])[None, ...]
                        values[:, [0, 1]] = values[:, [1, 0]]  # Different convention, swap x and y
                        values[:, [0, 1]] -= 10  # Corect from xlim [0,20] to [-10, 10]
                        values = torch.from_numpy(values).numpy()

                        #values = torch.einsum('ik,kj->ij', values.to(torch.float64), rot_mat.T).to(torch.float32).numpy()
                        ###values = values * np.array([1, 1, 1]) + np.array([-0, -20, 0])  # Different convetion
                        tmp_recs_keys.append(key)
                        tmp_recs_coords[key] = torch.from_numpy(values)

                self.num_srcs_per_scene[scene] = len(tmp_srcs_keys)  ## Just to calculate how many srcs per scene
                self.num_recs_per_scene[scene] = len(tmp_recs_keys)  ## Just to calculate how many srcs per scene

                n = 0
                #print('MRASSSS: Parsing fnames')
                do_break = False
                for src in tmp_srcs_keys:
                    if not do_break:
                        for rec in tmp_recs_keys:
                            fname = f'{scene}/{src}_{rec}'
                            if len(self.data_src) != len(self.fnames):
                                print(f'fname: {fname}')
                            
                            src_coords = tmp_srcs_coords[src]
                            rec_coords = tmp_recs_coords[rec]

                            self.fnames.append(fname)
                            self.data_scenes.append(scene)
                            self.data_src[fname] = src_coords.squeeze()
                            self.data_rec[fname] = rec_coords.squeeze()
                            self.data_src_ids.append(src)
                            self.data_rec_ids.append(rec)

                            n += 1
                            if self.n_files_per_scene is not None and n >= self.n_files_per_scene:
                                #print(f'break at n = {n}')
                                do_break = True  # to break outer loop
                                break  # break inner loop

        # Remove invalid scenes if needed
        # This is mostly for MRAS, because some scenes do not have sources.txt
        # So even if there are RIRs, I cannot create maps without the coordinates
        # This is mostly a big problem when computing baselines, because I sample with self.scenes
        if len(scenes_to_remove) > 0:
            self.scenes = [item for item in self.scenes if item not in set(scenes_to_remove)]

        self.data_scenes = np.array(self.data_scenes)
        self.data_src_ids = np.array(self.data_src_ids)
        self.data_rec_ids = np.array(self.data_rec_ids)

        print(f'len data scenes {len(self.data_scenes)}')
        print(f'len data_src_ids {len(self.data_src_ids)}')
        print(f'len data_rec_ids {len(self.data_rec_ids)}')
        print(f'len data_src {len(self.data_src)}')
        print(f'len data_ fnames {len(self.fnames)}')


        self.__validate__()
        print(self)


        #selected_rows = df[df["Path"].isin(self.scenes)]

        #self.fnames = [os.path.basename(x.str) for x in selected_rows["Path"].tolist()]å

    def __validate__(self):
        assert len(self.fnames) == len(self.data_scenes), 'Fnames and data_scenes should have the same count'
        assert len(self.fnames) == len(self.data_src_ids), 'Fnames and data_src_ids should have the same count'
        assert len(self.fnames) == len(self.data_rec_ids), 'Fnames and data_src_ids should have the same count'
        #assert len(self.fnames) == len(self.data_rirs), 'Fnames and data_rirs should have the same count'
        assert len(self.fnames) == len(self.data_src), 'Fnames and data_src should have the same count'
        assert len(self.fnames) == len(self.data_rec), 'Fnames and data_rec should have the same count'   

    def __extract_floormap__(self, scene: str) -> torch.Tensor:
        """ Extract a floormap from each mesh, using the floormap processor"""
        import open3d as o3d   # Import here because Open3d cannot be installed in triton
        assert self.floormap_processor is not None, f'ERROR, we need a valid FloorMapProcessor object to process floormaps.'
        
        subdivide_mesh = False
        swap_xy = False
        if scene in self.replica_scenes_names:
            mesh_fname = os.path.join(self.directory_geometry.format(scene))
        elif scene in self.matterport_scenes_names:
            mesh_fname = os.path.join(self.directory_geometry_matterport.format(scene, scene))
        elif match_pattern_scenes(scene):
            mesh_fname = os.path.join(self.directory_geometry_mras.format(scene))
            subdivide_mesh = True
            swap_xy = True
        print(f'mesh: {mesh_fname}')
        mesh = o3d.io.read_triangle_mesh(mesh_fname)
        if subdivide_mesh:
            if len(mesh.triangles) < 1000:
                #print(f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
                mesh = mesh.subdivide_midpoint(number_of_iterations=6)
                #print(f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
                #vertices = np.asarray(mesh.vertices)
                #print(vertices.shape)
                #print(vertices)
        vertices = np.asarray(mesh.vertices)
        if swap_xy:
            vertices[:, [0, 1]] = vertices[:, [1, 0]]  # Due to different convetion for coords
            vertices[:, 0] -= 10  # Correct from xlim[0, 20] to [-10, 10], so that rotations work
            vertices[:, 1] -= 10
            #print(vertices[0:5])  # TODO remove this, just for debugging

        z_range = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        self.scenes_heights[scene] = z_range

        # Pre centering
        #centroid = np.mean(vertices, axis=0)
        if self.do_precentering:
            warnings.warn('WARNING: Pre-centering is not fully ready. Src and rec positions will be off')
            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
            centroid = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2, 0])
            print(centroid)
            vertices = vertices - centroid

            self.data_centroids_per_scene[scene] = centroid

        floormap = self.floormap_processor.process(vertices, scene)
        return floormap

    def __extract_acumap__(self, scene: str, src_id: str) -> torch.Tensor:
        """ Extract a AcuMap from the given scene and source, using the acumap processor.
        NOTE: the src here is src_id, not coordinates."""
        assert self.acumap_processor is not None, f'ERROR, we need a valid AcuMapProcessor object to process acumaps.'

        # Read json of acoustic parameters per scene, if not previously loaded
        if scene not in self.scenes_jsons:
            if scene in self.replica_scenes_names:
                json_path = self.directory_jsons.format(scene)
            elif scene in self.matterport_scenes_names:
                json_path = self.directory_jsons_matterport.format(scene)
            elif match_pattern_scenes(scene):
                json_path = self.directory_jsons_mras.format(scene)
            else:
                raise ValueError(f'ERROR, I cannot find the jsons for scene {scene}. Are the acoustic parameters computed already?')
            try:
                with open(os.path.join(json_path, f'{scene}_metrics.json'), 'r') as f:
                    data = json.load(f)
                    self.scenes_jsons[scene] = data
            except Exception as e:
                print(f'ERROR reading json: {scene}')
                print(f'ERROR reading in path json: {os.path.join(json_path, f"{scene}_metrics.json")}')
                raise e
                
        acu_params_dict = self.scenes_jsons[scene]

        # UPDATE: This is when the jsons where dumped in batches, e.g. when doing beamforming
        # Howver, this is very messy. I might change this to jsonlines
        if isinstance(acu_params_dict, List):
            dict_data = {}
            for d in acu_params_dict:
                dict_data.update(d)
            acu_params_dict = dict_data

        if isinstance(src_id, int):
            src_id = str(src_id)
        recs_ids = self.data_rec_ids[self.data_scenes == scene]
        #print(recs_ids) # TODO, remove, just for debugging
        if self.do_precentering:
            centroid = self.data_centroids_per_scene[scene]
        else:
            centroid = None
        acumap = self.acumap_processor.process(acu_params_dict, src_id, recs_ids, centroid)  # This is needed if we want to divide receivers
        #acumap = self.acumap_processor.process(acu_params_dict, src_id)
        #del self.scenes_jsons[scene] # Just to release memory, this fails
        return acumap
            
    def parse_fname(self, fname: str) -> Tuple[str, str]:
        '''
        Helper function that parses the previously read fnames from soundpsaces wav files.
        These are "src_rec.wav" , in ids.
        e.g.
            '[\'101_10.wav\']' --> (101, 10)
        '''
        fname = os.path.basename(fname)  # ambisonic_rirs/replica/office_1/irs/9_8.wav  --> 9_8.wav
        fname = fname.replace('[', '')
        fname = fname.replace(']', '')
        fname = fname.replace('\'', '')
        fname = fname.replace('.wav', '')
        fname = fname.split('_')

        return int(fname[0]), int(fname[1])

    def get_acumap_key(self, scene: str, src_id: str) -> str:
        key_acumap = f'{scene}/{src_id}'
        return key_acumap

    def get_fnames(self) -> List[str]:
        return self.fnames

    def __len__(self) -> int:
        return len(self.fnames)

    def __read_rir__(self, fname: str, scene: str) -> torch.Tensor:
        '''
        Loads a rir (wav) file from disk.
        e.g.
            __read_rir__('office_0/10_39.wav')
        '''
        import soundfile

        #print('scene')
        #print(scene)
        if not match_pattern_scenes(scene):
            # replica, mp3d
            fpath = os.path.join(self.datapath, self.directory_rir.format(self.rir_type), fname)
            this_rir, fs = torchaudio.load(fpath, backend='soundfile')
        else:
            # scenes_proposed_v2
            _, tmp = fname.split('/')  # line70_materials_0/0_5 --> line70_materials_0, 0_5
            src, rec = tmp.split('_')  # 0_5 --> 0, 5
            fpath = os.path.join(self.directory_rir_mras.format(scene), f'ir_T#0_L#{rec}_S#{src}.wav')
            #print(f'fpath')
            #print(fpath)
            # This is here because the list of receivers is wrong for some scenes
            # So I will just ignore that
            try:
                this_rir, fs = torchaudio.load(fpath, backend='soundfile')
            except soundfile.LibsndfileError as e:
                print(f'ERRROROR, soundfile.LibsndfileError. I cannot read this file: \n{fpath}')
                print('Replacing with silence.')
                this_rir = torch.zeros((9, 100))
                fs = 24000

        if self.fs is None:
            self.fs = fs
        else:
            if self.fs != fs:
                warnings.warn(f'WARNING: I found different sampling rates when loading RIRs: {fs} and previous {self.fs}')

        if len(this_rir.shape) < 2:
            this_rir = this_rir.unsqueeze(0)  # shape is [channels, timesteps]

        # Sanity check
        #assert utils.validate_audio(this_rir), f'ERROR, the audio file seems wrong. {fname}'
        replace_silence = False
        ignore_validation = True
        this_rir[torch.logical_not(torch.isfinite(this_rir))] = 0.0  # Replace NaNs or Infs
        if not utils.validate_audio(this_rir):
            # So this is super wrong, but otherwise I get NaNs when training.
            # I dont know why some rirs are so bad
            
            if replace_silence:
                print(f'WARNING: Something is wrong with this RIR: {fname}. Replacing the RIR with silence.')
                this_rir = torch.zeros(size=(this_rir.shape[-2], 4000))  # create a fake impulse
            elif not ignore_validation:
                print(f'WARNING: Something is wrong with this RIR: {fname}. Replacing the RIR with a fake rir')
                this_rir = torch.zeros(size=(this_rir.shape[-2], 4000))  # create a fake impulse
                this_rir[:, 500] = 1e-10
            else:
                print(f'WARNING: Something is wrong with this RIR: {fname}. Ignoring for now.')
                pass
            print('DO NOT train any nets with this')
        

        if self.max_length > 0:
            this_rir = utils.crop_or_pad(this_rir, self.max_length)

        return this_rir[self.rir_output_channels, ...]
    
    def __read_rir_lmdb__(self, fname: str, scene: str) -> torch.Tensor:
        '''
        Reads a rir (wav) file from a precomputed LMDB database.
        e.g.
            __read_rir_lmdb__('office_0/10_39.wav')
        '''
        ##print(f'inside read_rir_lmdb')   # TODO remove
        if scene in self.replica_scenes_names:
            with self.env.begin(write=False) as txn:
                key = fname.encode()
                value = txn.get(key)
                #tmp = np.frombuffer(value, dtype=np.float32)
                #print(tmp.shape)
                ##print(fname)  # TODO remove this, once lmdb is debugged
                ##print(value)  # TODO remove this, once lmdb is debugged
                rir = np.frombuffer(value, dtype=np.float32).reshape((len(self.rir_output_channels), -1))  # This MIGHT fail if the number of channels is wrong, Be very careful
                this_rir = torch.from_numpy(rir)
                fs = 44100  # harcoded fs for now
        elif scene in self.matterport_scenes_names:
            with self.scenes_rir_lmdbs[scene].begin(write=False) as txn:
                key = fname.encode()
                value = txn.get(key)
                #tmp = np.frombuffer(value, dtype=np.float32)
                #print(tmp.shape)
                ##print(fname)  # TODO remove this, once lmdb is debugged
                ##print(value)  # TODO remove this, once lmdb is debugged
                rir = np.frombuffer(value, dtype=np.float32)#.reshape((len(self.rir_output_channels),))  # hardcoded number of channels
                this_rir = torch.from_numpy(rir).reshape(9,-1)  # TODO: Hardcoded reshape, assuming 2nd order ambisonics
                fs = 16000  # harcoded fs for now
        elif match_pattern_scenes(scene):
            #print('****************************************************************************************')
            #print('yolo')
            #print(scene)
            if 'line' in scene:
                this_env = self.scenes_rir_lmdbs['lines']
            elif 'grid' in scene:
                this_env = self.scenes_rir_lmdbs['grids']
            else:
                print(this_env)
                raise ValueError(f'ERROR, I cannot read the LMDB for this scene {scene}')
            #print(this_env)
            with this_env.begin(write=False) as txn:
                key = fname.encode()
                value = txn.get(key)
                #tmp = np.frombuffer(value, dtype=np.float32)
                #print(tmp.shape)
                #print(fname)  # TODO remove this, once lmdb is debugged
                #print(value)  # TODO remove this, once lmdb is debugged
                if value is None:
                    print(f'ERROR: value is None')
                    print(f'ERROR: fname {scene}')
                try:
                    rir = np.frombuffer(value, dtype=np.float32).reshape((len(self.rir_output_channels),-1))  # hardcoded number of channels
                except Exception as e:
                    print(f'ERROR: fname {scene}')
                    raise e
                this_rir = torch.from_numpy(rir)#.reshape(9,-1)  # TODO: Hardcoded reshape, assuming 2nd order ambisonics
                fs = 24000  # harcoded fs for now
        
        if self.fs is None:
            self.fs = fs
        else:
            if self.fs != fs:
                warnings.warn(f'WARNING: I found different sampling rates when loading RIRs: {fs} and previous {self.fs}')

        if len(this_rir.shape) < 2:
            this_rir = this_rir.unsqueeze(0)  # shape is [channels, timesteps]

        # Sanity check
        #assert utils.validate_audio(this_rir), f'ERROR, the audio file seems wrong. {fname}'
        this_rir[torch.logical_not(torch.isfinite(this_rir))] = 0.0
        if not utils.validate_audio(this_rir):
            print(f'WARNING: Something is wrong with this RIR: {fname}. Ignoring for now')
            print('DO NOT train any nets with this')
            #this_rir = torch.zeros(size=(this_rir.shape[-2], 4000))
            #this_rir[:, 0] = 1e-5

        ##print(f'inside read_rir_lmdb, before cropping')   # TODO remove
        ##print(f'rir shape = {rir.shape }')  # TODO remove
        if self.max_length > 0:
            this_rir = utils.crop_or_pad(this_rir, self.max_length)

        return this_rir[self.rir_output_channels, ...]

    def __read_maps_lmdb__(self, fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads a floormap and acumap from a precomputed LMDB file.
        e.g.
            __read_rir_lmdb__('office_0/10_39.wav')

        NOTE: Floormaps and acumaps do not depend on the rec, only scene/src

        NOTE: We hard code the resolution 128x128 so that we can do the reshaping.
        """
        with self.env_maps.begin(write=False) as txn:
            key = fname.rsplit("_")[-2]  # frl_apartment_1/22_13  --> frl_apartment_1/22
            key = fname.rsplit('_', 1)[0] # frl_apartment_1/22_13  --> frl_apartment_1/22
            #print(key)
            key = key.encode(encoding='utf-8')

            try:
                value = txn.get(key)
                value = pickle.loads(value)
                floormap = np.frombuffer(value[0], dtype=np.float32)
                acumap = np.frombuffer(value[1], dtype=np.float64)  # Acumaps are float64 for better precision
                #print('DEBUGGGG rgiht before loading maps lmdb')
                floormap = torch.from_numpy(floormap).reshape([-1, 128, 128])  # Hardcoded 128x128 resoltipn
                #print('DEBUGGGG rgiht after loading floormaps lmdb')
                acumap = torch.from_numpy(acumap).reshape([-1, 128, 128])
            except Exception as e:
                print(f'ERROR reading maps from lmdb with key: {fname}')
                print(self.env_maps.info())
                print(self.env_maps.path())
                raise e
        #print('DEBUGGGG rgiht after loading maps lmdb')
        return floormap, acumap
    
    def __read_maps_lmdb__OLD(self, fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads a floormap and acumap from a precomputed LMDB file.
        e.g.
            __read_rir_lmdb__('office_0/10_39.wav')

        NOTE: Floormaps and acumaps do not depend on the rec, only scene/src

        NOTE: We hard code the resolution 128x128 so that we can do the reshaping.
        """
        with self.env_maps.begin(write=False) as txn:
            key = fname.split("_")[-2]  # frl_apartment_1/22_13  --> frl_apartment_1/22
            key = key.encode(encoding='utf-8')

            value = txn.get(key)
            value = pickle.loads(value)
            floormap = np.frombuffer(value[0], dtype=np.float32)
            acumap = np.frombuffer(value[1], dtype=np.float32)
            floormap = torch.from_numpy(floormap).reshape([-1, 128, 128])  # Hardcoded 128x128 resoltipn
            acumap = torch.from_numpy(acumap).reshape([-1, 128, 128])

        return floormap, acumap

    def __repr__(self):
        fmt_str = '\nDataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of unique RIRs files: {}\n'.format(len(self.fnames))
        fmt_str += '    Root Location: {}\n'.format(self.datapath)
        #fmt_str += '    Subset: {}\n'.format(self.subset)
        fmt_str += '    n_scenes: {}\n'.format(len(self.scenes))
        if len(self.scenes) < 20:
            fmt_str += '    scenes: {}\n'.format(self.scenes)
        else:
            fmt_str += '    scenes: {} ...\n'.format(self.scenes[0:10])
        fmt_str += '    n_files_per_scene: {}\n'.format(self.n_files_per_scene)
        fmt_str += '    max_length: {}\n'.format(self.max_length)
        fmt_str += '    rir_output_channels: {}\n'.format(self.rir_output_channels)
        fmt_str += '    multi_story_removal: {}\n'.format(self.multi_story_removal)
        if self.multi_story_removal:
            fmt_str += '    max_height_distance: {}\n'.format(self.max_height_distance_for_multi_story)
        if self.return_rot_angle:
            fmt_str += '    return_rot_angle: {}\n'.format(self.return_rot_angle)
        fmt_str += '    read_lmdb: {}\n'.format(self.read_lmdb)
        if self.read_lmdb:
            fmt_str += '    lmdb: {}\n'.format(self.fname_lmdb)
        fmt_str += '    read_lmdb_maps: {}\n'.format(self.read_lmdb_maps)
        if self.read_lmdb_maps:
            fmt_str += '    lmdb_maps: {}\n'.format(self.fname_lmdb_maps)
        
        fmt_str += '    fs: {}\n'.format(self.fs)

        if self.floormap_processor is not None:
            fmt_str += ''
            fmt_str += str(self.floormap_processor)
            fmt_str += ''

        if self.acumap_processor is not None:
            fmt_str += ''
            fmt_str += str(self.acumap_processor)
            fmt_str += ''
        return fmt_str

    def __getitem__(self, item) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #print('DEBUGGGG right at thet beginnign of getitem')
        fname = self.fnames[item]

        src = self.data_src[fname]
        rec = self.data_rec[fname]
        src_id, _ = self.parse_fname(fname)
        scene = self.data_scenes[item]

        if self.read_rirs:
            if self.read_lmdb:
                rir = self.__read_rir_lmdb__(fname, scene=scene)
            elif fname not in self.data_rirs:
                rir = self.__read_rir__(fname, scene=scene)
                if not self.avoid_storing_rirs_locally:  # WARNING Only use when dumping to LMDB
                    self.data_rirs[fname] = rir
            else:
                rir = self.data_rirs[fname]
        else:
            rir = torch.zeros(1)

        if len(rir.shape) < 2:
            rir = rir[None, :]  # Add channel dimension if needed

        # Reading maps, from LMDB or computing into memory
        if self.read_lmdb_maps:
            floormap, acumap = self.__read_maps_lmdb__(fname)
        else:
            if self.read_floormaps:
                if scene not in self.data_floormaps:
                    floormap = self.__extract_floormap__(scene)
                    if not self.avoid_storing_maps_locally:
                        self.data_floormaps[scene] = floormap
                else:
                    floormap = self.data_floormaps[scene]
            else:
                floormap = None

            if self.read_acumaps:
                key = self.get_acumap_key(scene, src_id)
                if key not in self.data_acumaps:
                    acumap = self.__extract_acumap__(scene, src_id)
                    if not self.avoid_storing_maps_locally:
                        self.data_acumaps[key] = acumap
                else:
                    acumap = self.data_acumaps[key]
            else:
                if floormap is not None:
                    acumap = torch.zeros_like(floormap)
                else:
                    acumap = torch.zeros(1,1,1)

        if self.do_precentering:
            src = src - self.data_centroids_per_scene[scene]
            rec = rec - self.data_centroids_per_scene[scene]

        # Fixed masks, than can be use to split train/test with the same scenes but different areas
        if self.use_fixed_masks:
            key = self.get_acumap_key(scene, src_id)
            if key not in self.data_masks_acuamps:
                mask = None
            else:
                mask = self.data_masks_acuamps[key]

        if self.augmentation_transform is not None:
            if self.use_fixed_masks:
                src, rec, floormap, acumap, mask = self.augmentation_transform(src, rec, floormap, acumap, mask)
            elif self.return_rot_angle:  # NEW src_pose for easy diretional analysis of sources
                #src, rec, floormap, acumap, src_pose = self.augmentation_transform(src.clone(), rec.clone(), floormap, acumap, src_pose=src_pose)  
                # src_pose = torch.tensor(src_pose)
                src, rec, floormap, acumap, rot_angle = self.augmentation_transform(src.clone(), rec.clone(), floormap.clone(), acumap.clone())  # TODO debugging, this is not ok, but this works fine ignoring the pose
                rot_angle = torch.tensor(rot_angle)
                #src_pose = torch.tensor(1.0)
            else:
                src, rec, floormap, acumap = self.augmentation_transform(src.clone(), rec.clone(), floormap.clone(), acumap.clone())  # This works but it's super slow
                #src, rec, floormap, acumap = self.augmentation_transform(src, rec, floormap, acumap)  # This is wrong, the shape is nonsense
        elif self.return_rot_angle:
            rot_angle = torch.tensor(0.0)  # Not sure if this is ok

        # TODO fix this, should a modified tuple
        if self.use_fixed_masks:
            return fname, rir, src, rec, scene, floormap, acumap, mask
        if self.return_rot_angle:
            return fname, rir, src, rec, scene, floormap, acumap, rot_angle
        if self.read_acumaps:
            return fname, rir, src, rec, scene, floormap, acumap
        if self.read_floormaps:
            return fname, rir, src, rec, scene, floormap, acumap
        elif self.read_scenes:
            return fname, rir, src, rec, scene
        else:
            return fname, rir, src, rec

    def _remove(self, remove_list: List[int]):
        for item in remove_list:
            fname = self.fnames[item]

            # remove from dictionaries
            del self.data_src[fname]
            del self.data_rec[fname]
            if fname in self.data_rirs:
                del self.data_rirs[fname]
 
        # remove from lists
        self.data_scenes = np.delete(self.data_scenes, remove_list)
        self.data_src_ids = np.delete(self.data_src_ids, remove_list)
        self.data_rec_ids = np.delete(self.data_rec_ids, remove_list)
        self.fnames = np.delete(np.array(self.fnames), remove_list).tolist()
        
        self.__validate__()

    def __del__(self):
        if self.env is not None:
            self.env.close()

def split_by_recs(dataset_train: Dataset, dataset_test: Dataset, proportion_recs_test: float = 0.3, sampling='uniform', debug=False) -> (Dataset, Dataset):
    """Splits a dataset into two subsets, by keeping the same scenes but having different receivers
    for each split.
    This is the first method, which comptues the split with recs ids, but before the acuamps are extracted.
    So this is not very good, because there can be quite a bit of overlap in the acumaps."""

    if debug:
        proportion_recs_test = 0.5  # per scene, ignoring sources
        sampling = 'uniform'
        data_scenes = np.array([0,0,0,0,0,0, 1,1,1,1,1,1])
        data_rec_ids = np.array([0,1,2,3,4,5, 3,4,5,6,7,8])
        dataset_ids = np.arange(start=0, stop=len(data_scenes))
    else:
        data_scenes = dataset_train.data_scenes
        data_rec_ids = dataset_train.data_rec_ids
        dataset_ids = np.arange(start=0, stop=len(data_scenes))

    original_len = len(dataset_train)
    assert len(dataset_train) == len(dataset_test), 'ERROR, the split function requires the two datasets to be exactly the same before splitting.'
    assert 0.0 < proportion_recs_test < 1.0, f'ERROR, the proportion_recs_test should be between (0, 1), and not {proportion_recs_test}'
        
    test_ids = []
    for scene in tqdm(dataset_train.scenes, desc='\t\t Processing scenes'):
        mask_scene = data_scenes == scene  
        recs_this_scene = data_rec_ids[mask_scene]
        unique_recs_this_scene = np.unique(recs_this_scene) 
        this_n_recs = math.floor(len(unique_recs_this_scene) * proportion_recs_test)
        
        if sampling == 'uniform':
            selected_recs = np.random.choice(unique_recs_this_scene, this_n_recs, replace=False)  # uniform random sample
        if sampling == 'continuous':
            start_index = np.random.randint(0, len(unique_recs_this_scene) - this_n_recs + 1)  # Random continuous sample
            selected_recs = unique_recs_this_scene[start_index : start_index + this_n_recs]
        
        mask_recs = np.isin(recs_this_scene, selected_recs)
        tmp = dataset_ids[mask_scene] 
        tmp = tmp[mask_recs]
        test_ids.extend(tmp.tolist())
    
    if debug:
        print(test_ids)
    
    mask = torch.zeros(len(dataset_train), dtype=torch.bool)
    mask[test_ids] = True
    train_ids = dataset_ids[~mask]

    dataset_train._remove(test_ids)
    dataset_test._remove(train_ids)
        
    assert len(dataset_train) + len(dataset_test) == original_len, 'ERROR, the size of the datastets after split should be the same as before split.'
    assert len(dataset_train) > 0 and len(dataset_test) > 0, 'ERROR, one of the subsets has no items after splitting the dataset by receivers'

    return dataset_train, dataset_test

def split_by_srcs(dataset_train: Dataset, dataset_test: Dataset, proportion_srcs_test: float = 0.3, sampling='uniform', debug=False) -> (Dataset, Dataset):
    """Splits a dataset into two subsets, by keeping the same scenes but having different sources
    for each split."""

    if debug:
        proportion_srcs_test = 0.5  # per scene, ignoring sources
        sampling = 'uniform'
        data_scenes = np.array([0,0,0,0,0,0, 1,1,1,1,1,1])
        data_src_ids = np.array([0,1,2,3,4,5, 3,4,5,6,7,8])
        dataset_ids = np.arange(start=0, stop=len(data_scenes))
    else:
        data_scenes = dataset_train.data_scenes
        data_src_ids = dataset_train.data_src_ids
        dataset_ids = np.arange(start=0, stop=len(data_scenes))

    original_len = len(dataset_train)
    assert len(dataset_train) == len(dataset_test), 'ERROR, the split function requires the two datasets to be exactly the same before splitting.'
    assert 0.0 < proportion_srcs_test < 1.0, f'ERROR, the proportion_srcs_test should be between (0, 1), and not {proportion_srcs_test}'
        
    test_ids = []
    for scene in tqdm(dataset_train.scenes, desc='\t\t Processing scenes'):
        mask_scene = data_scenes == scene  
        scrs_this_scene = data_src_ids[mask_scene]
        unique_scrs_this_scene = np.unique(scrs_this_scene) 
        this_n_srcs = math.floor(len(unique_scrs_this_scene) * proportion_srcs_test)
        
        if sampling == 'uniform':
            selected_scrs = np.random.choice(unique_scrs_this_scene, this_n_srcs, replace=False)  # uniform random sample
        if sampling == 'continuous':
            start_index = np.random.randint(0, len(unique_scrs_this_scene) - this_n_srcs + 1)  # Random continuous sample
            selected_scrs = unique_scrs_this_scene[start_index : start_index + this_n_srcs]
        
        selected_scrs = np.isin(scrs_this_scene, selected_scrs)
        tmp = dataset_ids[mask_scene] 
        tmp = tmp[selected_scrs]
        test_ids.extend(tmp.tolist())
    
    if debug:
        print(test_ids)
    
    mask = torch.zeros(len(dataset_train), dtype=torch.bool)
    mask[test_ids] = True
    train_ids = dataset_ids[~mask]

    dataset_train._remove(test_ids)
    dataset_test._remove(train_ids)
        
    assert len(dataset_train) + len(dataset_test) == original_len, 'ERROR, the size of the datastets after split should be the same as before split.'
    assert len(dataset_train) > 0 and len(dataset_test) > 0, 'ERROR, one of the subsets has no items after splitting the dataset by receivers'

    return dataset_train, dataset_test

def test_dataset():
    print('>>>>>>>>>>>>>>>>> Testing Basic Dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    params = {'datapath': '/m/triton/cs/sequentialml/datasets/soundspaces/data',  # '/home/ricfalcon/00data/soundspaces/data',
              'directory_geometry': '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply', # '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
              'directory_jsons': '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}', # '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
              'directory_lmdb': '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb',
              'scenes': ['apartment_2'],
              'n_files_per_scene': 10,  
              'read_lmdb': True,
              'fname_lmdb': 'rirs_2ndorder_scenes_18.lmdb',  # rirs_mono_scenes_17.lmdb
              'rir_output_channels': [0,1,2,3,4,5,6,7,8],
              'fs': 48000}
    
    dataset = SoundspacesDataset(datapath=params['datapath'],
                                 directory_geometry=params['directory_geometry'],
                                 directory_jsons=params['directory_jsons'],
                                 directory_lmdb=params['directory_lmdb'],
                                 scenes=params['scenes'],
                                 n_files_per_scene=params['n_files_per_scene'],
                                 read_rirs=True,
                                 read_scenes=True,
                                 read_lmdb=params['read_lmdb'],
                                 fname_lmdb=params['fname_lmdb'],
                                 rir_output_channels=params['rir_output_channels'],
                                 max_length=-1)

    loss_fn = losses.MultiComplexSTFTLoss()
    for i in range(5):
        print(dataset[i])
    
    loss_0_0 = loss_fn(dataset[0][1], dataset[0][1])
    loss_0_1 = loss_fn(dataset[0][1], dataset[1][1])
    loss_1_0 = loss_fn(dataset[1][1], dataset[0][1])

    assert torch.isclose(loss_0_0, torch.zeros(1), atol=1e-8), 'ERROR, MSTFT loss should be 0 for the same RIR'
    assert loss_1_0 > 0.0, 'ERROR, MSTFT loss should be larger than 0 for different RIRs'
    assert torch.isclose(loss_0_1, loss_1_0, atol=1e-8), 'ERROR, MSTFT loss should be symmetrical'

    fname, rir, src, rec, scene = dataset[0]

    print('\n Shapes')
    print(f'rir: {rir.shape}')
    print(f'src: {src.shape}')
    print(f'rec: {rec.shape}')
    print(f'fs: {dataset.fs}')
    
    assert fname is not None and isinstance(fname, str), 'ERROR, fname should be a string'
    assert rir is not None and isinstance(rir, torch.Tensor), 'ERROR, rir should be a tensor'
    assert src is not None and isinstance(src,torch.Tensor) and list(src.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert rec is not None and isinstance(rec, torch.Tensor) and list(rec.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert scene is not None and isinstance(scene, str) and scene in dataset.replica_scenes_names, 'ERROR, the scene name should be a valid scene'

    print('>>>>>>>>>>>>>>>>> Unit test success!!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('')
    print('')
    print('')

    return 0

def test_floormap_processor():
    print('>>>>>>>>>>>>>>>>> Testing Dataset with FloormapProcessor <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    import features
    params = {'datapath': '/m/triton/cs/sequentialml/datasets/soundspaces/data', # '/home/ricfalcon/00data/soundspaces/data',
              'directory_geometry': '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply', # '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
              'directory_jsons': '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}', # '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
         'scenes': ['office_1'],
         'n_files_per_scene': 10,  
         'fs': 48000}
    params_processor = {'resolution': 100,
                        'height_selection': ['relative_ceiling'],
                        'slice_coord': 0.5}
    params_acu_processor = {'resolution': 100,
                            'k': 5,
                            's': 1,
                            'p': 2,
                            'std': 1}

    processor = features.FloorMapProcessor(resolution=params_processor['resolution'], 
                                           height_selection=params_processor['height_selection'], 
                                           slice_coord=params_processor['slice_coord'])
    acumap_processor = features.AcuMapProcessor(parameters=['c50', 't30', 'drr'], freq_bands=[250, 1000, 4000], 
                                                resolution=params_acu_processor['resolution'],
                                                use_pooling=True, use_lowpass=True,
                                                pooling_kernel=params_acu_processor['k'], 
                                                pooling_stride=params_acu_processor['s'], 
                                                pooling_padding=params_acu_processor['p'], 
                                                lowpass_std=params_acu_processor['std'])
    
    dataset = SoundspacesDataset(datapath=params['datapath'],
                                 directory_geometry=params['directory_geometry'],
                                 directory_jsons=params['directory_jsons'],
                                 scenes=params['scenes'],
                                 n_files_per_scene=params['n_files_per_scene'],
                                 read_rirs=True,
                                 read_scenes=True,
                                 read_floormaps=True,
                                 read_acumaps=True,
                                 floormap_processor=processor,
                                 acumap_processor=acumap_processor,
                                 max_length=96000,
                                 rir_output_channels=[i for i in range(9)])  # Change the ambisonic order

    loss_fn = losses.MultiComplexSTFTLoss()
    for i in range(5):
        print(dataset[i])
    
    loss_0_0 = loss_fn(dataset[0][1], dataset[0][1])
    loss_0_1 = loss_fn(dataset[0][1], dataset[1][1])
    loss_1_0 = loss_fn(dataset[1][1], dataset[0][1])

    assert torch.isclose(loss_0_0, torch.zeros(1), atol=1e-8), 'ERROR, MSTFT loss should be 0 for the same RIR'
    assert loss_1_0 > 0.0, 'ERROR, MSTFT loss should be larger than 0 for different RIRs'
    assert torch.isclose(loss_0_1, loss_1_0, atol=1e-8), 'ERROR, MSTFT loss should be symmetrical'

    fname, rir, src, rec, scene, floormap, acumap = dataset[0]
    floormap = processor.add_src_rec(floormap, src, rec)

    print('\n Shapes')
    print(f'rir: {rir.shape}')
    print(f'src: {src.shape}')
    print(f'rec: {rec.shape}')
    print(f'floormap: {floormap.shape}')
    print(f'acumap: {acumap.shape}')
    print(f'fs: {dataset.fs}')
    
    assert fname is not None and isinstance(fname, str), 'ERROR, fname should be a string'
    assert rir is not None and isinstance(rir, torch.Tensor), 'ERROR, rir should be a tensor'
    assert src is not None and isinstance(src,torch.Tensor) and list(src.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert rec is not None and isinstance(rec, torch.Tensor) and list(rec.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert floormap is not None and isinstance(floormap, torch.Tensor) and list(floormap.shape) == [4, params_processor['resolution'], params_processor['resolution']] , 'ERROR, floormap should be a tensor with shape [4, 100, 100]'
    assert acumap is not None and isinstance(acumap, torch.Tensor) and list(acumap.shape) == [9, params_acu_processor['resolution'], params_acu_processor['resolution']] , 'ERROR, acumap should be a tensor with shape [9, 100, 100]'
    assert scene is not None and isinstance(scene, str) and scene in dataset.replica_scenes_names, 'ERROR, the scene name should be a valid scene'

    mask = torch.logical_not(torch.isnan(floormap)) 
    assert mask.sum().item() > 0, 'ERROR, floormap is all NaNs'
    mask = torch.logical_not(torch.isnan(acumap)) 
    assert mask.sum().item() > 0, 'ERROR, acumap is all NaNs'


    # Testing removing some keys
    previous_len = len(dataset)
    keys_to_remvove = [1, 3]
    dataset._remove(keys_to_remvove)
    assert len(dataset) == previous_len - len(keys_to_remvove), 'ERROR, the removing of itmes was wrong'

    return 0

def test_split_receivers_per_scene():
    # Testing spliting receivers within scene
    # This is based on pruning the list of receivers, before computing acumaps
    # So its not very accurate
    print('')
    print('')
    print('>>>>>>>>>>>>>>>>> Testing Split by Recs <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    params = {'datapath': '/m/triton/cs/sequentialml/datasets/soundspaces/data', # '/home/ricfalcon/00data/soundspaces/data',
            'directory_geometry': '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply', # '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
            'directory_jsons': '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}', # '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
            'scenes': ['office_1'],
            'n_files_per_scene': 10,  
            'fs': 48000}
    params_processor = {'resolution': 100,
                        'height_selection': ['relative_ceiling'],
                        'slice_coord': 0.5}
    params_acu_processor = {'resolution': 100,
                            'k': 5,
                            's': 1,
                            'p': 2,
                            'std': 1}

    processor = features.FloorMapProcessor(resolution=params_processor['resolution'], 
                                           height_selection=params_processor['height_selection'], 
                                           slice_coord=params_processor['slice_coord'])
    acumap_processor = features.AcuMapProcessor(parameters=['c50', 't30', 'drr'], freq_bands=[250, 1000, 4000], 
                                                resolution=params_acu_processor['resolution'],
                                                use_pooling=True, use_lowpass=True,
                                                pooling_kernel=params_acu_processor['k'], 
                                                pooling_stride=params_acu_processor['s'], 
                                                pooling_padding=params_acu_processor['p'], 
                                                lowpass_std=params_acu_processor['std'])
    
    dataset_train = SoundspacesDataset(datapath=params['datapath'],
                                      scenes=params['scenes'],
                                      n_files_per_scene=params['n_files_per_scene'],
                                      read_rirs=True,
                                      read_scenes=True,
                                      read_floormaps=True,
                                      read_acumaps=True,
                                      floormap_processor=processor,
                                      acumap_processor=acumap_processor,
                                      max_length=96000,
                                      rir_output_channels=[i for i in range(9)])  # Change the ambisonic order
    dataset_test = SoundspacesDataset(datapath=params['datapath'],
                                      scenes=params['scenes'],
                                      n_files_per_scene=params['n_files_per_scene'],
                                      read_rirs=True,
                                      read_scenes=True,
                                      read_floormaps=True,
                                      read_acumaps=True,
                                      floormap_processor=processor,
                                      acumap_processor=acumap_processor,
                                      max_length=96000,
                                      rir_output_channels=[i for i in range(9)])  # Change the ambisonic order
    
    previous_len = len(dataset_train)
    train, test = split_by_recs(dataset_train=dataset_train, dataset_test=dataset_test, proportion_recs_test=0.8, sampling='continuous')
    print(f'len(train_subset) : {len(train)}')
    print(f'len(test_subset) : {len(test)}')
    print(f' print(train.data_rec_ids): {train.data_rec_ids}')
    print(f' print(test.data_rec_ids): {test.data_rec_ids}')
    #print(f' print(train.data_rec_ids): {[train.dataset.data_rec_ids[i] for i in train.indices]}')
    #print(f' print(test.data_rec_ids): {[test.dataset.data_rec_ids[i] for i in test.indices]}')

    print('>>>>>>>>>>>>>>>>> Testing Split by Scrs <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    dataset_train = SoundspacesDataset(datapath=params['datapath'],
                                       directory_geometry=params['directory_geometry'],
                                       directory_jsons=params['directory_jsons'],
                                      scenes=params['scenes'],
                                      n_files_per_scene=200,
                                      read_rirs=True,
                                      read_scenes=True,
                                      read_floormaps=True,
                                      read_acumaps=True,
                                      floormap_processor=processor,
                                      acumap_processor=acumap_processor,
                                      max_length=96000,
                                      rir_output_channels=[i for i in range(9)])  # Change the ambisonic order
    dataset_test = SoundspacesDataset(datapath=params['datapath'],
                                      directory_geometry=params['directory_geometry'],
                                      directory_jsons=params['directory_jsons'],
                                      scenes=params['scenes'],
                                      n_files_per_scene=200,
                                      read_rirs=True,
                                      read_scenes=True,
                                      read_floormaps=True,
                                      read_acumaps=True,
                                      floormap_processor=processor,
                                      acumap_processor=acumap_processor,
                                      max_length=96000,
                                      rir_output_channels=[i for i in range(9)])  # Change the ambisonic order
    
    train, test = split_by_srcs(dataset_train=dataset_train, dataset_test=dataset_test, proportion_srcs_test=0.5, sampling='continuous')
    print(f'len(train_subset) : {len(train)}')
    print(f'len(test_subset) : {len(test)}')
    print(f' print(train.data_scr_ids): {train.data_src_ids}')
    print(f' print(test.data_scr_ids): {test.data_src_ids}')
    
    
    print('')
    print('>>>>>>>>>>>>>>>>> Unit test success!!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    return 0

def test_matterport_scenes():
    print('>>>>>>>>>>>>>>>>> Testing Dataset with Matterport Scenes <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    import features
    params = {'datapath': '/m/triton/cs/sequentialml/datasets/soundspaces/data', # '/home/ricfalcon/00data/soundspaces/data',
              'directory_geometry': '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply', # '/home/ricfalcon/00data/replica/{:s}/mesh.ply'
              'directory_geometry_matterport': '/m/triton/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply', # scene_name, scene_name
              'directory_jsons': '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}', # '/home/ricfalcon/00data/soundspaces_processed/replica/{:s}'
              'directory_jsons_matterport': '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}',  # scene_name
              'scenes': ['pLe4wQe7qrG'],
              'n_files_per_scene': 100,  
              'fs': 48000}
    params_processor = {'resolution': 128,
                        'height_selection': ['relative_center'],
                        'slice_coord': 0.5,
                        'xlim': [-25, 25]}
    params_acu_processor = {'resolution': 128,
                            'k': 5,
                            's': 1,
                            'p': 2,
                            'std': 1}

    processor = features.FloorMapProcessor(resolution=params_processor['resolution'], 
                                           height_selection=params_processor['height_selection'], 
                                           slice_coord=params_processor['slice_coord'],
                                           xlim=params_processor['xlim'])
    acumap_processor = features.AcuMapProcessor(parameters=['c50', 't30', 'drr'], freq_bands=[250, 1000, 4000], 
                                                resolution=params_acu_processor['resolution'],
                                                use_pooling=True, use_lowpass=True,
                                                pooling_kernel=params_acu_processor['k'], 
                                                pooling_stride=params_acu_processor['s'], 
                                                pooling_padding=params_acu_processor['p'], 
                                                lowpass_std=params_acu_processor['std'],
                                                xlim=params_processor['xlim'])
    print('beofre datataset')
    dataset = SoundspacesDataset(datapath=params['datapath'],
                                 directory_geometry=params['directory_geometry'],
                                 directory_geometry_matterport=params['directory_geometry_matterport'],
                                 directory_jsons=params['directory_jsons'],
                                 directory_jsons_matterport=params['directory_jsons_matterport'],
                                 scenes=params['scenes'],
                                 n_files_per_scene=params['n_files_per_scene'],
                                 read_lmdb=True,
                                 read_rirs=True,
                                 read_scenes=True,
                                 read_floormaps=True,
                                 read_acumaps=True,
                                 floormap_processor=processor,
                                 acumap_processor=acumap_processor,
                                 max_length=96000,
                                 rir_output_channels=[i for i in range(9)])  # Change the ambisonic order

    loss_fn = losses.MultiComplexSTFTLoss()
    for i in range(5):
        print(dataset[i])
    
    loss_0_0 = loss_fn(dataset[0][1], dataset[0][1])
    loss_0_1 = loss_fn(dataset[0][1], dataset[1][1])
    loss_1_0 = loss_fn(dataset[1][1], dataset[0][1])

    assert torch.isclose(loss_0_0, torch.zeros(1), atol=1e-8), 'ERROR, MSTFT loss should be 0 for the same RIR'
    assert loss_1_0 > 0.0, 'ERROR, MSTFT loss should be larger than 0 for different RIRs'
    assert torch.isclose(loss_0_1, loss_1_0, atol=1e-8), 'ERROR, MSTFT loss should be symmetrical'

    fname, rir, src, rec, scene, floormap, acumap = dataset[0]
    floormap = processor.add_src_rec(floormap, src, rec)

    print('\n Shapes')
    print(f'rir: {rir.shape}')
    print(f'src: {src.shape}')
    print(f'rec: {rec.shape}')
    print(f'floormap: {floormap.shape}')
    print(f'acumap: {acumap.shape}')
    print(f'fs: {dataset.fs}')
    
    assert fname is not None and isinstance(fname, str), 'ERROR, fname should be a string'
    assert rir is not None and isinstance(rir, torch.Tensor), 'ERROR, rir should be a tensor'
    assert src is not None and isinstance(src,torch.Tensor) and list(src.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert rec is not None and isinstance(rec, torch.Tensor) and list(rec.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert floormap is not None and isinstance(floormap, torch.Tensor) and list(floormap.shape) == [4, params_processor['resolution'], params_processor['resolution']] , 'ERROR, floormap should be a tensor with shape [4, 100, 100]'
    assert acumap is not None and isinstance(acumap, torch.Tensor) and list(acumap.shape) == [9, params_acu_processor['resolution'], params_acu_processor['resolution']] , f'ERROR, acumap should be a tensor with shape [9, 100, 100], but found {acumap.shape}'
    assert scene is not None and isinstance(scene, str) and scene in dataset.matterport_scenes_names, 'ERROR, the scene name should be a valid scene'

    mask = torch.logical_not(torch.isnan(floormap)) 
    assert mask.sum().item() > 0, 'ERROR, floormap is all NaNs'
    mask = torch.logical_not(torch.isnan(acumap)) 
    assert mask.sum().item() > 0, 'ERROR, acumap is all NaNs'


    # Testing removing some keys
    previous_len = len(dataset)
    keys_to_remvove = [1, 3]
    dataset._remove(keys_to_remvove)
    assert len(dataset) == previous_len - len(keys_to_remvove), 'ERROR, the removing of itmes was wrong'

    print('>>>>>>>>>>>>>>>>> Unit test success!!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('')
    print('')
    print('')

    return 0

def test_mras_scenes():
    print('>>>>>>>>>>>>>>>>> Testing Dataset with MRAS Scenes <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    import features
    params = {'datapath': '/mnt/audio/home/ricfalcon/data/soundspaces/data/',  #'/m/triton/cs/sequentialml/datasets/soundspaces/data', # '/home/ricfalcon/00data/soundspaces/data',
              'directory_jsons_mras': '/mnt/audio/home/ricfalcon/data/soundspaces_processed/mras/{:s}',  #'/m/triton/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}',  # scene_name
              'directory_geometry_mras': '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/combined.obj',  #'/m/triton/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/combined.obj',  #'/m/triton/cs/sequentialml/datasets/scenes_proposed/{:s}/combined.obj', # scene_name
              'directory_rir_mras': '/mnt/audio/home/ricfalcon/data/multiroom3/{:s}/outputs/',  #'/m/triton/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/outputs/',  # scene_name'
              'directory_lmdb': '/mnt/audio/home/ricfalcon/data/soundspaces_processed/lmdb',
              'n_files_per_scene': 100,  
              'fs': 48000}

    params['scenes'] = ['line0_materials_0']  # v2
    params['scenes'] = ['grid0_materials_0']  # v3
    params['scenes'] = ['line1_materials_0']  # v5
    params['scenes'] = ["grid" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]

    scenes_train = ['grid50_materials_0', 'grid50_materials_1', 'grid50_materials_3', 'grid50_materials_4',
                    'grid51_materials_0', 'grid51_materials_1',
                    'grid52_materials_0',
                    'grid53_materials_2',]
                    #'grid85_materials_1', 'grid85_materials_4', ]
                    #'grid86_materials_0', 
                    #]
    # grid that I am missing
    grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(84,100) for mater in range(5)]
    scenes_train = scenes_train + grid_list
    params['scenes'] = scenes_train   # 10:25

    
    params_processor = {'resolution': 128,
                        'height_selection': ['relative_center'],
                        'slice_coord': 0.5,
                        'xlim': [-10, 10]}
    params_acu_processor = {'resolution': 128,
                            'k': 5,
                            's': 1,
                            'p': 2,
                            'std': 1}

    processor = features.FloorMapProcessor(resolution=params_processor['resolution'], 
                                           height_selection=params_processor['height_selection'], 
                                           slice_coord=params_processor['slice_coord'],
                                           xlim=params_processor['xlim'])
    acumap_processor = features.AcuMapProcessor(parameters=['c50', 't30', 'drr'], freq_bands=[250, 1000, 4000], 
                                                resolution=params_acu_processor['resolution'],
                                                use_pooling=True, use_lowpass=True,
                                                pooling_kernel=params_acu_processor['k'], 
                                                pooling_stride=params_acu_processor['s'], 
                                                pooling_padding=params_acu_processor['p'], 
                                                lowpass_std=params_acu_processor['std'],
                                                xlim=params_processor['xlim'])
    
    dataset = SoundspacesDataset(datapath=params['datapath'],
                                 directory_geometry_mras=params['directory_geometry_mras'],
                                 directory_jsons_mras=params['directory_jsons_mras'],
                                 directory_rir_mras=params['directory_rir_mras'],
                                 directory_lmdb=params['directory_lmdb'],
                                 scenes=params['scenes'],
                                 n_files_per_scene=params['n_files_per_scene'],
                                 read_lmdb=True,
                                 read_rirs=True,
                                 read_scenes=True,
                                 read_floormaps=True,
                                 read_acumaps=False,
                                 floormap_processor=processor,
                                 acumap_processor=acumap_processor,
                                 max_length=96000,
                                 rir_output_channels=[i for i in range(1)])  # Change the ambisonic order

    loss_fn = losses.MultiComplexSTFTLoss()
    for i in range(5):
        print(dataset[i])
    
    loss_0_0 = loss_fn(dataset[0][1], dataset[0][1])
    loss_0_1 = loss_fn(dataset[0][1], dataset[1][1])
    loss_1_0 = loss_fn(dataset[1][1], dataset[0][1])

    assert torch.isclose(loss_0_0, torch.zeros(1), atol=1e-8), 'ERROR, MSTFT loss should be 0 for the same RIR'
    assert loss_1_0 > 0.0, 'ERROR, MSTFT loss should be larger than 0 for different RIRs'
    assert torch.isclose(loss_0_1, loss_1_0, atol=1e-8), 'ERROR, MSTFT loss should be symmetrical'

    fname, rir, src, rec, scene, floormap, acumap = dataset[0]
    floormap = processor.add_src_rec(floormap, src, rec)

    print('\n Shapes')
    print(f'rir: {rir.shape}')
    print(f'src: {src.shape}')
    print(f'rec: {rec.shape}')
    print(f'floormap: {floormap.shape}')
    print(f'acumap: {acumap.shape}')
    print(f'fs: {dataset.fs}')
    
    assert fname is not None and isinstance(fname, str), 'ERROR, fname should be a string'
    assert rir is not None and isinstance(rir, torch.Tensor), 'ERROR, rir should be a tensor'
    assert src is not None and isinstance(src,torch.Tensor) and list(src.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert rec is not None and isinstance(rec, torch.Tensor) and list(rec.shape) == [3] , 'ERROR, src should be a tensor with shape [3]'
    assert floormap is not None and isinstance(floormap, torch.Tensor) and list(floormap.shape) == [4, params_processor['resolution'], params_processor['resolution']] , 'ERROR, floormap should be a tensor with shape [4, 100, 100]'
    assert acumap is not None and isinstance(acumap, torch.Tensor) and list(acumap.shape) == [9, params_acu_processor['resolution'], params_acu_processor['resolution']] , f'ERROR, acumap should be a tensor with shape [9, 100, 100], but found {acumap.shape}'
    assert scene is not None and isinstance(scene, str) and scene in dataset.matterport_scenes_names, 'ERROR, the scene name should be a valid scene'

    mask = torch.logical_not(torch.isnan(floormap)) 
    assert mask.sum().item() > 0, 'ERROR, floormap is all NaNs'
    mask = torch.logical_not(torch.isnan(acumap)) 
    assert mask.sum().item() > 0, 'ERROR, acumap is all NaNs'


    # Testing removing some keys
    previous_len = len(dataset)
    keys_to_remvove = [1, 3]
    dataset._remove(keys_to_remvove)
    assert len(dataset) == previous_len - len(keys_to_remvove), 'ERROR, the removing of itmes was wrong'

    print('>>>>>>>>>>>>>>>>> Unit test success!!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('')
    print('')
    print('')

    return 0

def validate_whole_dataset_maps():
    """ Here we iterate the whole dataset and validate that the precomputed maps are ok.
    We want:
        - no infs
        - at least 1 pixel not nan

    NOTE: Run this in triton, cpu.
    """

    resolution = 128
    xlim = [-10, 10]  # [-10, 10] for replica , -25,25 for mp3d
    lowpass=True
    k, s, p = 9, 1, 4   # for matterport
    k, s, p = 5, 1, 2  # Standard
    #k, s, p = 21, 1, 10
    std = 1

    params = {}
    params['datapath'] = '/m/triton/cs/sequentialml/datasets/soundspaces/data'
    params['fname_lmdb_maps'] = 'mras_relfloor_10x10_standard.lmdb'  # 'maps_replica_relcenter_10x10.lmdb'  # 'maps_mp3d_relcenter_25x25.lmdb' #'maps_relcenter_50x50.lmdb'  # 'maps_debug_floor.lmdb'  # 'maps.lmdb'  # maps_relcenter.lmdb  # maps_relcenter_15x15.lmdb
    params['n_files_per_scene'] = 1e6  # 1e7
    params['fname_lmdb'] = 'rirs_mono_scenes_18.lmdb'  # 'rirs_mono_scenes_17.lmdb'  # 'rirs_2ndorder_test_18.lmdb

    directory_jsons = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name, # NOTE, apartment_0 has no spatial metrics, because the json creating failed
    directory_jsons_matterport = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
    directory_geometry = '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
    directory_lmdb = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb'
    directory_lmdb_maps = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
    directory_jsons_mras = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
    directory_geometry_mras = '/m/triton/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/combined.obj' # scene_name, scene_name
    directory_rir_mras = '/m/triton/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/outputs/'  # scene_name

    # scenes, MRAS
    grids_train = [f'grid{x}_materials_{y}' for x in range(100) for y in range(5)]
    lines_train = [f'line{x}_materials_{y}' for x in range(100) for y in range(5)]
    scenes_train = grids_train + lines_train

    floormap_processor = features.FloorMapProcessor(resolution=resolution, 
                                                    height_selection= 'relative_floor', # 'relative_center',  #'relative_ceiling', 
                                                    slice_coord=0.5,
                                                    pos_enc_d=8,
                                                    xlim=xlim,
                                                    use_soft_position=True)

    # 0.0, 72.0, 144.0, 216.0, 288.0, 360.0 
    acumap_processor = features.AcuMapProcessor(parameters=['c50', 't30', 't0', 'n_slopes'], freq_bands=[250, 1000, 4000],   # [250, 1000, 4000]   # ['c50', 't30', 't0', 'n_slopes'], freq_bands=[125, 250, 1000],   # [250, 1000, 4000]   # ['c50', 't30', 'drr', 'edt']  # ['0.0/drr', '360.0/drr', '0.0/c50', '360.0/c50']
                                                resolution=resolution,
                                                use_pooling=True, use_lowpass=True,
                                                pooling_kernel=k, 
                                                pooling_stride=s, 
                                                pooling_padding=p, 
                                                lowpass_std=std,
                                                xlim=xlim)
    translator_tform = None
        
    dset_train = SoundspacesDataset(datapath=params['datapath'],
                                    directory_geometry=directory_geometry,
                                    directory_jsons=directory_jsons,
                                    directory_lmdb=directory_lmdb,
                                    directory_lmdb_maps=directory_lmdb_maps,
                                    directory_jsons_matterport=directory_jsons_matterport,
                                    directory_geometry_mras=directory_geometry_mras,
                                    directory_jsons_mras=directory_jsons_mras,
                                    directory_rir_mras=directory_rir_mras,
                                    fname_lmdb=params['fname_lmdb'],
                                    fname_lmdb_maps=params['fname_lmdb_maps'],
                                    scenes=scenes_train,
                                    n_files_per_scene=params['n_files_per_scene'],
                                    read_rirs=False,  # activate to analyze RIRs
                                    read_scenes=True,
                                    read_floormaps=True,
                                    read_acumaps=True,   # NOTE: disable this if the rirs have not been processed
                                    read_lmdb=True,
                                    read_lmdb_maps=True,  # NOTE: Do not use lmdb maps if testing acumap processing
                                    max_length=24000,  # 48000  # -1 when analizing rirs, but will fail when using batches
                                    floormap_processor=floormap_processor,
                                    acumap_processor=acumap_processor,
                                    rir_output_channels=[0],  # 0,1,2,3,4,5,6,7,8]
                                    augmentation_transform=translator_tform,
                                    multi_story_removal=True)
    
    dataloader = torch.utils.data.DataLoader(dset_train, batch_size=128, shuffle=False, drop_last=False)

    floormaps_with_errors = []
    acumaps_with_errors = []
    for i, datum in tqdm(enumerate(dataloader)):    
        fname, rir, src, rec, scene, floormap, acumap = datum

        ind_inf = torch.where(torch.isinf(floormap))  # tuple of inds [b,c,h,w]
        #print(ind_inf)
        if len(ind_inf[0] > 0):
            floormaps_with_errors.append(fname[ind_inf[0]])
        ind_nan = torch.where(~torch.any(~torch.isnan(floormap), dim=(-3,-2,-1)))  # tuple of inds [b]
        #print(ind_nan)
        if len(ind_nan[0] > 0):
            floormaps_with_errors.append(fname[ind_nan[0]])
        ind_inf = torch.where(torch.isinf(acumap))  # tuple of inds [b,c,h,w]
        if len(ind_inf[0] > 0):
            acumaps_with_errors.append(fname[ind_inf[0]])
        ind_nan = torch.where(~torch.any(~torch.isnan(acumap), dim=(-3,-2,-1)))  # tuple of inds [b]
        if len(ind_nan[0] > 0):
            acumaps_with_errors.append(fname[ind_nan[0]])

    print(f'===============================================================')
    print('Floormaps:')
    print(floormaps_with_errors)

    print(f'===============================================================')
    print('Acumaps:')
    print(acumaps_with_errors)


if __name__ == '__main__':
    # Call it like this:
    # (conda-fedora) python -m datasets.soundspaces_dataset
    import losses
    import features
    utils.seed_everything(1111, 'balanced')
    #test_dataset()
    #test_floormap_processor()
    #test_split_receivers_per_scene()
    #test_matterport_scenes()
    test_mras_scenes()
    #validate_whole_dataset_maps()



