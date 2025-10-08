# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import time
import os
import json
import argparse
import warnings
from tqdm import tqdm
from typing import Dict, List
from torch.utils.data import Subset

from datasets.soundspaces_dataset import SoundspacesDataset, match_pattern_scenes
import pytorch_acoustics
import utils

# This is needed to avoid threads problem with ONNX
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'


def get_params(dataset_name: str = 'replica', 
               n_files_per_scene=2000, 
               scenes: List[int] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
               fold: str = 'all_lines',
               mode: str = 'debug') -> Dict:
    if dataset_name == 'replica':
        #warnings.warn(f'WARNING: Hardcoded scene for aprtment_0 only. Remove this to process the whole replica dataset')
        #scenes = ['apartment_0']

        params = {'datapath': '/home/ricfalcon/00data/soundspaces/data',
                  'scenes': scenes,
                  'n_files_per_scene': n_files_per_scene,
                  'max_length': 48000,  # replica 48000, mras 24000 ?
                  'fname_lmdb': 'rirs_2ndorder_test_18.lmdb',  # rirs_mono_scenes_17.lmdb  # 2ndorder is wrong
                  'channel_metrics': 0,
                  'do_decayfitnet': True,
                  'do_beamforming': True}
        
        warnings.warn(f'WARNING: Check the sampling frequency, right now it is harcoded to 16k for matterport')
    elif dataset_name == 'mp3d':
        
        # All scene matterport
        scenes = ['pLe4wQe7qrG', 'gTV8FGcVJC9', 'YFuZgdQ5vWj', 'oLBMNvg9in8', 'ZMojNkEp431', 'r47D5H71a5s', 'wc2JMjhGNzB', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'r1Q1Z4BcV1o', 'Vt2qJdWjCF2', 'JeFG25nYj2p', 'gYvKGZ5eRqb', 'x8F5xyUWy9e', 'E9uDoFAP3SH', 'ULsKaCPVFJR', 's8pcmisQ38h', 'q9vSo1VnCiC', 'V2XKFyX4ASd', 'EDJbREhghzL', 'p5wJjkQkbXX', 'kEZ7cmS4wCh', 'b8cTxDM8gDG', 'Z6MFQCViBuw', 'VLzqgDo317F', 'D7N2EKCX4Sj', 'sKLMLpTHeUy', 'pRbA3pwrgk9', 'sT4fr6TAbpF', '2n8kARJN3HM', '17DRP5sb8fy', 'rPc6DW4iMge', 'XcA2TqTSSAj', 'rqfALeAoiTq', '29hnd4uzFmX', 'YmJkqBEsHnH', 'D7G3Y4RVNrH', 'i5noydFURQK', 'qoiz87JEwZ2', 'B6ByNegPMKs', '1pXnuDYAj8r', 'JF19kD82Mey', 'Uxmj2M2itWa', 'mJXqzFtmKg4', '5LpN3gDmAk7', 'JmbYfDe2QKZ', '7y3sRwLe3Va', 'jtcxE69GiFV', 'SN83YJsR3w2', 'EU6Fwq7SyZv', 'X7HyMhZNoso', 'VzqfbhrpDEA', 'HxpKQynjfin', 'WYY7iVyf5p8', 'Vvot9Ly1tCj', 'QUCTc6BB5sX', 'yqstnuAEVhm', 'aayBHfsNo7d', '2azQ1b91cZZ', 'ARNzJeq3xxb', 'zsNo4HB9uLZ', 'uNb9QFRL6hY', '82sE5b5pLXE', '5q7pvUzZiYa', '759xd9YjKW5', 'PX4nDJXEHrG', '1LXtFkjw3qL', '8194nk5LbLH', 'UwV83HsGsw3', 'S9hNv5qa7GM', 'e9zR4mvMWw7', 'jh4fc5c5qoQ', 'Pm6F8kyY3z2', '8WUmhLawc2A', '5ZKStnWn8Zo', 'VVfe2KiqLaN', 'PuKPg4mmafe', 'ac26ZMwG7aT', 'fzynW3qQPVF', 'ur6pFq6Qu1A', 'GdvgFV5R1Z5', 'TbHJrupSAjP', 'gxdoqLR6rwA', 'pa4otMbVnkk', 'vyrNrziPKCB', 'gxdoqLR6rwA']

        # TEMPORARY, manually removing scenes from matterport that had errors
        scenes.remove('gTV8FGcVJC9')
        scenes.remove('wc2JMjhGNzB')
        scenes.remove('VFuaQ6m2Qom')
        scenes.remove('r1Q1Z4BcV1o')
        scenes.remove('Vt2qJdWjCF2')
        scenes.remove('vyrNrziPKCB')
        scenes.remove('gxdoqLR6rwA')

        if False:  # Debugging
            scenes = scenes[0:1]

        params = {'datapath': '/home/ricfalcon/00data/soundspaces/data',
                  'scenes': scenes,
                  'n_files_per_scene': n_files_per_scene,
                  'max_length': -1,
                  'fname_lmdb': 'rirs_mono_scenes_17.lmdb',
                  'channel_metrics': 0,
                  'do_decayfitnet': True,
                  'do_beamforming': True,
                  'fadeout_length': 0.5}
    elif dataset_name == 'mras':
        # DEBUGGING ONLY
        # Seleced scenes
        scenes = ['line56_materials_0', 'line44_materials_0', 'line0_materials_0', 'line4_materials_0', 'line53_materials_0']  # v2
        scenes = ['grid0_materials_0',  'grid11_materials_0',  'grid17_materials_1', 'grid0_materials_1', 'grid16_materials_3', 'line0_materials_0', 'line1_materials_0']  # v3
        scenes = ['grid21_materials_2']

        # for v5, with all scenes
        #fold = 'all_lines'
        if fold == 'all':
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            line_list = ["line" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            scenes =  grid_list + line_list
        elif fold == 'all_grids':
            grid_list = ["grid" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            scenes =  grid_list
        elif fold == 'all_lines':
            line_list = ["line" + str(i) + "_materials_" + str(mater) for i in range(100) for mater in range(5)]
            scenes =  line_list

        params = {'datapath': '/home/ricfalcon/00data/soundspaces/data',
                  'directory_jsons_mras': '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}',  # scene_name
                  'directory_geometry_mras': '/m/triton/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/combined.obj',
                  'scenes': scenes,
                  'n_files_per_scene': n_files_per_scene,
                  'max_length': -1,
                  'fname_lmdb': 'rirs_mono_mras_lines.lmdb',
                  'channel_metrics': 0,
                  'do_decayfitnet': True,
                  'do_beamforming': True,
                  'fadeout_length': 0.05}

        warnings.warn(f'WARNING: Check the sampling frequency, right now it is harcoded to 16k for matterport')
    else:
        warnings.warn(f'WARNING: Check the sampling frequency, right now it is harcoded to 16k for matterport')
        raise ValueError(f'ERROR: Unrecognized dataset {dataset_name}')
    return params

def get_dataset(dataset_name: str, params: Dict, use_triton=False, use_vrgpu=False) -> torch.utils.data.Dataset:
    if use_triton:
        params['datapath'] = '/scratch/cs/sequentialml/datasets/soundspaces/data/'
        directory_jsons = '/scratch/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_jsons_mras = '/scratch/cs/sequentialml/datasets/soundspaces_processed/mras/{:s}'  # scene_name
        directory_geometry = '/scratch/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport = '/scratch/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/scratch/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/combined.obj'
        directory_lmdb = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/scratch/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/scratch/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/scratch/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/outputs/'  # scene_name
    elif use_vrgpu:
        params['datapath'] = '/m/triton/cs/sequentialml/datasets/soundspaces/data'
        directory_jsons = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/replica/{:s}'  # scene_name
        directory_jsons_matterport = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/mp3d/{:s}'  # scene_name
        directory_geometry = '/m/triton/cs/sequentialml/datasets/replica/data/{:s}/mesh.ply'
        directory_geometry_matterport = '/m/triton/cs/sequentialml/datasets/matterport3d/data/v1/scans/{:s}/{:s}.ply' # scene_name, scene_name
        directory_geometry_mras = '/scratch/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/combined.obj'
        directory_lmdb = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb'
        directory_lmdb_maps = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/lmdb_maps'
        directory_rir_lmdb_per_scene = '/m/triton/cs/sequentialml/datasets/soundspaces_processed/data/lmdb'
        directory_rir_mras = '/m/triton/cs/sequentialml/datasets/scenes_proposed_v5/{:s}/outputs/'  # scene_name

    if dataset_name == 'replica':
        dset = SoundspacesDataset(datapath=params['datapath'],
                                  scenes=params['scenes'],
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
                                  n_files_per_scene=params['n_files_per_scene'],
                                  read_rirs=True,
                                  read_scenes=True,
                                  read_lmdb=True,
                                  read_floormaps=False,
                                  read_acumaps=False,
                                  fname_lmdb=params['fname_lmdb'],
                                  max_length=params['max_length'],
                                  multi_story_removal=True,
                                  avoid_storing_rirs_locally=True)
    elif dataset_name == 'mp3d':
        dset = SoundspacesDataset(datapath=params['datapath'],
                                  directory_geometry=directory_geometry,
                                  directory_geometry_matterport=directory_geometry_matterport,
                                  directory_jsons=directory_jsons,
                                  directory_jsons_matterport=directory_jsons_matterport,
                                  directory_lmdb=directory_lmdb,
                                  directory_lmdb_maps=directory_lmdb_maps,
                                  directory_rir_lmdb_per_scene=directory_rir_lmdb_per_scene,
                                  fname_lmdb=params['fname_lmdb'],
                                  #fname_lmdb_maps=params['fname_lmdb_maps'],
                                  scenes=params['scenes'],
                                  n_files_per_scene=params['n_files_per_scene'],
                                  read_rirs=True,
                                  read_scenes=True,
                                  read_floormaps=False,
                                  read_acumaps=False,
                                  read_lmdb=True,
                                  max_length=params['max_length'],
                                  multi_story_removal=True,
                                  avoid_storing_rirs_locally=True,
                                  )
    elif dataset_name == 'mras':
        dset = SoundspacesDataset(datapath=params['datapath'],
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
                                  #fname_lmdb_maps=params['fname_lmdb_maps'],
                                  scenes=params['scenes'],
                                  n_files_per_scene=params['n_files_per_scene'],
                                  read_rirs=True,
                                  read_scenes=True,
                                  read_floormaps=False,
                                  read_acumaps=False,
                                  read_lmdb=True,
                                  max_length=params['max_length'],
                                  avoid_storing_rirs_locally=True,
                                  rir_output_channels=[0]
                                  )

    return dset

@torch.no_grad()
def preprocess_metrics(dataset_name, n_files_per_scene, scenes, use_triton=False, use_vrgpu=False, chunk=0, chunk_size=1, fold='all_lines'):
    """ This reads all RIRs, extracts acoustic parms, and dumps them into JSON.
    There is 1 json file for each scene.
    """
    
    utils.seed_everything(1111, 'balanced')
    params = get_params(dataset_name=dataset_name, n_files_per_scene=n_files_per_scene, scenes=scenes, fold=fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2500  # To dump json in batches, and reduce memory usage

    for k,v in params.items():
        print(f'{k}: {v}')
        
    start_time = time.time()
    all_errors = []

    # Iterate scenes, we generate 1 json for each scene
    n = len(params['scenes'])
    start = chunk * chunk_size
    stop = min(start + chunk_size, n)
    assert start < n, f'ERROR, start id should smaller than the number of scenes, {start} > {n}'

    print('')
    print('Processing scenes: ')
    print(params['scenes'][start:stop])
    params['scenes'] = params['scenes'][start:stop]

    # Load dataset
    dataset = get_dataset(dataset_name, params, use_triton=use_triton, use_vrgpu=use_vrgpu)
    scenes_per_file = np.arange(start=0, stop=len(dataset))
    for scene in tqdm(dataset.scenes, position=0, desc='scenes'):

        metrics_per_scene = {}
        mask = dataset.data_scenes == scene
        ids = scenes_per_file[mask] 
        subset = Subset(dataset, ids)

        # Create the directory if needed
        ####tmp = os.path.dirname(dataset.directory_jsons)
        ####tmp = os.path.dirname(tmp)
        if scene in dataset.replica_scenes_names:
            output_directory = dataset.directory_jsons.format(scene)
        elif scene in dataset.matterport_scenes_names:
            output_directory = dataset.directory_jsons_matterport.format(scene)
        elif match_pattern_scenes(scene):
            output_directory = dataset.directory_jsons_mras.format(scene)
        output_filename = f'{scene}_metrics.json'

        print(f'Dumping json to {output_directory}')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if os.path.exists(os.path.join(output_directory, output_filename)):
             os.remove(os.path.join(output_directory, output_filename))
             
        print('')
        print(f'There are {len(subset)} datapoints in this scene {scene}.')
        print('Processing .....')
        do_last_write = False
        for ii, (fname, rir, src, rec, this_scene) in enumerate(tqdm(subset, position=1, leave=False)):
            rir = rir.to(device)
            try:
                assert this_scene == scene, f'ERROR, file scene {this_scene}, does not match scene {scene}. Is the masking done properly?'
                
                src_id, rec_id = dataset.parse_fname(fname)
                key = f'{scene}/{src_id}_{rec_id}'
    
                #print(f'fname: {fname}')
                #print(f'directory_jsons: {dset.directory_jsons}')
                #print(f'directory_jsons formatted: {dset.directory_jsons.format(scene, fname.split("/")[-1])}')

                if params['do_decayfitnet']:
                    #print(f'dataset fs {dataset.fs}')
                    metrics = pytorch_acoustics.get_metrics_decayfitnet(rir[params['channel_metrics'], :].to(torch.float64), dataset.fs,
                                                                        remove_direct_sound=True, fadeout_length=params['fadeout_length'], device=device)
                else:
                    metrics = pytorch_acoustics.get_metrics(rir[params['channel_metrics'], :].cpu().to(torch.float64), dataset.fs)

                # Final dictionary to dump as json
                tmp = {'fname': fname,
                       'scene': scene,
                       'src': src.tolist(),
                       'rec': rec.tolist(),
                       'channel': params['channel_metrics'],
                       'metrics': metrics}

                # Directional metrics
                if params['do_beamforming']:
                    if dataset_name in ['replica', 'mp3d']:
                        rotate_alternative_convention = False
                    else:
                        rotate_alternative_convention = True
                    W, angles = pytorch_acoustics.get_beamforming_matrix(w_pattern='maxre', sph_order=2, polygon_sides=6, 
                                                                        rotate_alternative_convention=rotate_alternative_convention)
                    angles = torch.rad2deg(angles)
                    rir_processed = torch.einsum('it,di->dt', rir.to(torch.float64), W.to(device))

                    for jj, _ in enumerate(range(rir_processed.shape[0])):
                        if params['do_decayfitnet']:  # disabled for not, this is too slow, and we dont really need multislopes per direction
                            this_metrics = pytorch_acoustics.get_metrics_decayfitnet(rir_processed[jj,:], dataset.fs, 
                                                                                    remove_direct_sound=True, fadeout_length=params['fadeout_length'])
                        else:
                            this_metrics = pytorch_acoustics.get_metrics(rir_processed[jj,:].cpu().to(torch.float64), dataset.fs)
                        tmp[f'metrics_{angles[jj]}'] = this_metrics
                metrics_per_scene[key] = tmp

                do_last_write = True
                #print(f'{ii}  {ii % batch_size}')
                # Dumping to json, by batches (This is very hacky)
                if ii % batch_size == 0:
                    print(f'\n ============================================')
                    print(f'Dumping partial json to {output_directory}')
                    with open(os.path.join(output_directory, output_filename), 'a') as f:  # mode 'a' for append
                        if ii == 0:  # at the beginning of file, write the opening bracket of JSON array
                            f.write('[')
                        else: # for other items, add a comma separator
                            f.write(',')
                        # dump the new dictionary
                        json.dump(metrics_per_scene, f)
                        print('----------- DUMPED')
                    metrics_per_scene = {}  # clear memory
                    do_last_write = False
                
            except Exception as e:
                # TODO: Reduce exception scope here
                print(f'ERROR getting metrics for fname {fname}, ignoring file.')
                print(e)
                #raise e  # only for debugging
                all_errors.append(fname)
                continue
    
        print(f'\n ========================================================')
        print(f'Dumping final json to {output_directory}')
        # close the json file
        with open(os.path.join(output_directory, output_filename), 'a') as f:  # mode 'a' for append
            if do_last_write: # for other items, add a comma separator
                f.write(',')
                json.dump(metrics_per_scene, f)
            f.write(']')  # close the json file

    stop_time = time.time()
    elapsed = stop_time - start_time
    print(f'\n\n Elapsed time = {elapsed}')
    print(f'\n\n Elapsed time = {time.strftime("%H:%M:%S", time.gmtime(elapsed))}')
    print(f'\n Files with errors = {len(all_errors)} / {len(dataset)}')

if __name__ == '__main__':
    print('Inside dataset_preprocess.py')
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_files_per_scene", help="Number of files per scens", type=int, default=1000)
    parser.add_argument("--scenes", nargs='+', help="List of scenes to process", type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    parser.add_argument("--use_triton", help="Flag to change data paths if using Aalto Triton cluster", action='store_true')
    parser.add_argument("--use_vrgpu", help="Flag to change data paths if using workstation vrgpu cluster", action='store_true')
    parser.add_argument("--dataset", help="Dataset name", type=str, choices=['replica', 'mp3d', 'mras'], default='replica')
    parser.add_argument('--chunk', default='0', type=int)
    parser.add_argument('--chunk-size', default=1, type=int)
    parser.add_argument("--fold", help="Dataset fold, for MRAS", type=str, choices=['all_lines', 'all_grids'], default='all_lines')
    args = parser.parse_args()

    #args.n_files_per_scene = 50  # only for debugging
    if args.use_triton and args.use_vrgpu:
        raise ValueError('ERROR: We cannot use both vrgpu and triton at the same time. Pick one cluster.')

    preprocess_metrics(args.dataset, args.n_files_per_scene, args.scenes, args.use_triton, args.use_vrgpu, chunk=args.chunk, chunk_size=args.chunk_size, fold=args.fold)
    
    
    
