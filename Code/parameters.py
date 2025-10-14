# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
# Helper on how to use the parser:
# https://github.com/bw2/ConfigArgParse

import argparse
import math
import os
import yaml
import shutil
from datetime import datetime
from typing import Dict
import configargparse

def get_result_dir_path(experiment_description: str, root: str = './results'):
    """Returns path where to save training results of a experiment specific result.

    Args:
        root: root path of where to save
        experiment_description: "epoch=50-batch=128-arch=FCN-data=FULL"

    Create the directory, "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    Return directory path(str):
        "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    """
    from datetime import datetime
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d_%H_%M_%S")

    path = f"{experiment_description}__{date_time}"
    path = os.path.join(root, path)
    try:
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            print("Path already exists")
        else:
            print(f"Couldn't create {path}.")
            path = root
    else:
        print(f"Save weights to {path}")
    finally:
        return path

def list_of_2d_tuples(s):
    '''
    For the argparser, this reads a string in the format:
    --max_pool 2,2 4,1 3,3
    And returns a list of tuples, as:
    [(2,2), (4,1), (3,3)]

    # Call it like this in terminal
    #--max_pool 2,2 4,1 3,3

    # debug:
    # config = parser.parse_args(["--max_pool", "2,2", "2,2", "2,2", "2,2", "3,3", "1,4"])  # test for the list_of_2d_tuples type
    '''
    try:
        yolo = iter(s.split(' '))
        for tmp in yolo:
            x,y = tmp.split(',')
            return int(x), int(y)
    except:
        raise argparse.ArgumentTypeError("Error reading parameters. Tuples must be x,y")

def get_parameters() -> Dict:
    #p = configargparse.ArgParser(default_config_files=['./configs/*.yaml'], config_file_parser_class=configargparse.YAMLConfigFileParser)
    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    #p = configargparse.ArgParser(default_config_files=['./configs/*.yaml'])
    p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path', default='./configs/train_default.yaml')

    if False:
        # Experiment
        p.add_argument('--exp_name', help="Optional experiment name.")
        p.add_argument('--exp_group', help="Optional experiment group, useful to search for runs in the logs. This is added to the exp name.")
        p.add_argument('--seed_mode', help="Mode for random seeds.", choices=['balanced', 'random'])
        p.add_argument('--seed', type=int)
        p.add_argument('--mode', help='train or eval', choices=['train', 'train_resume', 'valid', 'eval', 'baseline'])
        p.add_argument('--debug', action='store_true', help='Enables debug mode, with short runs and no logging.')
        p.add_argument('--profiling', action='store_true', help='Enables profiling mode.')
        p.add_argument('--job_id', type=str, default='', help='Job id to append to the experiment name. Helps getting the job log.')
        p.add_argument('--task_id', type=str, default='', help='Task id when using array jobs.')
        p.add_argument('--logging_dir', help='Directory to save logs and results.')
        p.add_argument("--use_dgx", help="Flag to change data paths if using DGX", action='store_true')

        # Training arguments
        p.add_argument('--epochs', type=int, help='Num of training iterations, or epochs.')
        p.add_argument('--batch', type=int, help='Batch size.')
        p.add_argument('--num_workers', type=int, help='Num workers for dataloader.')
        p.add_argument('--print_every', type=int, help='Print current status every x training iterations.')
        p.add_argument('--validation_interval', type=int, help='Validation interval')
        p.add_argument('--optimizer', type=str, help='Optimizer for training.', choices=['ranger', 'adam'])
        p.add_argument('--lr', type=float, help='Learning rate for optimizer.')
        p.add_argument('--weight_decay', type=float, help='Learning rate for optimizer.')

        # Mode and features:
        p.add_argument('--use_projector', action='store_true', help='Enables linear projection of STFT, not very useful.')
        p.add_argument('--use_stft_input', action='store_true', help='Add STFT as additional channels to input of the Unet.')
        p.add_argument('--use_dropout', action='store_true', help='Enables dropout.')
        p.add_argument('--use_film', action='store_true', help='Use Unet with FILM layer in bottleneck')
        p.add_argument('--remove_src_rec', action='store_true', help='Removes the src and rec from the input channels.')
        p.add_argument('--use_pos_enc_src', action='store_true', help='Enables positional encodings for the source.')
        p.add_argument('--use_pos_enc_rec', action='store_true', help='Enables positional encodings for the receiver..')
        p.add_argument('--use_multiple_rirs', action='store_true', help='Uses 2 RIRs as conditioning.')

        # Augmentation
        p.add_argument('--use_rotations', action='store_true', help='Enables random rotations.')
        p.add_argument('--use_translations', action='store_true', help='Enables random translations')
        p.add_argument('--use_translations_centered', action='store_true', help='Enables random rotations with centered boudning boxes of the scenes.')
        p.add_argument('--use_rotations_centered', action='store_true', help='Enables random translations where the bouding box of the scene is valid.')
        
        # Dataset
        p.add_argument('--use_split_by_src', action='store_true', help='Enables profiling mode.')
        p.add_argument('--use_split_by_recs', action='store_true', help='Enables profiling mode.')
    
    # Experiment setup
    p.add_argument("--exp_name", help="List of scenes to process", type=str, default='debug')
    p.add_argument("--exp_group", help="Group for the experiment fro wandb.", type=str, default='debug')
    p.add_argument('--seed_mode', help="Mode for random seeds.", choices=['balanced', 'random'])
    p.add_argument('--seed', type=int)
    p.add_argument("--use_dgx", help="Flag to change data paths if using DGX", action='store_true')
    p.add_argument("--use_triton", help="Flag to change data paths if using Aalto Triton cluster", action='store_true')
    p.add_argument("--use_vrgpu", help="Flag to change data paths if using the deskptop vrgpu in Aalto", action='store_true')
    p.add_argument("--num_workers", help="Number of workers for the ", type=int, default=2)
    p.add_argument('--task_id', type=str, default='', help='Task id when using array jobs.')
    p.add_argument("--job_id", help="Slurm job id", type=str, default='debug')
    p.add_argument('--wandb', action='store_true', help='Enable wandb to log runs.')
    p.add_argument("--comment", help="Free field to add to exp name", type=str, default='')

    # Optimization
    p.add_argument("--loss_function", help="Loss function used for training", type=str, default='l1', choices=['l1', 'sloped_l1', 'filterbank2d', 'wavelet', 'l2'])  # 

    # Dataset
    p.add_argument("--dataset", help="fold for the dataset", type=str, default='replica')
    p.add_argument("--fold", help="fold for the dataset", type=str, default='00')
    p.add_argument("--n_files_per_scene", help="Number of files per scens", type=int, default=2)
    p.add_argument("--max_length", help="Crop or pad rirs to this lenght, in samples. Use 48000 for replica, or 24000 for mras. -1 for no trimming", type=int, default=48000)
    p.add_argument("--read_lmdb", help="Read rirs from LMDB", action='store_true')
    p.add_argument("--fname_lmdb", help="Filenma of the lmdb file for rirs.", type=str, default='rirs_mono_scenes_18.lmdb')
    p.add_argument("--read_lmdb_maps", help="Read floormaps and acumaps from LMDB", action='store_true')
    p.add_argument("--fname_lmdb_maps", help="Filenma of the lmdb file for rirs.", type=str, default='maps_relcenter_10x10.lmdb')
    p.add_argument("--rir_output_channels", nargs='*', help="List of channels to use for the rirs, mono=[0], 1storder=[0,1,2,3], etc...", type=int)
    p.add_argument("--validation_random_sample", help="When computing baselines, takes a subsample from the validaiton set. This is because the validation can take forever", action='store_true')
    #p.add_argument("--scenes", nargs='*', help="List of scenes to process", type=int, default=[0])

    # Operation Mode
    p.add_argument('--do_baseline', action='store_true', help='Runs the selected baseline only.')
    p.add_argument('--do_validation', action='store_true', help='Runs the validation only, computes metrics, saves example outputs.')
    p.add_argument('--do_create_map_animation', action='store_true', help='Runs the animation mode, to create animation of acumaps.')
    p.add_argument('--do_animation_channel', help='When running animation mode, selects the channel to plot.', type=int, default=4)
    p.add_argument('--do_continue_training', action='store_true', help='Loads a checkpoint and resumes training. Might fail if the network is different..')
    p.add_argument('--validation_checkpoint', type=str, help='Directory of the model used for validation')
    p.add_argument('--do_precompute_rirs_lmdb', action='store_true', help='Precomputes the lmdb for the RIRs.')
    p.add_argument('--do_precompute_maps_lmdb', action='store_true', help='Precompute the lmdb for the floormaps and acumaps.')
    # baseline_methods = ['average_rir_all', 'average_rir_same_scene', 'average_acumap_scene', 'same_rir_NN',  'average_rir_same_scene_and_source', 'sample_acumaps_scene']
    p.add_argument("--baseline_method", help="baseline method", type=str)
    p.add_argument("--do_baseline_spatial", help="Computes spatial metrics for the baselines", action='store_true')
    p.add_argument("--do_baseline_spatial_force_omni", help="Computes spatial metrics for the baselines, using only the omni channel.", action='store_true')
    p.add_argument("--do_baseline_multislope", help="Computes multislope metrics for the baselines, using decayfitnet", action='store_true')
    p.add_argument("--use_within_room_generalization_method", help="Here we split the dataset with the same scens in train/test, but different receivers.", type=str, choices=['none', 'uniform', 'uniform_inras', 'neighbors', 'regions'])
    p.add_argument("--use_within_room_generalization_threshold", help="Threshold to split train/test receivers.", type=float, default=0.9)

    # Floormaps
    p.add_argument("--fmap_resolution", help="Resolution in pixels for the floormaps, usually 128", type=int)
    p.add_argument("--fmap_height_selection", help="Mode for the selection of height for the floormap slices", type=str, nargs='+')
    p.add_argument("--fmap_slice_coord", help="Coordinate value to use when selecting a slice of the floormap", type=float)
    p.add_argument("--fmap_use_slices_variance", help="Adds a channel with the variace per pixel for the slices", action='store_true')
    p.add_argument("--fmap_bbox_channel", help="Channel to use as reference when computing boudning boxes. Should be the mask channel after all slices", type=int)
    p.add_argument("--fmap_pos_enc_d", help="Dimensionality for the positonal encodings, if they are needed.", type=int)
    p.add_argument("--fmap_xlim", nargs='*', help="Limits in meters for the maps. Typical value is [-10, 10] for replica and MRAS, or [-25, 25] for mp3d", type=int)
    p.add_argument("--fmap_use_soft_sources", help="Enables soft positions for source and receiver.", action='store_true')
    p.add_argument("--fmap_add_pose", help="Adds an extra channel for the floormaps that shows the orientation of the receiver.", action='store_true')

    # Acumaps
    p.add_argument("--acumap_resolution", help="Resolution in pixels for the acuamps, usually 128. Should match floormaps", type=int)
    p.add_argument("--acumap_k", help="Kernel size for the acumaps", type=int)
    p.add_argument("--acumap_s", help="Stride for the acumaps", type=int)
    p.add_argument("--acumap_p", help="Padding for the acumaps", type=int)
    p.add_argument("--acumap_std", help="Standard deviation for the gaussian low pass filter of the acumaps", type=float)
    p.add_argument("--acumap_parameters", nargs='*', help="Acoustic parameters for the acumaps. Standard set is ['c50', 't30', 'drr'].", type=str)
    p.add_argument("--acumap_frequency_bands", nargs='*', help="Acoustic parameters for the acumaps. Standard set is [250, 1000, 4000].", type=int)
    p.add_argument("--acumap_distances", nargs='*', help="Distance functions for each acoustic parameter. Standard set is ['l1', 'mape', 'l1'].", type=str)

    # Normalizers
    p.add_argument("--normalizer_vmins", nargs='*', help="Vmin per channel for acuamps. Standard is [-20, -20, -20, 0.125, 0.125, 0.125, -10, -10, -10].", type=float)
    p.add_argument("--normalizer_vmaxs", nargs='*', help="Vmax per channel for acumaps. Standard is [20, 20, 20, 4.0, 4.0, 4.0, 10, 10, 10].", type=float)
    p.add_argument("--normalizer_logchannels", nargs='*', help="Indices for channels that need log. Standard is [3,4,5]. ", type=int)

    # Model
    p.add_argument("--dan_disc_activation", help="Activation function for the discrimantor. Only for adversarial training", type=str)
    p.add_argument("--dan_use_multi_disc", help="Enables one discrmininator per acoustic parameter. Only for adversarial training", action='store_true')
    p.add_argument("--dan_conditional_disc", help="Enables conditional discriminator, using the full floormap. Only for adversarial training", action='store_true')
    p.add_argument("--net_use_threshold", help="Enables custom thresholding layer. Only for multislope predictions", action='store_true')
    p.add_argument("--net_stft_encoder_concat", help="Enables a separate bracnch as a STFT encoder, with concatenation of latents", action='store_true')
    p.add_argument("--net_use_dropout", help="Enables dropout for the Unet model", action='store_true')
    p.add_argument("--net_use_stft_input", help="Enables RIR stft input to the main Unet model", action='store_true')
    p.add_argument("--net_remove_src_rec", help="Removes src and rec from input features", action='store_true')
    p.add_argument("--net_use_pos_enc", help="Enables sinusoidal postional encodings for the positions of src and receivers.", action='store_true')
    p.add_argument("--net_use_film", help="Enables Film layers for the Unet model", action='store_true')
    p.add_argument("--net_use_multiple_rirs", help="Enables multiple rirs as input to the network", action='store_true')
    p.add_argument("--net_block", help="Block type to use in the Unet model", type=str, choices=['resnet', 'dense'])
    p.add_argument("--net_unet_model", help="Unet model to use. This is still a bit experimental", type=str, choices=['basic', 'bottleneck'])

    # Augmentation
    p.add_argument('--use_augmentation_getitem', action='store_true', help='Enables augmentation for training, in the get item.')
    p.add_argument('--use_floormap_augmentation', action='store_true', help='Enables floormap augmentation for training, this is basically pixel displacement.')
    p.add_argument("--aug_kernel_size", nargs='*', help="Kernel size for floormap augmentaiton.", type=int)
    p.add_argument("--aug_kernel_num", help="Number of random kernels", type=int)
    
    params = p.parse_args()
    params = vars(params)  # Cast to Dictionary

    # For array jobs
    if params['task_id'] is not None and params['task_id'] != '':
        params['job_id'] = f"{params['job_id']}_{params['task_id']}"
    
    # Fix empty normalizeR_logchannels:
    if params['normalizer_logchannels'] == [-1]:
        params['normalizer_logchannels'] = []
    
    if False:
        if 'debug' in params['exp_name']:
            params['experiment_description'] = f'{params["exp_name"]}'
        else:
            params['experiment_description'] = f'{params["exp_group"]}-{params["exp_name"]}-{params["job_id"]}_{params["task_id"]}__' \
                                            f'n_work:{params["num_workers"]}_' 

        params['logging_dir'] = f'{params["logging_dir"]}/{params["experiment_description"]}'
        params['directory_output_results'] = f'{params["logging_dir"]}/tmp_results'

        # During evaluation, we dump to a subdirectory to avoid replacing previously created files
        if params['mode'] in ['valid']:
            params['experiment_description'] = f"{params['model_checkpoint_run']}/evaluation_{params['dataset_root'].split('/')[-1]}"
            params['logging_dir'] = f'{params["model_checkpoint_root"]}/{params["experiment_description"]}'
            params['directory_output_results'] = f'{params["model_checkpoint_root"]}/tmp_results'

        # Save config to disk, create directories if needed
        if 'debug' in params['logging_dir'] and os.path.exists(params['logging_dir']):
            shutil.rmtree(params['logging_dir'])
        if not os.path.exists(params['logging_dir']):
            os.makedirs(params['logging_dir'])
        with open(os.path.join(params['logging_dir'], 'params.yaml'), 'w') as f:
            yaml.dump(params, f, default_flow_style=None)
        if not os.path.exists(params['directory_output_results']):
            os.makedirs(params['directory_output_results'])

    print("")
    print("================ Experiment ================")
    print(params['exp_name'])
    print("")

    # Print the experiment config
    ctr = 0
    for k, v in params.items():
        ctr += 1
        if ctr % 10 == 0: print(' ')
        print('{} \t {}'.format(k.ljust(15, ' '), v))
    print("")

    return params

def save_config_to_disk(params):
    # Save config to disk, create directories if needed
    if not os.path.exists(params['logdir']):
        os.makedirs(params['logdir'])
    with open(os.path.join(params['logdir'], 'params.yaml'), 'w') as f:
        print(f"dumping params....  {params['logdir']}")
        yaml.dump(params, f, default_flow_style=None)

def setup_wandb(params: Dict):
    """ Prepares the wandb environment."""
    if params['wandb']:
        import wandb

        if not (params['do_precompute_rirs_lmdb'] or params['do_precompute_maps_lmdb'] or params['do_validation']):
            wandb_config = {
                "job_id": params['job_id'],
                "num_workers": params['num_workers'],
                "loss_function": params["loss_function"],
                "dataset": params["dataset"],
                "fold": params["fold"],
                "config": params["my_config"],
                "seed": params['seed'],
            }
            wandb.init(project='nvas_2dparams',
                    name=params['run'],
                    tags=['debug' if 'deubg' in params['run'] else 'exp',
                            str(params['fmap_xlim']),
                            'baseline' if params['do_baseline'] else 'model',
                            'mono' if len(params['rir_output_channels']) == 1 else 'spatial',
                            params['dataset'],
                            params['fold'],
                            params['loss_function']],
                    group=params['exp_group'] if (params['exp_group'] is not None or params['exp_group'] != '') else None,
                    config=wandb_config,
                    dir=params["logdir"],
                    sync_tensorboard=True)
            wandb.tensorboard.patch(save=False)
        return True
    return False

# Test
if __name__ == '__main__':
    # Run unit test like this:
    # >> python -m parameters -c configs/train_default.yaml
    config = get_parameters()
    print(config)

