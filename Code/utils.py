# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import io
import pickle
import random
import warnings
import datetime
from typing import Iterable, Tuple, TypeVar, Callable, Any, List, Union
from matplotlib.colors import to_rgb
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import spaudiopy as spa

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

    Ref: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            # warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]  # wrong code
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                # self.after_scheduler.step(metrics, None)  # not necessary epoch parameter
                self.after_scheduler.step(metrics)
            else:
                # self.after_scheduler.step(metrics, epoch - self.total_epoch)  # not necessary epoch parameter
                self.after_scheduler.step(metrics)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class CudaToCpuUnpickler(pickle.Unpickler):
    """ This is neede to deserialize objects that have torch tensors as properties.
    Mostly for the processors, that we create in gpu, but then we might want to deserialized them in cpu.
    
    Example:
    with open(os.path.join(this_directory_lmdb_maps, 'floormap_processor.pkl'), 'rb') as handle:
        floormap_processor = CudaToCpuUnpickler(handle).load()

    NOTE: This is wrong. Do not use.
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return self.load_from_bytes
        return super().find_class(module, name)
            
    def load_from_bytes(self, b):
        loaded_tensor = torch.load(io.BytesIO(b))
        if loaded_tensor.is_cuda:
            return loaded_tensor.cpu()
        else:
            return loaded_tensor


def is_multislope_tag(tag: str):
    """ Returns whether or not this tag is part of th multilope model"""
    multislope_tags = ['t0', 't1', 't2', 'a0', 'a1', 'a2', 'n_slopes']
    return tag in multislope_tags

def validate_audio(audio, message=''):
    """ Checks if the audio is valid by:
        -- Not having any infinite values
        -- Not having NaNs
        -- Not being silence (only 0s)
        -- All values in the range [-1, 1]

        -- Max magnitude = 50
        -- 
    """
    if isinstance(audio, np.ndarray):
        tmp = torch.from_numpy(audio)
    else:
        tmp = audio

    tolerance = 1e-10  # Some rirs have very low gain, so we use a small tolerance
    max_magnitude = 50
    check = torch.all(torch.isfinite(tmp))
    check = check and torch.all(~torch.isnan(tmp))
    check = check and torch.any(torch.logical_not(torch.isclose(tmp, torch.zeros_like(tmp), atol=tolerance)))
    #check = check and torch.all(torch.logical_and(tmp <= 1, tmp >= -1))  # This is mostly only needed when writing wav files
    check = check and tmp.abs().max() < max_magnitude

    #import matplotlib
    #matplotlib.use('TkAgg')
    #plot_waveform(tmp.transpose(1, 0), sample_rate=24000)

    if message != '':
        print('>>>>>>>> ' + message + f'{check}')
    return check

def crop_or_pad(input_sig, target_length):
    """ Crops or pads the input signal so that it has the target lemght """
    if target_length > 0:
        # Crop if needed
        output_size = target_length
        input_size = input_sig.shape[-1]  # Assuming last axis is time
        if input_size < output_size:
            padding = output_size - input_size       
            value = 0
            padder = torch.nn.ConstantPad1d((0, padding), value)  # padding to the right
            trimmed_sig = padder(input_sig)
        else:
            trimmed_sig = torch.narrow(input_sig, dim=-1, start=0, length=output_size)
    else:
        trimmed_sig = input_sig
    return trimmed_sig


def seed_everything(seed=12345, mode='balanced'):
    # ULTIMATE random seeding for either full reproducibility or a balanced reproducibility/performance.
    # In general, some operations in cuda are non deterministic to make them faster, but this can leave to
    # small differences in several runs.
    #
    # So as of 21.10.2021, I think that the best way is to use the balanced approach during exploration
    # and research, and then use the full reproducibility to get the final results (and possibly share code)
    #
    # References:
    # https://pytorch.org/docs/stable/notes/randomness.html
    #
    # Args:
    #   -- seed = Random seed
    #   -- mode {'balanced', 'deterministic'}

    print(f'Setting random seed {seed} with mode {mode}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode == 'balanced':
        torch.backends.cudnn.deterministic = False   # if set as true, dilated convs are really slow
        torch.backends.cudnn.benchmark = True  # True -> better performance, # False -> reproducibility
    elif mode == 'deterministic':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Throws error:
        # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)`
        # or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS
        # and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable
        # before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
        # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

        # torch.use_deterministic_algorithms(True)

def vecs2dirs(vecs, positive_azi=True, include_r=False, use_elevation=False):
    """Helper to convert [x, y, z] to [azi, colat].
    From Spaudiopyy, but with safe case when r=0"""
    azi, colat, r = spa.utils.cart2sph(vecs[:, 0], vecs[:, 1], vecs[:, 2], steady_colat=True)
    if positive_azi:
        azi = azi % (2 * np.pi)  # [-pi, pi] -> [0, 2pi)
    if use_elevation:
        colat = colat2ele(colat)
    if include_r:
        output = np.c_[azi, colat, r]
    else:
        output = np.c_[azi, colat]
    return output

def colat2ele(colat: Union[float, torch.Tensor]) -> torch.Tensor:
    """Transforms colatitude to elevation (latitude). In radians.

    The polar angle on a Sphere measured from the North Pole instead of the equator.
    The angle $\phi$ in Spherical Coordinates is the Colatitude.
    It is related to the Latitude $\delta$ by $\phi=90^\circ-\delta$.
    """
    ele = math.pi/2 - colat
    return ele


def ele2colat(ele: Union[float, torch.Tensor]) -> torch.Tensor:
    """Transforms colatitude to elevation (latitude). In radians.

    The polar angle on a Sphere measured from the North Pole instead of the equator.
    The angle $\phi$ in Spherical Coordinates is the Colatitude.
    It is related to the Latitude $\delta$ by $\phi=90^\circ-\delta$.
    """
    colat = math.pi/2 - ele
    return colat

def get_grid(degree: int):
    """Returns the cartesian coordinates for a t_design.
    This represents a grid of directions, that uniformly samples the unit sphere."""
    t_design = spa.grids.load_t_design(degree=degree)
    return t_design
    
def get_rotation_matrix(rotation_phi, rotation_theta, rotation_psi, device='cpu') -> torch.Tensor:
    """ Returns a full 3d Rotation matrix, for the given rotation angles phi, thetha, psi
    that are rotations over:
    phi --> rotation over x axis (roll)
    theta --> rotation over y axis (pitch)
    psi --> rotation over z axis (yaw)
    
    Note: This rotates cartesian coordinates.
    Rotation angles are in radians.
    
    Reference:
    See: [1] M. Kronlachner, 'Spatial transformations for the alteration of ambisonic recordings.'
    Equation 2.15
    """
    roll = torch.tensor([[1, 0, 0],
                         [0, np.cos(rotation_phi), -np.sin(rotation_phi)],
                         [0, np.sin(rotation_phi), np.cos(rotation_phi)]], device=device).to(torch.float64)
    pitch = torch.tensor([[np.cos(rotation_theta), 0, np.sin(rotation_theta)],
                          [0, 1, 0],
                          [-np.sin(rotation_theta), 0, np.cos(rotation_theta)]], device=device).to(torch.float64)
    yaw = torch.tensor([[np.cos(rotation_psi), -np.sin(rotation_psi), 0],
                        [np.sin(rotation_psi), np.cos(rotation_psi), 0],
                        [0, 0, 1]], device=device).to(torch.float64)
    R = torch.matmul(torch.matmul(roll, pitch), yaw)
    return R

def compute_Y_and_W(grid: torch.Tensor, rotation_matrix: torch.Tensor = None, order_input: int = 1, order_output: int = 1, backend='basic', w_pattern='hypercardioid') -> torch.Tensor:
    """ Computes the reconstruction matrix Y, and beamforming matrix W 
    This is for spherical harmonics manipulation.
    """
    # Directions for the grid in spherical coordinates. Discrete sampling of the sphere
    assert len(grid.shape) == 2 and grid.shape[-1] == 3, 'ERROR, the grid should be a list of cartesian points with shape [n, 3]'
    tmp_directions = vecs2dirs(grid)  
    
    if rotation_matrix is not None:
        tmp_directions_rotated = vecs2dirs(torch.matmul(torch.from_numpy(grid).float(), rotation_matrix.float()))
    else:
        tmp_directions_rotated = tmp_directions
        
    if backend == 'basic':
        Y = spa.sph.sh_matrix(order_input, tmp_directions[:, 0], tmp_directions[:, 1], SH_type='real', weights=None)
        W = spa.sph.sh_matrix(order_output, tmp_directions_rotated[:, 0], tmp_directions_rotated[:, 1], SH_type='real', weights=None)
    elif backend == 'spatial_filterbank':
        assert order_input == order_output, 'When using spatial filterbank, the input and output orders should be the same'

        # Weights for polar patterns
        if w_pattern.lower() == "cardioid":
            c_n = spa.sph.cardioid_modal_weights(order_output)
        elif w_pattern.lower() == "hypercardioid":
            c_n = spa.sph.hypercardioid_modal_weights(order_output)
        elif w_pattern.lower() == "maxre":
            c_n = spa.sph.maxre_modal_weights(order_output, True)  # works with amplitude compensation and without!
        else:
            raise ValueError(f'ERROR: Unknown w_pattern type: {w_pattern} . Check spelling? ')
        [W, Y] = spa.sph.design_spat_filterbank(order_output, tmp_directions[:, 0], tmp_directions[:, 1], c_n, 'real', 'perfect')
        if rotation_matrix is not None:
            [W, _] = spa.sph.design_spat_filterbank(order_output, tmp_directions_rotated[:, 0], tmp_directions_rotated[:, 1], c_n, 'real', 'perfect')
    else:
        raise ValueError(f'ERROR: Unknown backend : {backend} . Should be either "basic", or "spatial_filterbank"')

    Y = Y.astype(np.double)
    W = W.astype(np.double)
    W = torch.from_numpy(W)
    Y = torch.from_numpy(Y)

    return Y, W

class SphericalRotation(nn.Module):
    """ Class to do 3d rotations to signals in spherical harmonics domain 
    
    mode == 'single' --> Applies a single rotation by the specified rotation angles.
    
    moded == 'random' --> Precomputes num_random_rotations, so that it can be applied fast in runtime.  """
    
    def __init__(self, rotation_angles_rad: Tuple[float, float, float] = [0.0, 0.0, 0.0],
                 mode = 'single', num_random_rotations: int = -1, device: str = 'cpu', 
                 t_design_degree: int = 3, order_input: int = 1 , order_output: int = 1,
                 backend='basic', w_pattern='hypercardioid'):
        super(SphericalRotation, self).__init__()
        
        assert t_design_degree > 2 * order_output, 'The t-design degree should be > 2 * N_{tilde} of the output order '

        
        self.rotation_angles_rad = rotation_angles_rad
        self.grid = get_grid(degree=t_design_degree)
        self.n_directions = self.grid.shape[0]
        self.order_input = order_input
        self.order_output = order_output
        self.mode = mode
        self.num_random_rotations = num_random_rotations
        self.device = device
        self.backend = backend
        self.w_pattern = w_pattern

        if mode == 'single':
            self.R = utils.get_rotation_matrix(*self.rotation_angles_rad)
            self.Y = None
            self.W = None
            self.T_mat = None
        
            self.rotation_angles_rad = rotation_angles_rad
            Y, W = compute_Y_and_W(self.grid, self.R, self.order_input, self.order_output, 
                                   backend=self.backend, w_pattern=self.w_pattern)
            self.Y = Y.to(self.device)
            self.W = W.to(self.device)
            T_mat = torch.matmul(self.Y.transpose(1,0), self.W)
        
            if self.backend == 'basic':
                scale = 4 * np.pi / self.n_directions  # TODO August 05 2022, this works ok , except for input_order > 1
                T_mat = scale * T_mat
            
            self.T_mat = T_mat.to(self.device)

        elif mode == 'random':
            # Precompute a bunch of transformation matrices
            self.R, self.Y, self.W, self.T_mat = [], [], [], []
            for i in range(self.num_random_rotations):
                this_R = get_rotation_matrix(*get_random_rotation_angles(False, False, True))
                
                self.rotation_angles_rad = rotation_angles_rad
                this_Y, this_W = compute_Y_and_W(self.grid, this_R, self.order_input, self.order_output, backend=self.backend, w_pattern=self.w_pattern)
                this_Y = this_Y.to(self.device)
                this_W = this_W.to(self.device)
                T_mat = torch.matmul(this_Y.transpose(1,0), this_W).to(self.device)
                
                self.R.append(this_R)
                self.Y.append(this_Y)
                self.W.append(this_W)
                self.T_mat.append(T_mat)
            
    def forward(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double()
        if X.shape[-2] > X.shape[-1]:  # Channels first format
            warnings.warn('WARNING: It seems that the input tensor X is NOT in channels-first format')
        if self.mode == 'single':
            assert X.shape[-2] == self.W.shape[-1], 'ERROR: The order of the input signal does not match the rotation matrix'
            
            if self.W is not None and self.Y is not None:
                assert X.shape[-2] == self.W.shape[-1], 'Wrong shape for input signal or matrix W.'
            assert self.T_mat.shape[-1] == X.shape[-2], 'Wrong shape for input signal or matrix T.'
    
            out = torch.matmul(self.T_mat, X)
        elif self.mode == 'random':
            id = np.random.randint(0, self.num_random_rotations, size=(1))[0]

            assert X.shape[-2] == self.W[id].shape[-1], f'ERROR: The order of the input signal does not match the rotation matrix. \n I found X: {X.shape}, W:{self.W[id].shape}'
            out = torch.matmul(self.T_mat[id], X)
            
        return out
    
    def __repr__(self):
        rep = "SphericalRotation with: \n"
        rep += f'Mode = {self.mode} \n'
        rep += f'num_random_rotations = {self.num_random_rotations} \n'
        rep += f'Device = {self.device} \n'
        rep += f'order_input = {self.order_input} \n'
        rep += f'order_output = {self.order_output} \n'
        rep += f'backend = {self.backend} \n'
        rep += f'w_pattern = {self.w_pattern} \n'
        rep += f'n_directions = {self.n_directions} \n'
        rep += f'Rotation_angles = {self.rotation_angles_rad} \n'

        return rep

def get_random_rotation_angles(rotate_phi=False, rotate_theta=False, rotate_psi=True):
    """ Returns a random rotation matrix """
    if rotate_phi:
        phi = np.random.rand(1) * 2*np.pi
    if rotate_theta:
        theta = np.random.rand(1) * 2*np.pi - np.pi
    if rotate_psi:
        psi = np.random.rand(1) * 2*np.pi

    rotation_angles = [rotate_phi, rotate_theta, rotate_psi]
    return rotation_angles
    
class Timer(object):
    """
    Timer class to keep track of time per step and epoch during training.
    """

    def __init__(self):
        self.elapsed_per_counter = {
            'step': [],
            'epoch': [],
            'valid': []}
        self._tic = {
            'step': None,
            'epoch': None,
            'valid': None}

    def start(self, message="Timer start"):
        """Starts the timer"""
        self.start = datetime.datetime.now()
        return message + f'{self.start:.4f}'

    def stop(self, message="Total: "):
        """Stops the timer.  Returns the time elapsed"""
        self.stop = datetime.datetime.now()
        return message + f'{(self.stop - self.start):.4f}'

    def now(self):
        return datetime.datetime.now()

    def toc(self, counter='step'):
        tmp = datetime.datetime.now() - self._tic[counter]
        val = tmp.total_seconds()
        self.elapsed_per_counter[counter].append(val)
        return val

    def tic(self, counter='step'):
        self._tic[counter] = datetime.datetime.now()

    def plot_timers(self, filepath='./results_debug'):
        fig, axes = plt.subplots(3, 1)
        for ii, counter in enumerate(['step', 'epoch', 'valid']):
            axes[ii].plot(self.elapsed_per_counter[counter])
            axes[ii].grid(True)
            axes[ii].set_ylabel(f'Elapse for f{counter}')

        if filepath is not None:
            fname = 'Timer.png'
            plt.savefig(os.path.join(filepath, fname))
        plt.show()

        return fig

    def report(self):
        tmp_step = np.asarray(self.elapsed_per_counter["step"])
        tmp_epoch = np.asarray(self.elapsed_per_counter["epoch"])
        tmp_valid = np.asarray(self.elapsed_per_counter["valid"])
        rep = f'First elapsed per step = {tmp_step[0]} \n'
        rep += f'First elapsed per epoch = {tmp_epoch[0]} \n'
        rep += f'First elapsed per valid = {tmp_valid[0]} \n'
        rep += f'Mean elapsed per step = {np.mean(tmp_step)} \n'
        rep += f'Mean elapsed per epoch = {np.mean(tmp_epoch)} \n'
        rep += f'Mean elapsed per valid = {np.mean(tmp_valid)} \n'
        return rep


if __name__ == '__main__':
    test_beta_distributions()
    print('Finished test')


