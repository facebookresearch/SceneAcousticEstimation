# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import math
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms
import torchvision
import torch.optim as optim
import torch.profiler
import matplotlib.pyplot as plt

from datetime import timedelta
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List

from models.map_models import UnetBasic, Encoder, Decoder
from datasets.soundspaces_dataset import SoundspacesDataset
import features
import utils
import pytorch_acoustics


def set_requires_grad(nets: List, requires_grad=False):
    """Set requies_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not

    Adapted from:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/361f8b00d671b66db752e66493b630be8bc7d67b/models/base_model.py#L219
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class SolverBasic(object):
    def __init__(self, config, tensorboard_writer=None, model_checkpoint=None):

        # Data and configuration parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device set to = {self.device}')
        self.config = config
        self.writer = tensorboard_writer
        self.model_checkpoint = model_checkpoint
        self.data = {
            's_0': None,
            's_1': None,
            's_hat_0': None,
            's_hat_1': None,
            'mics': None,
            'sources': None}

 
        self._fixed_data_id = 0  # Id for the fixed input used to monitor the performance of the generator
        self._fixed_data = None
        self._fixed_data_counter = 0

        # If using multiple losses, each loss has a name, value, and function (criterion)
        # We have losses that we optimize, and other losses that we only monitor
        self.loss_fns, self.monitor_loss_fns, self.monitor_cSTFT_fns = self._get_loss_fn()
 
        # Build models
        self.net = self.build_predictor()
        self.init_optimizers()

        if self.config['model_input_dtype'] == 'complex64':
            dtype = torch.complex64
        elif self.config['model_input_dtype'] == 'float32':
            dtype = torch.float32
        print(f'Input predictor = {self.config.model_input_shape}')
        summary(self.net, input_size=tuple([1, *self.config.model_input_shape]),
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=10, dtypes=[dtype])

        if self.model_checkpoint is not None:
            self.load(checkpoint_path=model_checkpoint)

    def save(self, monitor_path=None, iteration=None):
        self.model_checkpoint = f'{monitor_path}/net_params_{iteration:07}.pth'
        self.model_checkpoint = f'{monitor_path}/net_params_best.pth'

        checkpoint = {'net_state_dict': self.net.state_dict(),
                      'optimizer_state_dict': self.optimizer_predictor.state_dict(),
                      'scheduler_state_dict': self.lr_scheduler.state_dict(),
                      'rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state()}
        torch.save(checkpoint, self.model_checkpoint)
        print('Checkpoint saved to {}.'.format(self.model_checkpoint))

    def load(self, checkpoint_path):
        self.model_checkpoint = checkpoint_path

        print(f"Loading model state from: \n{self.model_checkpoint}")
        checkpoint = torch.load(self.model_checkpoint, map_location=self.device)

        self.net = self.build_predictor()
        # Use to restart training or do inference only
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer_predictor.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Checkpoint was loaded from to {}.'.format(self.model_checkpoint))

    def build_predictor(self):
        if self.config['model'] == 'Unet_Basic':
            model = UnetBasic(input_channels=self.config.model_input_shape[-3], 
                              output_channels=elf.config.model_output_shape[-3], 
                              debug=False)
        return model.to(self.device)

    def _get_loss_fn(self) -> Tuple[Dict, Dict, Dict]:
        # Main loss function (training objective)
        if self.config.loss_fn == 'mse':
            loss_fn = nn.MS
            loss_fns = {'loss': loss_fn}
        

        return loss_fns, loss_monitor, loss_monitor_cSTFT

    def init_optimizers(self):
        # Optimizer
        if self.config['optimizer'] == 'ranger':
            self.optimizer_predictor = Ranger(self.net.parameters(), lr=self.config['lr'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer_predictor = optim.Adam(self.net.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))

        # Scheduler
        if self.config['lr_scheduler'] == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_predictor,
                                                                self.config['lr_scheduler_step'],
                                                                gamma=0.9,
                                                                last_epoch=- 1,
                                                                verbose=False)
        elif self.config['lr_scheduler'] == 'plateau':   # TODO this scheduler is not tested
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_predictor,
                factor=self.config.lr_decay_rate,
                patience=self.config.lr_patience_times,
                min_lr=self.config.lr_min
            )
            self.lr_scheduler = GradualWarmupScheduler(
                self.optimizer_predictor,
                multiplier=1, total_epoch=5, after_scheduler=lr_scheduler)

    def set_input(self, s_0: torch.Tensor, s_1: torch.Tensor, y: torch.Tensor,
                  mics: torch.Tensor, sources: torch.Tensor,
                  ref0: torch.Tensor = None, ref1: torch.Tensor = None):
        """ Sets the local input data from the minibatch. I expect:
         s_0: clean source 0
         s_1: clean source 1
         y: mixed input (so sum of the clean sources, with additional encoding features)
         mics: position for the microphones
         sources: positions for the sources
         ref0: anechoic source 0, single channel
         ref1: anechoic source 1, single channel"""
        self.data['s_0'] = s_0
        self.data['s_1'] = s_1
        self.data['y'] = y
        self.data['mics'] = mics
        self.data['sources'] = sources
        self.data['ref0'] = ref0
        self.data['ref1'] = ref1
#        self.data['n_mics'] = mics.shape[-1]

        if self.config['loss_fn'] == 'weakSup':
            self.loss_fns['loss_sep'].set_n_mics(mics.shape[-1])

        self.monitor_loss_fns['Directional'].set_n_mics(mics.shape[-1])

    def set_input_fixed(self, s_0: torch.Tensor, s_1: torch.Tensor, y: torch.Tensor,
                        mics: torch.Tensor, sources: torch.Tensor,
                        ref0: torch.Tensor = None, ref1: torch.Tensor = None):
        # Assign a fixed input to monitor the task
        data = {}
        data['s_0'] = s_0
        data['s_1'] = s_1
        data['y'] = y
        data['mics'] = mics
        data['sources'] = sources
        data['ref0'] = ref0
        data['ref1'] = ref1

        if self._fixed_data is None:
            self._fixed_data = {key: value.detach().clone() if value is not None else value for (key, value) in data.items()}

    def lr_step(self, val_loss):
        """ step in iterations"""
        self.lr_scheduler.step()

    def get_lrs(self):
        return [self.optimizer_predictor.state_dict()['param_groups'][0]['lr']]

    def get_fixed_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._fixed_data['s_0'], self._fixed_data['s_1'], self._fixed_data['y'], self._fixed_data['mics'], self._fixed_data['sources']

    def get_fixed_output(self):
        if self._fixed_data is not None:
            out = self.net(self._fixed_data).detach().cpu()
            return out

    def get_grad_norm(self):
        """ Returns the gradient norms for predictor. """
        grad_norm_model = grad_norm(self.net.parameters())
        return [grad_norm_model]

    def forward(self):
        self.data['s_hat_0'], self.data['s_hat_1'] = self.net(self.data['y'])

    def forward_no_grad(self, iSTFT: torch.nn.Module, STFT: torch.nn.Module, validation=False):
        """ This runs a modified forward that computes the additional losses that we monitor.
        Here we do beamforming, spatial covariance, and stuff like that."""
        n_mics = self.data['mics'].shape[-1]
        if not validation:
            var_monitor_loss = self.monitor_loss_values
            var_monitor_cSTFT = self.monitor_cSTFT_values
        else:
            var_monitor_loss = self.valid_monitor_loss_values
            var_monitor_cSTFT = self.valid_monitor_cSTFT_values

        # Spatial Covariances
        with torch.no_grad():
            tmp = self.monitor_loss_fns['Spatial1D'](iSTFT(self.data['s_hat_0']), iSTFT(self.data['s_0'])) + \
                  self.monitor_loss_fns['Spatial1D'](iSTFT(self.data['s_hat_1']), iSTFT(self.data['s_1']))
            var_monitor_loss['Spatial1D'] = tmp.item()

            tmp = self.monitor_loss_fns['Spatial2D'](self.data['s_hat_0'], self.data['s_0']) + \
                  self.monitor_loss_fns['Spatial2D'](self.data['s_hat_1'], self.data['s_1'])
            var_monitor_loss['Spatial2D'] = tmp.item()

        # Spectral Sum
        with torch.no_grad():
            y_hat = iSTFT(self.data['s_hat_0']) + iSTFT(self.data['s_hat_1'])
            div = torch.maximum(y_hat.abs().max(), torch.maximum(iSTFT(self.data['s_hat_0']).abs().max(),
                                                                 iSTFT(self.data['s_hat_1']).abs().max()))
            y_hat /= div
            tmp = self.monitor_loss_fns['Spectral2d'](STFT(y_hat, ignore_encodings=True), self.data['y'][:, 0:n_mics, ...])
            var_monitor_loss['Spectral2d'] = tmp.item()

            y_hat = iSTFT(self.data['s_hat_0']) + iSTFT(self.data['s_hat_1'])
            tmp = self.monitor_loss_fns['Spectral1d'](y_hat, iSTFT(self.data['y'][:, 0:n_mics, ...]))
            var_monitor_loss['Spectral1d'] = tmp.item()

        # Directional Loss
        with torch.no_grad():
            self.monitor_loss_fns['Directional'].set_STFT(STFT, self.data['s_hat_0'].device)
            tmp = self.monitor_loss_fns['Directional'](self.data['s_hat_0'], self.data['s_hat_1'],
                                                       self.data['s_0'][:, 0:n_mics, ...], self.data['s_1'][:, 0:n_mics, ...],
                                                       self.data['mics'], self.data['sources'], iSTFT)
            var_monitor_loss['Directional'] = tmp.item()

        # Fixed beamforming
        with torch.no_grad():
            ss_ids = [0, 1]
            for ss in range(2):  # HARDCODED 2 sources
                ss_other = -(ss + 1)

                doas = self.data['sources'][..., ss]
                beam_y = self.beamformer(self.data['y'][:, 0:n_mics, ...], doas, self.data['mics'])
                beam_s_hat = self.beamformer(self.data[f's_hat_{ss}'], doas, self.data['mics'])

                doas_other = self.data['sources'][..., ss_other]
                beam_y_other = self.beamformer(self.data['y'][:, 0:n_mics, ...], doas_other, self.data['mics'])
                beam_s_hat_other = self.beamformer(self.data[f's_hat_{ss}'], doas_other, self.data['mics'])

                beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y, beam_s_hat, beam_y_other, beam_s_hat_other)
                ###beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y, beam_s_hat, beam_y_other, beam_s_hat)
                #########beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y, beam_s_hat_other, beam_y_other, beam_s_hat_other)
                #beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y_other, beam_s_hat)
                var_monitor_loss[f'Beam_s_{ss}'] = beamloss.item()

                # Loss breakdown
                beamloss = self.monitor_loss_fns[f'Beam_s{ss}_theta{ss}'](beam_s_hat, beam_y)
                var_monitor_loss[f'Beam_s{ss}_theta{ss}'] = beamloss.item()
                beamloss = self.monitor_loss_fns[f'Beam_s{ss}_theta{ss_ids[ss_other]}'](beam_s_hat, beam_y_other)
                var_monitor_loss[f'Beam_s{ss}_theta{ss_ids[ss_other]}'] = beamloss.item()

                var_monitor_loss[f'Beam_s_{ss}'] = var_monitor_loss[f'Beam_s{ss}_theta{ss}'] - var_monitor_loss[f'Beam_s{ss}_theta{ss_ids[ss_other]}']

                if False:
                    distance = 1  # TODO: REMOVE this. This is a super hack to test the beamformer

                    # Doas for delay and sum
                    doas = self.data['sources'][..., ss]
                    doas = doas / torch.linalg.vector_norm(doas)  # unitary direction, relative to the center of the array
                    doas = distance * doas * torch.tensor([-1, -1, 1], device=doas.device)  # reflect across XZ plane, because I think they are using other coordinate convention
                    doas = doas[:, None, :].repeat(1, self.data['s_0'].shape[-1], 1)  # [batch, time_steps, 3]
                    doas_other = self.data['sources'][..., ss_other]
                    doas_other = doas_other / torch.linalg.vector_norm(doas_other)  # unitary direction, relative to the center of the array
                    doas_other = distance * doas_other * torch.tensor([-1, -1, 1], device=doas_other.device)  # reflect across XZ plane, because I think they are using other coordinate convention
                    doas_other = doas_other[:, None, :].repeat(1, self.data['s_0'].shape[-1], 1)  # [batch, time_steps, 3]

                    # Beamform from the mixed inputs Y
                    Xs = self.data['y'][:, 0:n_mics, ...]  # Noisy observed signal, input to the nets but only audio channels
                    Xs = Xs.permute([0, 3, 2, 1])  # [batch, timesteps, freqs, channels]
                    Xs = torch.stack([Xs.real, Xs.imag], dim=-2)  # [batch, timesteps, freqs, 2, channels]
                    # TODO delaysum does not support batches for mics  # mics=self.data['mics'][0].permute([1, 0])
                    beam_y = self.delaysum(Xs, localization_tensor=doas, doa_mode=True,
                                           mics=self.data['mics'].permute([0,2,1]), fs=self.config['dataset_fs'])
                    beam_y_other = self.delaysum(Xs, localization_tensor=doas_other, doa_mode=True,
                                                 mics=self.data['mics'].permute([0,2,1]), fs=self.config['dataset_fs'])

                    # Beamform from the source separation multichannel outputs
                    Xs = self.data[f's_hat_{ss}'][:, 0:n_mics, ...]  # Noisy observed signal, input to the nets but only audio channels
                    Xs = Xs.permute([0, 3, 2, 1])  # [batch, timesteps, freqs, channels]
                    Xs = torch.stack([Xs.real, Xs.imag], dim=-2)  # [batch, timesteps, freqs, 2, channels]
                    # TODO delaysum does not support batches for mics
                    beam_s_hat = self.delaysum(Xs, localization_tensor=doas, doa_mode=True,
                                               mics=self.data['mics'].permute([0,2,1]), fs=self.config['dataset_fs'])
                    beam_s_hat_other = self.delaysum(Xs, localization_tensor=doas_other, doa_mode=True,
                                                     mics=self.data['mics'].permute([0,2,1]), fs=self.config['dataset_fs'])

                    # Back to our shape convention
                    beam_y = torch.complex(beam_y[..., 0, :], beam_y[..., 1, :]).permute([0, 3, 2, 1])  # [batch, channels, freqs, timesteps]
                    beam_y_other = torch.complex(beam_y_other[..., 0, :], beam_y_other[..., 1, :]).permute([0, 3, 2, 1])  # [batch, channels, freqs, timesteps]
                    beam_s_hat = torch.complex(beam_s_hat[..., 0, :], beam_s_hat[..., 1, :]).permute([0, 3, 2, 1])  # [batch, channels, freqs, timesteps]
                    beam_s_hat_other = torch.complex(beam_s_hat_other[..., 0, :], beam_s_hat_other[..., 1, :]).permute([0, 3, 2, 1])  # [batch, channels, freqs, timesteps]

                    beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y, beam_s_hat, beam_y_other, beam_s_hat_other)
                    ###beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y, beam_s_hat, beam_y_other, beam_s_hat)
                    #########beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y, beam_s_hat_other, beam_y_other, beam_s_hat_other)
                    #beamloss = self.monitor_loss_fns[f'Beam_s_{ss}'](beam_y_other, beam_s_hat)
                    var_monitor_loss[f'Beam_s_{ss}'] = beamloss.item()

                    # Loss breakdown
                    beamloss = self.monitor_loss_fns[f'Beam_s{ss}_theta{ss}'](beam_s_hat, beam_y)
                    var_monitor_loss[f'Beam_s{ss}_theta{ss}'] = beamloss.item()
                    beamloss = self.monitor_loss_fns[f'Beam_s{ss}_theta{ss_ids[ss_other]}'](beam_s_hat, beam_y_other)
                    var_monitor_loss[f'Beam_s{ss}_theta{ss_ids[ss_other]}'] = beamloss.item()

        # Complex STFT losses
        with torch.no_grad():
            for key in self.monitor_cSTFT_names:
                tmp = self.monitor_cSTFT_fns[key](self.data['s_hat_0'], self.data['s_0'][..., 0:n_mics, :, :]) + \
                      self.monitor_cSTFT_fns[key](self.data['s_hat_1'], self.data['s_1'][..., 0:n_mics, :, :])
                var_monitor_cSTFT[key] = tmp.item()

    def backward(self, iSTFT: nn.Module = None, STFT: nn.Module = None) -> None:
        """Calculate loss for the predictor"""
        n_mics = self.data['mics'].shape[-1]

        # Weakly supervised
        if self.config['loss_fn'] == 'weakSup':
            #y_hat = STFT(iSTFT(self.data['s_hat_0']) + iSTFT(self.data['s_hat_1']),
            #             localization_tensor_source=self.data['sources'], mics=self.data['mics'],
            #             ignore_encodings=False)
            y_hat = STFT(iSTFT(self.data['s_hat_0']) + iSTFT(self.data['s_hat_1']), ignore_encodings=True)
            tmp_recons = self.loss_fns['loss_recons'](y_hat, self.data['y'][..., 0:n_mics, :, :], iSTFT)

            self.loss_fns['loss_sep'].set_STFT(STFT, self.device)
            tmp_sep = self.loss_fns['loss_sep'](self.data['s_hat_0'], self.data['s_hat_1'],
                                                self.data['s_0'][:, 0:n_mics, ...], self.data['s_1'][:, 0:n_mics, ...],
                                                self.data['mics'], self.data['sources'], iSTFT)
            loss = tmp_recons.to(torch.float64) + tmp_sep
            loss = loss.to(torch.float32)

            self.loss_values['loss_recons'] = tmp_recons
            self.loss_values['loss_sep'] = tmp_sep

        else:  # Supervised case
            if self.config['loss_supervised_targets'] == 'reverb':
                loss = self.loss_fns['loss'](self.data['s_0'][..., 0:n_mics, :, :], self.data['s_hat_0']) + \
                       self.loss_fns['loss'](self.data['s_1'][..., 0:n_mics, :, :], self.data['s_hat_1'])
            elif self.config['loss_supervised_targets'] == 'anechoic':  # TODO: Hardcoded refernce mic = 5
                loss = self.loss_fns['loss'](self.data['ref0'], self.data['s_hat_0'][..., None, 5, :, :]) + \
                       self.loss_fns['loss'](self.data['ref1'], self.data['s_hat_1'][..., None, 5, :, :])
            else:
                raise ValueError(f"ERROR, target {self.config['loss_supervised_targets']} is not supported")

        self.loss_values['loss'] = loss
        total_loss = loss
        total_loss.backward()

    def train_step(self, iSTFT: nn.Module = None, STFT: nn.Module = None):
        """ Calculates losses, gradients, and updates the network parameters"""
        self.net.train()
        self.forward()
        # Update Predictor
        self.net.zero_grad()
        self.backward(iSTFT=iSTFT, STFT=STFT)
        self.optimizer_predictor.step()

    def evaluation_step(self, iSTFT: nn.Module = None, STFT: nn.Module = None) -> torch.Tensor:
        """ Calculates losses, but does not update the network parameters."""
        self.net.eval()
        with torch.no_grad():
            self.forward()
            n_mics = self.data['mics'].shape[-1]

            # Weakly supervised
            if self.config['loss_fn'] == 'weakSup':
                y_hat = STFT(iSTFT(self.data['s_hat_0']) + iSTFT(self.data['s_hat_1']), ignore_encodings=True)
                tmp_recons = self.loss_fns['loss_recons'](y_hat, self.data['y'][..., 0:n_mics, :, :], iSTFT)

                self.loss_fns['loss_sep'].set_STFT(STFT, self.device)
                tmp_sep = self.loss_fns['loss_sep'](self.data['s_hat_0'], self.data['s_hat_1'],
                                                    self.data['s_0'][:, 0:n_mics, ...],
                                                    self.data['s_1'][:, 0:n_mics, ...],
                                                    self.data['mics'], self.data['sources'], iSTFT)
                loss = tmp_recons + tmp_sep
                self.loss_values['loss_recons'] = tmp_recons
                self.loss_values['loss_sep'] = tmp_sep

            else:  # Supervised case
                if self.config['loss_supervised_targets'] == 'reverb':
                    loss = self.loss_fns['loss'](self.data['s_0'][..., 0:n_mics, :, :], self.data['s_hat_0']) + \
                           self.loss_fns['loss'](self.data['s_1'][..., 0:n_mics, :, :], self.data['s_hat_1'])
                elif self.config['loss_supervised_targets'] == 'anechoic':  # TODO: Hardcoded refernce mic = 5
                    loss = self.loss_fns['loss'](self.data['ref0'], self.data['s_hat_0'][..., None, 5, :, :]) + \
                           self.loss_fns['loss'](self.data['ref1'], self.data['s_hat_1'][..., None, 5, :, :])
                else:
                    raise ValueError(f"ERROR, target {self.config['loss_supervised_targets']} is not supported")

            self.valid_loss_values['loss'] = loss.item()
            total_loss = self.valid_loss_values['loss']
        return total_loss


