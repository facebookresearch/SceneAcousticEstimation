# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from scipy.signal import windows, check_COLA
from typing import Union, Tuple, List, Dict, Optional

scenes_matterport = ['pLe4wQe7qrG', 'gTV8FGcVJC9', 'YFuZgdQ5vWj', 'oLBMNvg9in8', 'ZMojNkEp431', 'r47D5H71a5s', 'wc2JMjhGNzB', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'r1Q1Z4BcV1o', 'Vt2qJdWjCF2', 'JeFG25nYj2p', 'gYvKGZ5eRqb', 'x8F5xyUWy9e', 'E9uDoFAP3SH', 'ULsKaCPVFJR', 's8pcmisQ38h', 'q9vSo1VnCiC', 'V2XKFyX4ASd', 'EDJbREhghzL', 'p5wJjkQkbXX', 'kEZ7cmS4wCh', 'b8cTxDM8gDG', 'Z6MFQCViBuw', 'VLzqgDo317F', 'D7N2EKCX4Sj', 'sKLMLpTHeUy', 'pRbA3pwrgk9', 'sT4fr6TAbpF', '2n8kARJN3HM', '17DRP5sb8fy', 'rPc6DW4iMge', 'XcA2TqTSSAj', 'rqfALeAoiTq', '29hnd4uzFmX', 'YmJkqBEsHnH', 'D7G3Y4RVNrH', 'i5noydFURQK', 'qoiz87JEwZ2', 'B6ByNegPMKs', '1pXnuDYAj8r', 'JF19kD82Mey', 'Uxmj2M2itWa', 'mJXqzFtmKg4', '5LpN3gDmAk7', 'JmbYfDe2QKZ', '7y3sRwLe3Va', 'jtcxE69GiFV', 'SN83YJsR3w2', 'EU6Fwq7SyZv', 'X7HyMhZNoso', 'VzqfbhrpDEA', 'HxpKQynjfin', 'WYY7iVyf5p8', 'Vvot9Ly1tCj', 'QUCTc6BB5sX', 'yqstnuAEVhm', 'aayBHfsNo7d', '2azQ1b91cZZ', 'ARNzJeq3xxb', 'zsNo4HB9uLZ', 'uNb9QFRL6hY', '82sE5b5pLXE', '5q7pvUzZiYa', '759xd9YjKW5', 'PX4nDJXEHrG', '1LXtFkjw3qL', '8194nk5LbLH', 'UwV83HsGsw3', 'S9hNv5qa7GM', 'e9zR4mvMWw7', 'jh4fc5c5qoQ', 'Pm6F8kyY3z2', '8WUmhLawc2A', '5ZKStnWn8Zo', 'VVfe2KiqLaN', 'PuKPg4mmafe', 'ac26ZMwG7aT', 'fzynW3qQPVF', 'ur6pFq6Qu1A', 'GdvgFV5R1Z5', 'TbHJrupSAjP', 'gxdoqLR6rwA', 'pa4otMbVnkk', 'vyrNrziPKCB', 'gxdoqLR6rwA']

# These are in absolute mesh coordinates, so same as the locations in the points.txt
fixed_slice_replica = {'apartment_0': -0.25}

# These are in absolute mesh coordinates, so same as the locations in the points.txt
fixed_slice_coords_matterport = {'pLe4wQe7qrG': 1.75,
                                 'YFuZgdQ5vWj': 2.5,
                                 'oLBMNvg9in8': 2.5,
                                 'cV4RVeZvu5T': 1.75,
                                 'x8F5xyUWy9e': 2.0,
                                 's8pcmisQ38h': -1.0,
                                 'V2XKFyX4ASd': 1.5,
                                 'EDJbREhghzL': 1.5,
                                 'pRbA3pwrgk9': 3.5,
                                 '17DRP5sb8fy': 1.5,
                                 'XcA2TqTSSAj': 1.25,  # or 1.5
                                 '29hnd4uzFmX': 1.5,
                                 'YmJkqBEsHnH': 2.0,
                                 'D7G3Y4RVNrH': 2.0,
                                 'i5noydFURQK': 2.0,
                                 'JF19kD82Mey': 2.5,
                                 'JmbYfDe2QKZ': 1.5,
                                 'EU6Fwq7SyZv': 2.5,
                                 'HxpKQynjfin': 1.0,
                                 'WYY7iVyf5p8': 1.5,
                                 'yqstnuAEVhm': 1.75,
                                 'aayBHfsNo7d': 2.0,
                                 '8194nk5LbLH': 1.5,
                                 'e9zR4mvMWw7': 1.0,
                                 'jh4fc5c5qoQ': 1.5,
                                 'Pm6F8kyY3z2': 1.5,
                                 'GdvgFV5R1Z5': 1.5,
                                 'TbHJrupSAjP': 2.0,
                                 }  

def LogMagSTFT(nfft=1024, hop_length=120):
    spectrogram_transform = nn.Sequential(
        torchaudio.transforms.Spectrogram(n_fft=nfft,
                                          hop_length=hop_length,
                                          power=2),
        torchaudio.transforms.AmplitudeToDB())

    return spectrogram_transform

class MinMaxScalerFixed(torch.nn.Module):
    ''' Normalizes the data to the range [-1, 1] using predefined, fixed ranges for each dimension.
    This appplies the normalizing to each channel indenpendently, and supports linear case (for data in dB),
    and log case (for reverberation time in seconds).

    Linear case:
    normalized =  2 * ((tensor - vmin) / (vmax - vmin)) - 1 
    denormalized = (vmin + (vmax - vmin) * (tensor + 1) / 2)

    Log case:
    normalized =  2 * ((log_tensor - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))) - 1 
    denormalized = (np.log10(vmin) + (np.log10(vmax) - np.log10(vmin)) * (log_tensor + 1) / 2)
    '''
    def __init__(self, vmin_per_channel: Union[List[float], torch.Tensor], vmax_per_channel: Union[List[float], torch.Tensor], log_channels: List[int] = None):
        super(MinMaxScalerFixed, self).__init__()
        if not isinstance(vmin_per_channel, torch.Tensor):
            self.vmins = torch.tensor(vmin_per_channel)
        if not isinstance(vmax_per_channel, torch.Tensor):
            self.vmaxes = torch.tensor(vmax_per_channel)

        self.log_channels = log_channels
        if self.log_channels is not None:
            self.vmins[self.log_channels] = np.log10(self.vmins[self.log_channels])
            self.vmaxes[self.log_channels] = np.log10(self.vmaxes[self.log_channels])

    def forward(self, tensor):
        vmins = self.vmins.view(-1, 1, 1).to(tensor.device)
        vmaxes = self.vmaxes.view(-1, 1, 1).to(tensor.device)

        if self.log_channels is not None:
            tensor[..., self.log_channels, :, :] = torch.log10(tensor[..., self.log_channels, :, :] + 1e-10)
        normalized = 2 * ((tensor - vmins) / (vmaxes - vmins)) - 1 

        # Check: at least 1 not NaN value, and no infinites
        assert torch.any(~torch.isnan(normalized)) and torch.all(~torch.isinf(normalized)), "ERROR The normalized tensor contains infinite values or is all NaNs, (possible dividsion by 0)"
        return normalized

class MinMaxDeScalerFixed(torch.nn.Module):
    ''' Normalizes the data to the range [-1, 1] using predefined, fixed ranges for each dimension.
    This appplies the normalizing to each channel indenpendently, and supports linear case (for data in dB),
    and log case (for reverberation time in seconds).

    Linear case:
    normalized =  2 * ((tensor - vmin) / (vmax - vmin)) - 1 
    denormalized = (vmin + (vmax - vmin) * (tensor + 1) / 2)

    Log case:
    normalized =  2 * ((log_tensor - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))) - 1 
    denormalized = (np.log10(vmin) + (np.log10(vmax) - np.log10(vmin)) * (log_tensor + 1) / 2)
    '''
    def __init__(self, vmin_per_channel: List[float], vmax_per_channel: List[float], log_channels: List[int] = None):
        super(MinMaxDeScalerFixed, self).__init__()
        if not isinstance(vmin_per_channel, torch.Tensor):
            self.vmins = torch.tensor(vmin_per_channel)
        if not isinstance(vmax_per_channel, torch.Tensor):
            self.vmaxes = torch.tensor(vmax_per_channel)

        self.log_channels = log_channels
        if self.log_channels is not None:
            self.vmins[self.log_channels] = np.log10(self.vmins[self.log_channels])
            self.vmaxes[self.log_channels] = np.log10(self.vmaxes[self.log_channels])

    def forward(self, normalized):
        vmins = self.vmins.view(-1, 1, 1).to(normalized.device)
        vmaxes = self.vmaxes.view(-1, 1, 1).to(normalized.device)
        
        denormalized = (vmins + (vmaxes - vmins) * (normalized + 1) / 2)
        if self.log_channels is not None:
            denormalized[..., self.log_channels, :, :] = 10**denormalized[..., self.log_channels, :, :]

        #assert torch.all(~torch.isnan(denormalized) | torch.isfinite(denormalized)), "ERROR The normalized tensor contains infinite values or is all NaNs,, (possible dividsion by 0)"
        # Check: at least 1 not NaN value, and no infinites
        assert torch.any(~torch.isnan(denormalized)) and torch.all(~torch.isinf(denormalized)), "ERROR The normalized tensor contains infinite values or is all NaNs, (possible dividsion by 0)"
        return denormalized


class Feature_Stft(nn.Module):
    def __init__(self, nfft=1024, hop=240, pad=0, normalize_specs=False, log=True, window='hann',
                 n_mels=128, use_complex=False, fs=None):
        """
        Basic STFT features.

        Arguments:
        normalize_specs :
            Normalizes the spec to [-1, 1] range. Only for log scale. This is useful when using networks,
            but not for visualization.
        """
        super(Feature_Stft, self).__init__()

        #assert (use_complex != log) or (not use_complex), "Log scale not support when using complex values"

        self.nfft = nfft
        self.hop = hop
        self.log = log
        self.pad = pad
        self.use_complex = use_complex
        self.normalize_specs = normalize_specs
        self.fs = fs
        self.window = window

        self.eps = 1e-10
        self.clamp_min = -80

        # Check that the STFT parameters comply with COLA so that we can do perfect reconstruction
        # NOTE: We use double precision window due to issues with torch.
        # But it works ok casting back to single precision.
        # See:
        # https://github.com/pytorch/audio/issues/452
        if window == 'hann':
            window_tmp = torch.hann_window(nfft, dtype=torch.float64)
            window_fn = torch.hann_window
        elif window == 'sqrt_hann':
            window_tmp = torch.sqrt(torch.hann_window(nfft, dtype=torch.float64))
            raise AttributeError(f'ERROR, unsupported window type {window}')
        else:
            raise AttributeError(f'ERROR, unsupported window type {window}')
        cola = check_COLA(window=window_tmp.numpy(), nperseg=nfft, noverlap=nfft-hop)
        self.is_COLA = cola

        if not cola:
            warnings.warn('WARNING, the STFT features are not COLA, so perfect reconstruction will fail.')

        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft,
                                                      hop_length=hop,
                                                      power=None,
                                                      pad=pad, window_fn=window_fn)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=self.fs, f_min=0.0, f_max=None, n_stft=nfft // 2 + 1, norm=None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # STFT
        output = self.stft(input)

        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            warnings.warn('WARNING: NaNs or INFs right after STFT.')
            tb = traceback.format_exc()
            print(tb)

        n_channels = output.shape[-3]
        if self.use_complex:  # Complex spectrograms
            if self.log:
                #raise NotImplementedError
                div_mag = torch.amax(output.abs(), dim=(-3, -2, -1), keepdim=True)
                mag = 20 * torch.log10((output.abs() + self.eps) / div_mag)
                mag = torch.clamp(mag, self.clamp_min)  # [-80, 0] range, in dB
                output = torch.complex(mag, output.imag)

        else:  # Real valued spectrograms
            output = output.abs()

            if self.log:
                div = torch.amax(output, dim=(-3, -2, -1), keepdim=True)  # Single max across all channels, for each sample in the batch
                mag = 20 * torch.log10((output + self.eps) / div)
                mag = torch.clamp(mag, self.clamp_min)  # [-80, 0] range, in dB
                if self.normalize_specs:
                    t = torch.tensor(self.clamp_min)
                    mag = mag / t.abs()  # [-1, 0] range
                    mag = mag * 2 + 1  # [-1, 1] range
                output = mag

        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            warnings.warn('WARNING: NaNs or INFs when computing features.')
        return output

class Feature_MelPlusPhase(nn.Sequential):
    """ Mel Spectoram with interchannel phase difference features"""
    def __init__(self, normalize_specs=True, n_mels=86, nfft=1024, hop=240, use_phase_diff=False):
        super(Feature_MelPlusPhase, self).__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft,
                                                      hop_length=hop,
                                                      power=None)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=24000, f_min=0.0, f_max=None, n_stft=nfft // 2 + 1, norm=None)
        self.eps = 1e-10
        self.clamp_min = -80
        self.normalize_specs = normalize_specs
        self.use_phase_diff = use_phase_diff

    def forward(self, input):
        tmp = self.stft(input)
        mag = tmp.abs()
        mag = self.mel_scale(mag)
        div = torch.amax(mag, dim=(-3, -2, -1), keepdim=True)  # Singla max across all channels, for each sample in the batch
        mag = 20 * torch.log10((mag + self.eps) / div)
        mag = torch.clamp(mag, self.clamp_min)  # [-80, 0] range, in dB
        if self.normalize_specs:
            t = torch.tensor(self.clamp_min)
            mag = mag / t.abs()  # [-1, 0] range
            mag = mag * 2 + 1  # [-1, 1] range

        if self.use_phase_diff:
            phase = tmp.angle()
            phase = self.mel_scale(phase)
            phase_diff = torch.cos(torch.diff(phase, dim=-3))
            output = torch.concat([mag, phase_diff], dim=-3)
        else:
            output = mag
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            warnings.warn('WARNING: NaNs or INFs when computing features.')
        return output

class MaskedAvgPool(torch.nn.Module):
    """
        A class used to apply average pooling only on masked 2D inputs, with scaling correction.
        
    Attributes:
        kernel_size (int): The size of the window over which to take the average.
        stride (int, optional): The stride of the window. Default is None which means `kernel_size`.
        padding (int, optional): Implicit zero padding to be added on both sides. Default is 0.

    Example:
        >>> # Create a MaskedAvgPool layer with kernel size 3x3 and stride 2x2
        >>> masked_avg_pool = MaskedAvgPool(kernel_size=3, stride=2)
        
        >>> # Apply the layer to an input tensor and a mask tensor
        >>> output = masked_avg_pool(input_tensor, mask_tensor)
    
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Count the number of elements to keep in each window
        count = F.conv2d(mask.float(), 
                         weight=torch.ones((1, 1, self.kernel_size, self.kernel_size), device=x.device), 
                         stride=self.stride, 
                         padding=self.padding)

        # Mask and pool
        x_masked = x * mask.float()
        output = F.avg_pool2d(x_masked, self.kernel_size, self.stride, self.padding)
        output = output * (self.kernel_size * self.kernel_size) / count.clamp(min=1)      # Correct the output by the count of kept elements)
        
        return output
        
class FloorMapProcessor(object):
    """
    A class used to extract floor maps from meshes.
    
    Attributes:
        resolution (int): The resolution of the floor map.
        slice_coord (list): The height coordinates for the 2d slice.
        height_selection (list): The method used to select the height of the slices.
        xlim (tuple): The x-axis limits of the floor map.
        use_mask (bool): Whether to use a mask in the floor map.
        use_slices_variance (bool): Whether to use slices variance in the floor map.
        channel_names (list): The names of the channels in the floor map.
        pos_enc_d (int): Dimensions of positonal encodings.
        pos_enc_n (int): Hyperparameter n, used in positional encodings. 
        use_soft_position (bool): Enables soft masking for the src and rec positions.
        use_directional_sources (bool): Enagles direcitonal masking for source positions.
    """
    def __init__(self,
                 resolution: int = 100,
                 slice_coord: Union[str, List[float]] = None,
                 height_selection: Union[str, List[str]] = ['random'],
                 xlim: Tuple[int, int]=[-10, 10],
                 use_mask: bool = True,
                 use_slices_variance = False,
                 pos_enc_d: int = 64,
                 pos_enc_n: int = 10,
                 use_soft_position: bool = False,  # For src and rec, convolves with gaussian for a smooth position
                 use_directional_sources: bool = False,  # For src, convovles with a triangular mask to denote direction
                 device: str = 'cpu'):  

        if not isinstance(slice_coord, list):
            slice_coord = [slice_coord]
        if not isinstance(height_selection, list):
            height_selection = [height_selection]

        if len(slice_coord) != len(height_selection):
            max_len = max(len(slice_coord), len(height_selection))

            while len(slice_coord) < max_len:
                slice_coord.append(slice_coord[-1])
            while len(height_selection) < max_len:
                height_selection.append(height_selection[-1]) 

        self.resolution = resolution
        self.slice_coord = slice_coord
        self.height_selection = height_selection
        self.xlim = xlim
        self.use_mask = use_mask
        self.use_slices_variance = use_slices_variance
        self.pos_enc_d = pos_enc_d
        self.pos_enc_n = pos_enc_n
        self.use_soft_position = use_soft_position
        self.use_directional_sources = use_directional_sources

        self.channel_names = []

        # For larger maps, we blur the src position, just to avoid rounding errors
        validation_kernel = 3
        gaussian_kernel = windows.gaussian(validation_kernel, std=1).reshape(validation_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel * gaussian_kernel.transpose())
        gaussian_kernel /= gaussian_kernel.sum()  # Normalize
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=validation_kernel, stride=1, padding=validation_kernel//2, bias=False)
        self.conv.weight.data = gaussian_kernel[None, None, ...].to(torch.float32)

        self.init_smoother(device)
        if self.use_directional_sources:
            self.init_triangulator(device)

    def init_smoother(self, device):
        # Is using soft postions, we convolve trasnposed the position of src and rec with a large gaussian, for a smooth location
        smoothing_kernel_size = 99
        smoothing_kernel = windows.gaussian(smoothing_kernel_size, std=10).reshape(smoothing_kernel_size, 1)
        smoothing_kernel = torch.from_numpy(smoothing_kernel * smoothing_kernel.transpose())
        smoothing_kernel /= smoothing_kernel.max()  # Normalize
        self.smoother = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=smoothing_kernel_size, stride=1, padding=smoothing_kernel_size//2, bias=False).to(device)
        self.smoother.weight.data = smoothing_kernel[None, None, ...].to(torch.float32).to(device)

    def init_triangulator(self, device):
        # if using directional sources, we convolve tranposed the position of the src with a triangle indicating the pose of the source
        triangle_lernel_size = 31
        triangle_kernel = create_equilateral_triangle_kernel((triangle_lernel_size, triangle_lernel_size))
        triangle_kernel /= triangle_kernel.max()  # Normalize
        self.triangulator = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=triangle_lernel_size, stride=1, padding=triangle_lernel_size//2, bias=False)
        self.triangulator.weight.data = triangle_kernel[None, None, ...].to(torch.float32).to(device)

    def process(self, 
                vertices: Union[np.array, torch.Tensor],
                scenes: Optional[Union[List[str], str]] = None) -> torch.Tensor:
        """
        Parameters:
            scene (Optional[str]): The scene name. Only neede for matterport scenes, when using fixed slices.
        """
        map_list = []
        self.channel_names = []

        for height_sel, coord_slice, in zip(self.height_selection, self.slice_coord):
            floormap = get_floormap_slice(vertices=vertices,
                                          height_selection=height_sel,
                                          resolution=self.resolution,
                                          slice_coord=coord_slice,
                                          scenes=scenes,
                                          xlim=self.xlim)
            map_list.append(floormap)
            self.channel_names.append(f'slice_{height_sel}_{coord_slice}')

        if self.use_slices_variance:
            variance = get_floormap_variance(map_list)
            map_list.append(variance)
            self.channel_names.append(f'slice_variance')

        if self.use_mask:
            mask = get_floormap_mask(vertices=vertices,
                                     resolution=self.resolution,
                                     xlim=self.xlim)
            map_list.append(mask)
            self.channel_names.append('mask')

        map = torch.stack(map_list, dim=0)
        map = map.to(torch.float32)
        return map

    def add_src_rec(self, 
                    floormap: torch.Tensor,
                    src_pos: Union[np.array, torch.Tensor], 
                    rec_pos: Union[np.array, torch.Tensor],
                    rot_angle_src: float = None,
                    debug=False) -> torch.Tensor:
        """
        Adds source and receiver positions to the floor map.
        Parameters:<
            floormap (torch.Tensor): The floor map.
            src_pos (Union[np.array, torch.Tensor]): The source position.
            rec_pos (Union[np.array, torch.Tensor]): The receiver position.
        Returns:
            torch.Tensor: The floor map with added source and receiver positions.
        """
        if torch.tensor(self.xlim).abs().sum() > 20:  
            self.conv.to(floormap.device)
        map_list = []
        if debug:
            print(f'src_pos: {src_pos.shape}')
            print(f'rec_pos: {rec_pos.shape}')
            
        src, rec = get_floormap_src_rec(src_pos=src_pos,
                                        rec_pos=rec_pos,
                                        resolution=self.resolution,
                                        xlim=self.xlim,
                                        device=floormap.device)   
        if debug:
            print(f'src: {src.shape}')
            print(f'rec: {rec.shape}')

        # NOTE: I think thi sis not needed, as self.use_directional_sources ius not really used, nor is the triangulator
        if self.use_soft_position and not self.use_directional_sources:
            src = self.smoother(src[..., None, :, :])
            rec = self.smoother(rec[..., None, :, :])
        elif self.use_soft_position and self.use_directional_sources:
            tmp_dir = self.triangulator(src[..., None, :, :])
            tmp_smooth = self.smoother(src[..., None, :, :])
            src = tmp_dir * tmp_smooth
            rec = self.smoother(rec[..., None, :, :])
        elif not self.use_soft_position and self.use_directional_sources:
            src = self.triangulator(src[..., None, :, :])
        else:
            # Expand channel dimension
            src = src[..., None, :, :]
            rec = rec[..., None, :, :]

        if rot_angle_src is not None:  # TODO this is wrong
            src = torchvision.transforms.functional.affine(src, translate=[0,0], fill=0.0, angle=1 * np.rad2deg(rot_angle_src), shear=0.0, scale=1.0)

        map_list.append(src)
        map_list.append(rec)

        if 'source' not in self.channel_names:
            self.channel_names.append('source')
        if 'receiver' not in self.channel_names:
            self.channel_names.append('receiver')

        if debug:
            for map in map_list:
                print(f'map: {map.shape}')
            print(f'floormap: {floormap.shape}')

        # Batchsize mismatch, remove batch from maps
        if len(floormap.shape) < 4 and len(map_list[0].shape) > 3:
            print('Removing batch from src and rec maps')
            for i, map in enumerate(map_list):
                map_list[i] = map[0, ...]

        if debug:
            for map in map_list:
                print(f'map: {map.shape}')
            print(f'floormap: {floormap.shape}')

        ch_mask = floormap.shape[-3] - 1  # We assume last channel is the floormap mask 
        floormap = torch.concatenate([floormap, *map_list], dim=-3)
        floormap = floormap.to(torch.float32)

        # Validate src and rec
        # The src and rec should inside the scene area, defined by the floorplan mask
        for ii in range(ch_mask+1, ch_mask+3):  
            # For larger maps, we blur the position of src and rec, to consider rounding errors
            if torch.tensor(self.xlim).abs().sum() > 20:  
                #print(floormap.shape)
                #print(self.conv(floormap[..., ii:ii+1, :, :]).squeeze().shape)
                tmp = floormap[..., ch_mask, :, :] * self.conv(floormap[..., ii:ii+1, :, :]).squeeze()
            else:
                tmp = floormap[..., ch_mask, :, :] * floormap[..., ii, :, :]
            try:
                assert torch.min(tmp.sum(dim=[-1,-2])).item() > 0.0, f'ERROR: For some items in the batch, the source or rec is outside the floorplan mask. Check augmentaion or the floormaps generation.'
            except AssertionError as e:
                print(f'Inside exception in features.py')
                print(tmp.sum(dim=[-1,-2]))
                print(torch.where(tmp.sum(dim=[-1,-2]) < 1))
                #raise e  # Disable for now
        return floormap

    def add_pose_vector(self,
                        floormap: torch.Tensor,
                        rot_angle_src: torch.Tensor = None,
                        debug=False) -> torch.Tensor:
        """
        This appends an extract channel to the floormap, that represents the orientation of the source.
        This is a very rudimentary way to add some directional information to the maps, when using augmentations.
        This assumes that the rotation angle is applied to all elements in the batch.
        """
        map_list = []
        # Use numpy here because it supports advanced indexing
        grid = np.zeros((self.resolution, self.resolution))
        center = [self.resolution//2, self.resolution//2]  
        length = self.resolution//2 - 1

        vector = np.array([-1, 0])
        vector = np.round(vector * length).astype(int)

        spaces = np.linspace(0, 1, length+1).reshape(-1,1)  # Generate an array of equally spaced values between 0 and 1
        points = np.round(center + spaces*vector).astype(int)  # Calculate all points along the line using numpy
        grid[points[:,0], points[:,1]] = 1  # Draw a line starting from center using numpy
        grid = torch.from_numpy(grid)[..., None, :, :].to(floormap.device)
        
        # We find the direction for every item in the batch
        # That can have different rotation angles
        if rot_angle_src is not None: 
            if not isinstance(rot_angle_src, torch.Tensor):
                rot_angle_src = torch.Tensor(rot_angle_src)
            if len(rot_angle_src.shape) < 1:
                rot_angle_src = rot_angle_src[None, ...]
            for j, this_rot_angle in enumerate(rot_angle_src):
                this_grid = torchvision.transforms.functional.affine(grid.clone(), translate=[0,0], fill=0.0, angle=1 * np.rad2deg(this_rot_angle.item()), shear=0.0, scale=1.0)
                if len(floormap.shape) > 3:
                    this_grid = this_grid.clone()[None, ...]
                map_list.append(this_grid)

        map_list = torch.cat(map_list, dim=0)
        floormap = torch.cat([floormap, map_list], dim=-3).to(torch.float32)
        if 'pose' not in self.channel_names:
            self.channel_names.append('pose')

        return floormap

        if False:
            # This is wrong, I was assuming that we have the same rotation angle for all items in the batch
            # DELETE
            if isinstance(rot_angle_src, torch.Tensor):
                #this_angle = rot_angle_src.item()
                if len(rot_angle_src.shape) > 0:  # for batches, we want to have the same pose for all elements
                    diffo = rot_angle_src.diff()
                    if diffo.sum() != 0.0:
                        print(f'WARNING: I found different poses i the batch:')
                        print(rot_angle_src)
                    rot_angle_src = rot_angle_src[0]    
                this_angle = rot_angle_src.item()
            grid = torchvision.transforms.functional.affine(grid, translate=[0,0], fill=0.0, angle=1 * np.rad2deg(this_angle), shear=0.0, scale=1.0)

            if len(floormap.shape) > 3:
                grid = grid.repeat(floormap.shape[0], 1, 1, 1)
            map_list.append(grid)
            floormap = torch.cat([floormap, *map_list], dim=-3)
            floormap = floormap.to(torch.float32)

    def add_pose_vector_OLD(self,
                        floormap: torch.Tensor,
                        rot_angle_src: float = None,
                        debug=False) -> torch.Tensor:
        """
        This appends an extract channel to the floormap, that represents the orientation of the source.
        This is a very rudimentaty way to add some directional information to the maps, when using augmentations.
        """
        map_list = []
        grid = torch.zeros((self.resolution, self.resolution))
        center = [self.resolution//2, self.resolution//2]  
        length = self.resolution//2
        vector = torch.tensor([0, 1])

        # Multiply by length and round for indexing.
        vector = torch.round(vector * length).to(torch.int32)

        if False:
            # Draw a line starting from center
            end_point = center + vector
            for i in range(length+1):
                point = torch.round(center + i*(vector)/length).to(torch.int32)
                grid[point[0], point[1]] = 1
        
        spaces = torch.linspace(0, 1, length+1).reshape(-1,1)  # Generate an array of equally spaced values between 0 and 1
        points = torch.round(center + spaces*vector).to(torch.int32)  # Calculate all points along the line
        grid[points[:,0], points[:,1]] = 1  # Draw a line starting from center

        map_list.append(grid[..., None, :, :])
        floormap = torch.concatenate([floormap, *map_list], dim=-3)
        floormap = floormap.to(torch.float32)

        return floormap
    
    def add_positional_encoding(self, 
                                floormap: torch.Tensor,
                                this_pos: Union[np.array, torch.Tensor], 
                                name: str =  'src',
                                debug=False) -> torch.Tensor:
        """
        Adds posotional encodings of the source or reciever to the floor map.
        This takes the (x,y) pixel coordinate (discrete) of the src or rec, and first maps to a continuos vector
        using sinusaidal encodings. This vector is then tiled to generate an image-like tensor.

        Parameters:
            floormap (torch.Tensor): The floor map.
            pos (Union[np.array, torch.Tensor]): The position of either the source or the reciever to add.
        Returns:
            torch.Tensor: The floor map with added source and receiver positions.
        """
        map_list = []
        if debug:
            print(f'this_pos: {this_pos.shape}')
            
        # Map cartesian coordinates to pixel coordinates
        this_pos = this_pos.clone()
        pix2meter = self.resolution / torch.tensor(self.xlim).abs().sum()
        this_pos[..., 0] = this_pos[..., 0] * pix2meter + self.resolution // 2
        this_pos[..., 1] = self.resolution // 2 - this_pos[..., 1] * pix2meter


        enc = get_floormap_positonal_encodings(pos=this_pos,
                                               d=self.pos_enc_d,
                                               n=self.pos_enc_n,
                                               resolution=self.resolution,
                                               device=floormap.device)   
        map_list.append(enc)

        if debug:
            print(f'src: {this_pos.shape}')

        if f'pos_enc_x_{name}' not in self.channel_names:
            self.channel_names.append(f'pos_enc_x_{name}')
        if f'pos_enc_y_{name}' not in self.channel_names:
            self.channel_names.append(f'pos_enc_y_{name}')

        if debug:
            for map in map_list:
                print(f'map: {map.shape}')
            print(f'floormap: {floormap.shape}')

        # Batchsize mismatch, remove batch from maps
        if len(floormap.shape) < 4 and len(map_list[0].shape) > 3:
            print('Removing batch from src and rec maps')
            for i, map in enumerate(map_list):
                map_list[i] = map[0, ...]

        if debug:
            for map in map_list:
                print(f'map: {map.shape}')
            print(f'floormap: {floormap.shape}')

        floormap = torch.concatenate([floormap, *map_list], dim=-3)
        floormap = floormap.to(torch.float32)
        return floormap

    def __repr__(self):
        fmt_str = f'FloorMapProcessor \n'
        fmt_str += f'    resolution = {self.resolution} \n'
        fmt_str += f'    slice_coord = {self.slice_coord} \n'
        fmt_str += f'    height_selection = {self.height_selection} \n'
        fmt_str += f'    xlim = {self.xlim} \n'
        fmt_str += f'    use_slices_variance = {self.use_slices_variance} \n'
        fmt_str += f'    use_soft_position = {self.use_soft_position} \n'
        fmt_str += f'    channel_names = {self.channel_names} \n'
        return fmt_str

def get_floormap_slice(vertices: Union[np.array, torch.Tensor], 
                       height_selection: str = "random",
                       resolution=100,
                       slice_coord=None,
                       xlim=[-10, 10],
                       scenes: Optional[Union[List[str], str]] = None) -> torch.Tensor:
    """ Returns a discretized floormap, by selecting a slice of the mesh at some height z

        Parameters:
            vertices (Union[np.array, torch.Tensor]): The vertices of the mesh.
            height_selection (str, optional): The method used to select the height of the slice. Can be one of:
                - 'random': Select a random height within the range of the mesh.
                - 'fixed': Use a fixed height specified in the `slice_coord` parameter.
                - 'relative_floor': Select a height relative to the floor level of the mesh, in meters (e.g. 1.0 meters above the floor).
                - 'relative_ceiling': Select a height relative to the ceiling level of the mesh.
                - 'relative_center': Select a height relative to the middle height of the mesh.
                - 'relative_percent': Select a height relative to the floor level of the mesh, in percent (e.g. 0.5 is half the total heigh).
            resolution (int, optional): The resolution of the discretized floormap. Default is 100.
            slice_coord (float, optional): The coordinate of the slice. Required when height_selection is 'fixed'.
            xlim (list, optional): The x limits for the floormap. Default is [-10, 10].
            scenes: Scene names, only needed for matterport when using fixed slices.
        
        Returns:
            torch.Tensor: The discretized floormap.
        
        Raises:
            ValueError: If height_selection is 'fixed' but slice_coord is None.
    """
    if height_selection == 'random':
        slice_coord = np.random.uniform(np.min(vertices[:, 2]) + 0.3, np.max(vertices[:, 2]) - 0.3)
        z_range = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        if z_range > 4:
            warnings.warn(f'WARNING: The range for height is {z_range}. Is this a multi-story apartment?')
    elif height_selection == 'fixed':
        assert slice_coord is not None, f'ERROR, slice_coord should be a valid float when using fixed height seleciton mode, not {slice_coord}'
        slice_coord = slice_coord
        if scenes is not None:
            tmp = []
            if isinstance(scenes, str):
                scenes = [scenes]
            for this_scene in scenes:
                if this_scene in scenes_matterport:
                    tmp.append(fixed_slice_coords_matterport[this_scene])
                else:
                    tmp.append(slice_coord)
    elif height_selection == 'relative_floor':
        slice_coord = vertices[:, 2].min() + slice_coord  # above floor
    elif height_selection == 'relative_ceiling':
        slice_coord = vertices[:, 2].max() - slice_coord  # below ceiling
    elif height_selection == 'relative_percent':
        assert 0.0 < slice_coord < 1.0, f'ERROR, slice coord most be without [0, 1] when using heigh_mode relative_percent, not {slice_coord}'
        height_range = vertices[:, 2].max() - vertices[:, 2].min()
        slice_coord = height_range * slice_coord
    elif height_selection == 'relative_center':
        if scenes is not None:
            tmp = []
            if isinstance(scenes, str):
                this_scene = scenes
                if this_scene in scenes_matterport:
                    slice_coord = fixed_slice_coords_matterport[this_scene]
                elif this_scene in fixed_slice_replica.keys():
                    slice_coord = fixed_slice_replica[this_scene]
                else:
                    height_range = vertices[:, 2].max() - vertices[:, 2].min()
                    center_height = vertices[:, 2].min() + height_range/2
                    slice_coord = center_height + slice_coord  # relative to center
    else:
        raise ValueError(f'ERROR, height_selection should be a valid float when using fixed height seleciton mode, not {height_selection}')

    # Select vertices in slice
    vertices = vertices[np.isclose(vertices[:, 2], slice_coord, atol=1e-1), :]

    #print('INSIDE features')
    #print(vertices.shape)
    #print(scenes)
    # Discretize into bins
    binning = np.linspace(xlim[0], xlim[1], resolution + 1)
    vertices = np.digitize(vertices, binning) - 1
    vertices, indices = np.unique(vertices, axis=0, return_index=True)  # Remove duplicates

    # Floormap extraction
    i = vertices[:, 0]
    j = resolution - 1 - vertices[:, 1]
    floormap = np.zeros((resolution, resolution))
    floormap[i, j] = 1
    floormap = torch.tensor(floormap)

    return floormap

def get_floormap_src_rec(src_pos: Union[np.array, torch.Tensor] = None, 
                         rec_pos: Union[np.array, torch.Tensor] = None, 
                         resolution=100,
                         xlim=[-10, 10],
                         device=None,
                         debug=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns a 2d binary map for the location of the source and receiver.

        Parameters:
            src_pos (Union[np.array, torch.Tensor]): The batch of source positions.
            rec_pos (Union[np.array, torch.Tensor]): The batch of receiver positions.
            resolution (int, optional): The resolution of the discretized floormap. Default is 100.
            xlim (list, optional): The x limits for the floormap. Default is [-10, 10].
        
        Returns:
            torch.Tensor: The discretized mask for the floormap.
    """
    if len(src_pos.shape) < 2:  # Should be [1,3]
        src_pos = src_pos[None, ...]
    if len(rec_pos.shape) < 2:  # Should be [1,3]
        rec_pos = rec_pos[None, ...]

    if not isinstance(src_pos, torch.Tensor):
        src_pos = torch.from_numpy(src_pos)
    if not isinstance(rec_pos, torch.Tensor):
        rec_pos = torch.from_numpy(rec_pos)
    if device is None:
        device = src_pos.device
        
    assert src_pos.shape[1:] == (3,), f"ERROR, source should be [batch_size, 3], not {src_pos.shape}"
    assert rec_pos.shape[1:] == (3,), f"ERROR, receiver should be [batch_size, 3], not {rec_pos.shape}"

    # Hack to prevent errors with rotations and translations
    # Where the src or rec can be slightly outside the [-10, 10] range.
    # I dont think this is a huge problem, as long as the error is small, a few centimeters.
    src_pos = torch.clamp(src_pos, xlim[0], xlim[1])
    rec_pos = torch.clamp(rec_pos, xlim[0], xlim[1])

    if debug:
        print(f'src_pos.shape {src_pos.shape}')
        print(f'src_pos {src_pos}')
        print(f'rec_pos.shape {rec_pos.shape}')
    
    # Discretize into bins    
    vertices = torch.stack([src_pos, rec_pos], dim=1)  # [batch, pos_id,  3]
    binning = torch.linspace(xlim[0], xlim[1], resolution + 1, device=device)
    vertices = torch.bucketize(vertices, binning, out_int32=True, right=True) - 1

    if True:
        # Batch support without for loops, this seems to work ok
        maps = []
        vertices = [vertices[:, 0, :], vertices[:, 1, :]]

        for ctr in range(2):
            this_vertices = vertices[ctr]
            map = torch.zeros((src_pos.shape[0], resolution, resolution), device=device)
            i = this_vertices[:, 0]
            j = resolution - 1 - this_vertices[:, 1]
            b = torch.arange(src_pos.shape[0])

            # Hack, make sure all indices are inside the image
            # This is usually a problem when the augmention is wrong,
            # otherwise this should not really happen
            if torch.logical_or(i >= resolution, i < 0).any(): 
                print(f'WARNING: Some coordinates are outisde the 0-128 range when adding source or receivers.')
                print(i)
            if torch.logical_or(j >= resolution, j < 0).any(): 
                print(f'WARNING: Some coordinates are outisde the 0-128 range when adding source or receivers.')
                print(j)
            i = torch.clamp(i, min=0, max=resolution-1)
            j = torch.clamp(j, min=0, max=resolution-1)
            map[b, i, j] = 1
            if debug:
                print(f'i {i}')
                print(f'j {j}')
                print(f'map.shape {map.shape}')
                print(f'map.sum() {map.sum()}')
                print(f'map {map}')
            cond = map > 0.0

            assert cond.any(dim=1, keepdims=True).any(dim=2, keepdims=True).all(), f'ERROR, the src map appears to be empty for some examples in the batch, is the src inside the image range? src:{src_pos} rec:{rec_pos}'
            maps.append(map)

        return maps[0], maps[1]

    if False:
        # Floormap extraction, slow but works
        maps = []
        vertices = [vertices_src, vertices_rec]
        for ctr in range(2):
            this_vertices = vertices[ctr]
            map = torch.zeros((src_pos.shape[0], resolution, resolution), device=device)
            for b in range(src_pos.shape[0]):
                i = this_vertices[b, 0]
                j = resolution - 1 - this_vertices[b, 1]
                if debug:
                    print(f'i {i}')
                    print(f'j {j}')
                map[b, i, j] = 1
            maps.append(map)
        return maps[0], maps[1]
    
def get_floormap_positonal_encodings(pos: Union[np.array, torch.Tensor] = None, 
                                     resolution=100,
                                     d=64,
                                     n=10,
                                     device=None,
                                     debug=False) -> torch.Tensor:
    """ Returns a 2d image-like tensor for sinusoidal positional encodings of the positions

    Basic mapping from  (x,y) --> (2, resolution, resolution) using positional encodings
    The main idea is to expand an scalar 2d coordinate into a 2 x d_dimensional vector 
    using sinusoidal positonal encodings like a transformer.
    Inputs should be (x,y) batched coordinates in PIXEL coordinates, so integers.

        Parameters:
            pos (Union[np.array, torch.Tensor]): The batch of positions, either source or recs.
            resolution (int, optional): The resolution of the discretized floormap. Default is 100.
            xlim (list, optional): The x limits for the floormap. Default is [-10, 10].
        
        Returns:
            torch.Tensor: The 2 channel image-like tensor of the positonal encodings.
    """
    if len(pos.shape) == 1:
        pos = pos[None, ...]  # Add batch if needed
    assert len(pos.shape) == 2 and (pos.shape[1] == 2 or pos.shape[1] == 3), f'This only support batched 2d or 3d tensors, not {pos.shape}'

    if not isinstance(pos, torch.Tensor):
        pos = torch.from_numpy(pos)
    if device is None:
        device = pos.device

    if ~torch.all(pos < 1.0) or torch.any(pos < 0.0):
        warnings.warn(f'WARNING: The coordinates in tensor do not look like pixel coordinates (e.g. integers in the range [0, {resolution})')

    # Hack to prevent errors with rotations and translations
    # I dont think this is a huge problem, as long as the error is small, a few centimeters.
    # Remember, these are pixel coordinates
    pos = torch.clamp(pos, 0, resolution)
    
    # Encodings:
    i = torch.arange(d, device=device)  # dimensionality of embeddings
    x = pos[:, 0:1]  # [batch, 1]
    y = pos[:, 1:2]  # [batch, 1]
    
    encodings = []
    for ii, k in enumerate([x, y]):  # only for x,y
        freq = k / (n**(i/d))  # [batch, d]
        tmp = torch.cat([torch.cos(freq), torch.sin(freq)], dim=-1)
        tmp = torch.repeat_interleave(tmp, resolution//(d*2), dim=-1) 
        tmp = torch.tile(tmp[:, None, :], dims=(1, resolution, 1))
        if ii > 0:
            tmp = tmp.permute((-3, -1,-2))
        encodings.append(tmp)
    encodings = torch.stack(encodings, dim=-3)

    return encodings

def get_floormap_mask(vertices: Union[np.array, torch.Tensor], 
                      resolution=100,
                      xlim=[-10, 10]) -> torch.Tensor:
    """ Returns a mask for the floormap, buy doing a linear projection of the vertices to the (x,y) plane.

        Parameters:
            vertices (Union[np.array, torch.Tensor]): The vertices of the mesh.
            resolution (int, optional): The resolution of the discretized floormap. Default is 100.
            xlim (list, optional): The x limits for the floormap. Default is [-10, 10].
        
        Returns:
            torch.Tensor: The discretized mask for the floormap.
    """
    # Linear projection to x,y plane
    vertices = vertices[:, 0:2]

    # Discretize into bins
    binning = np.linspace(xlim[0], xlim[1], resolution + 1)
    vertices = np.digitize(vertices, binning) - 1
    vertices, indices = np.unique(vertices, axis=0, return_index=True)  # Remove duplicates

    # Floormap extraction
    i = vertices[:, 0]
    j = resolution - 1 - vertices[:, 1]
    floormap = np.zeros((resolution, resolution))
    floormap[i, j] = 1
    floormap = torch.tensor(floormap)

    return floormap

def get_floormap_variance(floormaps: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """ Computes the variace across channels for different slices of floormaps """
    if isinstance(floormaps, list):
        floormaps = torch.stack(floormaps, dim=0)
    if len(floormaps.shape) > 4:
        raise ValueError('ERROR: variance calculation not support for batched floormaps yets')
    var = torch.var(floormaps, dim=0, keepdim=False)

    return var

def create_equilateral_triangle_kernel(size):
    # Calculate height of equilateral triangle
    height = size[1] * torch.sqrt(torch.tensor(3.)) / 2

    # Define vertices
    top_vertex = [size[0]/4, 0]  # [size[0]/2, 0]
    bottom_left_vertex = [0, height*2]
    bottom_right_vertex = [size[0]/2, height*2]  # [size[0], height]

    # Create grid
    x_grid, y_grid = torch.meshgrid(torch.linspace(0, size[0], steps=size[0]), torch.linspace(0, size[1], steps=size[1]))
    x_grid, y_grid = x_grid.t(), y_grid.t()  # Transpose to match grid orientation

    # Check if a point (x, y) lies inside the triangle
    x, y = x_grid, y_grid
    v0 = [bottom_right_vertex[0] - bottom_left_vertex[0], bottom_right_vertex[1] - bottom_left_vertex[1]]
    v1 = [top_vertex[0] - bottom_left_vertex[0], top_vertex[1] - bottom_left_vertex[1]]
    v2 = [x - bottom_left_vertex[0], y - bottom_left_vertex[1]]
    invDenom = 1 / (v0[0] * v1[1] - v1[0] * v0[1])
    u = (v2[1] * v0[0] - v2[0] * v0[1]) * invDenom
    v = (v1[1] * v2[0] - v1[0] * v2[1]) * invDenom

    mask = ((u >= 0) & (v >= 0) & (u + v < 1))

    # Return Triangle Kernel
    return mask.float()

def get_positional_encodings(x : torch.Tensor, k=10, F=1025):
    """ Returns frequency positional encodings for spectrogram x, that can be concatenated as extra features.
    Based on:
    PoCoNet: Better Speech Enhancement with Frequency-Positional Embeddings, Semi-Supervised Conversational
    Data, and Biased Loss
    https://arxiv.org/pdf/2008.04470.pdf

    Arguments
    ---------
    k : int
        Number of encodings.
    F : int
        Number of frequency bins, so 1025 for a 2048 nfft

    Useful for debugging:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(features[0])
        plt.show()
    """
    # the positional encodings for freq are divided by F, section 2.1.1 of POCONET paper
    n = torch.arange(F+1, device=x.device) / (F-1)
    features = []
    for ii in range(k):
        cos = torch.cos(2**(ii-1) * torch.pi * n)
        features.append(cos)
    features = torch.stack(features, dim=0)
    features = torch.tile(features[..., None], [1, 1, x.shape[-1]])
    if len(x.shape) == 4:  # Add batch dim
        features = torch.tile(features[None, ...], (x.shape[0], 1, 1, 1))
    x = torch.cat([x, features], dim=-3)  # concat across channels

def get_basic_positional_encoding(x : torch.Tensor) -> torch.Tensor:
    _, height, width = tensor.shape
    x_pos = torch.arange(width)[None, :] / width
    y_pos = torch.arange(height)[:, None] / height

    x_pos = x_pos.unsqueeze(0)
    y_pos = y_pos.unsqueeze(0)
    
    tensor += x_pos + y_pos
    return tensor


class AcuMapProcessor(object):
    """
    A class used to extract acoustic maps from previously processed acoustic parameters.
    
    Attributes:
        resolution (int): The resolution of the acoustic map.
        parameters (list): The acoustic parameters to use.
        freq_bands (list): The frequency bands to use.
        xlim (tuple): The x limits for the acoustic map.
        use_pooling (bool): Whether to use pooling in the acoustic map.
        use_lowpass (bool): Whether to use lowpass filter in the acoustic map.
        pooling_kernel (int): The kernel size for pooling.
        pooling_stride (int): The stride for pooling.
        pooling_padding (int): The padding for pooling.
        channel_names (list): The names of the channels in the acoustic map.
        mask (torch.Tensor): The mask for pooling.
        pooler (torch.nn.Module): The pooling layer.
        available_freq_bands (list): The available frequency bands.
        lowpass (torch.nn.Module): The lowpass filter layer.
        distances (list): List of distance functions to eveluate performance. [l1, l2, mape]
        clamp_parameter_values (bool): Clamps values of perimeters to a predetermined range. This is because e.g. 100 dB of C50 is too much.
    """
    def __init__(self,
                 parameters: Union[str, List[str]] = ['c50', 't30', 'drr'],
                 freq_bands: List[int] = [1000],  # frequency bands to use, choose [125, 250, 500, 1000, 2000, 4000, 8000]
                 resolution: int = 100,
                 xlim: Tuple[int, int] = [-10, 10],
                 use_pooling: bool = False,
                 pooling_kernel: int = 9,
                 pooling_stride: int = 1,
                 pooling_padding: int = 4,
                 use_lowpass: bool = False,
                 lowpass_std: float = 1,
                 distances: List[str] = ['l1'],
                 clamp_parameter_values: bool = False  # Clamps values of permiters to a fixed range
                 ):

        if not isinstance(parameters, list):
            parameters = [parameters]
        
        self.resolution = resolution
        self.parameters = parameters
        self.freq_bands = freq_bands
        self.xlim = xlim
        self.use_pooling = use_pooling
        self.use_lowpass = use_lowpass
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride
        self.pooling_padding = pooling_padding
        self.channel_names = []
        self.mask = None
        self.mask_pooled = None  # For plotting only
        self.pooler = None
        self.distances = distances
        self.clamp_parameter_values = clamp_parameter_values

        if self.clamp_parameter_values:
            raise NotImplementedError  # This is not ready

        self.available_freq_bands = [125, 250, 500, 1000, 2000, 4000, 8000]

        # Lowpass filter by convolution with a gaussian kernel
        if self.use_lowpass:
            gaussian_kernel = windows.gaussian(pooling_kernel, std=lowpass_std).reshape(pooling_kernel, 1)
            gaussian_kernel = torch.from_numpy(gaussian_kernel * gaussian_kernel.transpose())
            #gaussian_kernel = gaussian_kernel[None, None, ...].to(torch.float32)
            gaussian_kernel /= gaussian_kernel.sum()  # Normalize

            conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=pooling_kernel, stride=pooling_stride, padding=pooling_padding, bias=False)
            conv.weight.data = gaussian_kernel[None, None, ...]
            self.lowpass = conv
        else:
            self.lowpass = None

        for freq in freq_bands:
            assert freq in self.available_freq_bands, f'ERROR: freuency band {freq} is not avaialable in {sefl.available_freq_bands}'

    def process(self, 
                acu_params_dict: Dict,
                src_id: str,
                rec_ids: np.array = None,
                centroid: np.array = None) -> torch.Tensor:
        map_list = []
        self.channel_names = []
        assert isinstance(src_id, str), 'ERROR: src_id should be a str, not int'
        if rec_ids is not None:
            assert rec_ids.dtype == np.int64 or rec_ids.dtype == np.int32, 'ERROR: rec_ids should be a np array of ints'
        
        # Find entries with the selected source
        keys = []
        selected_acu_params = []
        ctr = 0
        for k,v in acu_params_dict.items():
            _, tmp = k.split('/')  # office_0/9_19  --> office_0 , 9_19
            this_src_id, this_rec_id = tmp.split('_')  # 9_1  --> 9 , 19   # src-id, rec-id
            keys.append(k)
            if False:  # TODO, clear this
                if ctr < 100:
                    print(f'YOLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO {this_src_id}  {this_rec_id}  {src_id}')
                    print(f'{this_src_id == src_id}')
                    print(f'{type(this_src_id)} {type(src_id)}')
                ctr += 1
            if rec_ids is None:  # Append all recs, for a valid source
                if this_src_id == src_id:                
                    selected_acu_params.append(v)
            else:
                #print(f'this_rec_id ==  rec_ids {this_rec_id} {rec_ids} {int(this_rec_id) in rec_ids}')
                if this_src_id == src_id and int(this_rec_id)in rec_ids: 
                    selected_acu_params.append(v)

        # Check if source is valid, otherwise return NaNs (empty acumap)
        if len(selected_acu_params) < 1:
            warnings.warn(f'WARNING: The src_id {src_id} was not found in the processed json. Returning empty acumap.')
            map = torch.ones((self.resolution, self.resolution))
            map[:,:] = np.nan
            for acu_param in self.parameters:
                for freq in self.freq_bands:
                    map_list.append(map)
                    self.channel_names.append(f'map_{acu_param}_{freq}')
            map = torch.stack(map_list, dim=0)
            return map 
        
        if len(selected_acu_params) < 100:
            warnings.warn(f'WARNING: I found {len(selected_acu_params)} recs for this source. The acumap will be very sparse. Make sure that n_files_per_scene is large enough.')
        
        # Get mask for pooling
        if self.use_pooling:
            if True or self.mask is None:  # TODO fix masking, right now we are forcing the mask computation every tme
                rec_coords = []
                for metrics in selected_acu_params:
                    rec_coords.append(metrics['rec'])

                # Optional centering
                if centroid is not None:
                    rec_coords = rec_coords - centroid

                mask = get_acumap_mask(vertices=np.array(rec_coords)[:, 0:2],
                                       resolution=self.resolution,
                                       xlim=self.xlim)
                self.mask = mask
            if self.pooler is None:
                pooler = MaskedAvgPool(kernel_size=self.pooling_kernel, 
                                       stride=self.pooling_stride, 
                                       padding=self.pooling_padding)
                self.pooler = pooler

            if True or self.mask_pooled is None:  # TODO fix masking
                self.mask_pooled = self.pooler(self.mask[None, ...], self.mask[None, ...])
                self.mask_pooled = self.mask_pooled[0].to(torch.bool)

        # Get acumaps
        for acu_param in self.parameters:
            for freq in self.freq_bands:
                freq_id = self.available_freq_bands.index(freq)
                tmp_values = []
                rec_coords = []
                for metrics in selected_acu_params:
                    if '/' in acu_param:  # directional metrics
                        # f'metrics_{angles[jj]}
                        #print(metrics.keys())
                        this_acu_param_dir, this_acu_param_label = acu_param.split('/')  # 1/c50 --> dir_1, c50
                        #print(this_acu_param_dir)
                        #print(this_acu_param_label)
                        this_value = metrics[f'metrics_{this_acu_param_dir}'][this_acu_param_label][freq_id]    
                    else:
                        this_value = metrics['metrics'][acu_param][freq_id]
                    if torch.isinf(torch.tensor(this_value)):
                        # Ignore recs with infinite values
                        # Infinite values are most likely due to errors in the acoustic parameter extraction
                        continue  
                    if self.clamp_parameter_values:
                        this_value = clamp_acu_param(this_value, acu_param)
                    if not validate_acu_param(this_value, acu_param):
                        # Ignore recs with suspicous values
                        # This is because some RIRs have problems
                        #continue
                        # Set to nan those rirs that look strange
                        this_value = np.nan
                        #pass

                    # Hack for multislope values of DecayFitNet
                    # We force the ampltiude to dB
                    if acu_param in ['a0', 'a1', 'a2']:
                        this_value = 10 * np.log10((this_value/100) + 1e-12)
                        this_value = np.clip(this_value, a_min=-50, a_max=None)  # clamp everything under -50 db
                    if acu_param in ['t0', 't1', 't2']:
                        this_value = np.clip(this_value, a_min=0.0, a_max=None)  # clamp negative RT to 0.0                    
                    rec_coords.append(metrics['rec'])
                    tmp_values.append(this_value)
                    
                # Optional centering
                if centroid is not None:
                    rec_coords = rec_coords - centroid
                
                map = get_acumap_base(vertices=np.array(rec_coords)[:, 0:2],
                                      values=tmp_values,
                                      resolution=self.resolution,
                                      xlim=self.xlim)
                
                if self.use_pooling:
                    map = self.pooler(map[None, ...], self.mask[None, ...])
                    map = map[0]

                    if self.use_lowpass:
                        with torch.no_grad():
                            map = self.lowpass(map[None, ...])
                            map = map[0]

                    if self.mask_pooled is not None:
                        map[torch.logical_not(self.mask_pooled)] = np.nan
                
                map_list.append(map)
                self.channel_names.append(f'map_{acu_param}_{freq}')

        map = torch.stack(map_list, dim=0)
        return map

    def __repr__(self):
        fmt_str = f'AcuMapProcessor \n'
        fmt_str += f'    resolution = {self.resolution} \n'
        fmt_str += f'    parameters = {self.parameters} \n'
        fmt_str += f'    freq_bands = {self.freq_bands} \n'
        fmt_str += f'    freq_bands = {self.distances} \n'
        fmt_str += f'    use_pooling = {self.use_pooling} \n'
        fmt_str += f'    use_lowpass = {self.use_lowpass} \n'
        fmt_str += f'    xlim = {self.xlim} \n'
        fmt_str += f'    pooling:  k={self.pooling_kernel}, s={self.pooling_stride}, p={self.pooling_padding} \n'
        fmt_str += f'    channel_names = {self.channel_names} \n'
        return fmt_str

def validate_acu_param(value, acu_param: str):
    """ Validates that the value for the acoustic parameters is ok.
    This is mostly needed becase some RIRs have errors, and creates problems for the acoustic params.
    For example, some RIRs are a single impulse, and this creates very large C50.
    """
    max_rt = 8.0  # in seconds  4.0
    min_rt = 0.05  # in seconds  0.125, or 0.05
    max_db = 40  # in db
    min_db = -40  # in db

    if acu_param in ['edt', 't10', 't20', 't30']:
        return (min_rt < value < max_rt)
    if acu_param in ['drr', 'c50']:
        return (min_db < value < max_db)
    else:
        return True  # For other paramters not considered here, like n_slopes, curvature, etc ...
    
def clamp_acu_param(value, acu_param: str):
    """ Clamps the values of the acoustic parameter to a pre detemrined range.
    This is an alternative, instead of setting outlier values to NaN, we clip it so something more reasonable."""
    raise NotImplementedError

def get_acumap_mask(vertices: Union[np.array, torch.Tensor], 
                    resolution=100,
                    xlim=[-10, 10]) -> torch.Tensor:
    """ Returns a discretized mask, 

        Parameters:
            vertices (Union[np.array, torch.Tensor]): The vertices (2d coordinates( of the recierves.
            values: Union[List[float], np.array, torch.Tensor] : The acoustic parameter values for each point
            resolution (int, optional): The resolution of the discretized floormap. Default is 100.
            xlim (list, optional): The x limits for the floormap. Default is [-10, 10].
        
        Returns:
            torch.Tensor: The discretized floormap.
    """
    assert vertices.shape[-1] == 2, 'AcuMap processing only works with 2d coordinates for the receivers'

    # Discretize into bins
    binning = np.linspace(xlim[0], xlim[1], resolution + 1)
    vertices = np.digitize(vertices, binning) - 1
    vertices, indices = np.unique(vertices, axis=0, return_index=True)  # Remove duplicates

    # Mask extraction
    i = vertices[:, 0]
    j = resolution - 1 - vertices[:, 1]
    mask = np.zeros((resolution, resolution)) 
    mask[i, j] = 1
    mask = torch.tensor(mask, dtype=torch.bool)

    return mask

def get_acumap_base(vertices: Union[np.array, torch.Tensor], 
                    values: Union[List[float], np.array, torch.Tensor],
                    resolution=100,
                    xlim=[-10, 10]) -> torch.Tensor:
    """ Returns a discretized acuamp, 

        Parameters:
            vertices (Union[np.array, torch.Tensor]): The vertices (2d coordinates( of the recierves.
            values: Union[List[float], np.array, torch.Tensor] : The acoustic parameter values for each point
            resolution (int, optional): The resolution of the discretized floormap. Default is 100.
            xlim (list, optional): The x limits for the floormap. Default is [-10, 10].
        
        Returns:
            torch.Tensor: The discretized floormap.
    """
    assert vertices.shape[-1] == 2, 'AcuMap processing only works with 2d coordinates for the receivers'

    # Discretize into bins
    binning = np.linspace(xlim[0], xlim[1], resolution + 1)
    vertices = np.digitize(vertices, binning) - 1
    vertices, indices = np.unique(vertices, axis=0, return_index=True)  # Remove duplicates

    # Floormap extraction
    i = vertices[:, 0]
    j = resolution - 1 - vertices[:, 1]
    acumap = np.zeros((resolution, resolution)) 
    #for row, (i, j) in enumerate(zip(i,j)):
    #    acumap[i, j] = values[row]
    acumap[i, j] = np.array(values)[indices]
    acumap = torch.tensor(acumap)

    return acumap
