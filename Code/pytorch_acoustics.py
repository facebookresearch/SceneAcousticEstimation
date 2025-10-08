# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


################################################
# This file contains functions to process acoustical parameters of room impulse responses
#
# Based heavily on:
# https://github.com/georg-goetz/pytorch-acoustics-toolbox/
# https://github.com/georg-goetz/DecayFitNet/
# https://github.com/samuiui/roomaco/blob/master/roomparameters.py
# https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/room.py
# Ricardo Falcon 2022
import warnings

# Here we vcan get these acoustical parameters:
#   -

################################################
import torch
import torch.nn as nn
import torchaudio
import scipy.signal
import numpy as np
import time
import spaudiopy as spa
from typing import List, Tuple, Dict

import losses
import utils

eps = 1e-10

class FilterByOctaves(nn.Module):
    """Generates an octave wide filterbank and filters tensors.
    This coulbd be gpu compatible if using torch backend, but it is super slow and should not be used at all.
    UNLESS, the new update to torchaudio has massively improved the biquad filters.
    The octave filterbanks is created using cascade Buttwerworth filters, which then are processed using
    the biquad function native to PyTorch.
    This is useful to get the decay curves of RIRs.
    """

    def __init__(self, center_frequencies=None, order=5, sample_rate=48000, backend='scipy'):
        super(FilterByOctaves, self).__init__()

        if center_frequencies is None:
            center_frequencies = [125, 250, 500, 1000, 2000, 4000, 8000]
        self._center_frequencies = center_frequencies
        self._order = order
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(center_frequencies, self._sample_rate, self._order)
        self.backend = backend

    def _forward_scipy(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for this_sos in self._sos:
            tmp = torch.clone(x).cpu().numpy()
            tmp = scipy.signal.sosfilt(this_sos, tmp, axis=-1)
            out.append(torch.from_numpy(tmp.copy()))
        out = torch.stack(out, dim=-2)  # Stack over frequency bands

        return out

    def set_sample_rate(self, sample_rate: int):
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(self._center_frequencies, self._sample_rate, self._order)

    def set_order(self, order: int):
        self._order = order
        self._sos = self._get_octave_filters(self._center_frequencies, self._sample_rate, self._order)

    def set_center_frequencies(self, center_freqs: List[int]):
        center_freqs_np = np.asarray(center_freqs)
        assert not np.any(center_freqs_np < 0) and not np.any(center_freqs_np > self._sample_rate / 2), \
            'Center Frequencies must be greater than 0 and smaller than fs/2. Exceptions: exactly 0 or fs/2 ' \
            'will give lowpass or highpass bands'
        self._center_frequencies = np.sort(center_freqs_np).tolist()
        self._sos = self._get_octave_filters(center_freqs, self._sample_rate, self._order)

    def get_center_frequencies(self):
        return self._center_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == 'scipy':
            out = self._forward_scipy(x)
        else:
            raise NotImplementedError('No good implementation relying solely on the pytorch backend has been found yet')
        return out

    def get_filterbank_impulse_response(self):
        """Returns the impulse response of the filterbank."""
        impulse = torch.zeros(1, self._sample_rate * 20)
        impulse[0, self._sample_rate] = 1
        response = self.forward(impulse)
        return response

    @staticmethod
    def _get_octave_filters(center_freqs: List, fs: int, order: int = 5) -> List[torch.Tensor]:
        """
        Design octave band filters (butterworth filter).
        Returns a tensor with the SOS (second order sections) representation of the filter
        """
        sos = []
        for band_idx in range(len(center_freqs)):
            center_freq = center_freqs[band_idx]
            if abs(center_freq) < 1e-6:
                # Lowpass band below lowest octave band
                f_cutoff = (1 / np.sqrt(2)) * center_freqs[band_idx + 1]
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='lowpass', analog=False, output='sos')
            elif abs(center_freq - fs / 2) < 1e-6:
                f_cutoff = np.sqrt(2) * center_freqs[band_idx - 1]
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='highpass', analog=False,
                                               output='sos')
            else:
                f_cutoff = center_freq * np.array([1 / np.sqrt(2), np.sqrt(2)])
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='bandpass', analog=False,
                                               output='sos')

            sos.append(torch.from_numpy(this_sos))

        return sos

def rir_onset(rir: torch.Tensor) -> int:
    spectrogram_trans = torchaudio.transforms.Spectrogram(n_fft=64, win_length=64, hop_length=4)
    spectrogram = spectrogram_trans(rir)
    windowed_energy = torch.sum(spectrogram, dim=len(spectrogram.shape)-2)
    delta_energy = windowed_energy[..., 1:] / (windowed_energy[..., 0:-1]+1e-16)
    highest_energy_change_window_idx = torch.argmax(delta_energy)
    onset = int((highest_energy_change_window_idx-2) * 4 + 64)
    return onset

def discard_last_n_percent(edc: torch.Tensor, n_percent: float) -> torch.Tensor:
    # Discard last n%
    last_id = int(np.round((1 - n_percent / 100) * edc.shape[-1]))
    out = edc[..., 0:last_id]

    return out

def _discard_below(edc: torch.Tensor, threshold_val: float) -> torch.Tensor:
    # set all values below minimum to 0
    out = edc.detach().clone()
    out[out < threshold_val] = 0

    out = _discard_trailing_zeros(out)
    return out

def _discard_trailing_zeros(rir: torch.Tensor) -> torch.Tensor:
    # find first non-zero element from back
    last_above_thres = rir.shape[-1] - torch.argmax((rir.flip(-1) != 0).squeeze().int())

    # discard from that sample onwards
    out = rir[..., :last_above_thres]
    return out

def schroeder(self, rir: torch.Tensor, analyse_full_rir=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Computes the reverse integral (Schroeder method) of a RIR"""
    if not analyse_full_rir:
        onset = rir_onset(rir)
        rir = rir[..., onset:]

    out = _discard_trailing_zeros(rir)

    # Filter
    out = self.filterbank(out)

    # Remove filtering artefacts (last 5 percentile)
    out = discard_last_n_percent(out, 0.5)

    # Backwards integral
    out = torch.flip(out, [2])
    out = (1 / out.shape[2]) * torch.cumsum(out ** 2, 2)
    out = torch.flip(out, [2])

    # Normalize to 1
    norm_vals = torch.max(out, dim=-1, keepdim=True).values  # per channel
    out = out / norm_vals

    return out, norm_vals.squeeze(2)

def get_reverberation_time(rir: torch.Tensor, fs: int, plot: bool = False):
    # polyfit with torch
    #https://stackoverflow.com/questions/70701476/how-to-code-pytorch-to-fit-a-different-polynomal-to-every-column-row-in-an-image
    
    if rir.dtype == torch.float32:
        warnings.warn('WARNING: Operating on float32 tensors can lead to numerical accuracy issues.')

    if len(rir.shape) > 1:
        raise ValueError('Acoustic analysis only supports 1D tensors. So no batches or channels.')

    # EDC
    rir /= rir.abs().max()
    edc = rir ** 2
    edc = torch.flip(edc, [-1])
    edc = (1 / edc.shape[-1]) * torch.cumsum(edc, -1)
    edc = torch.flip(edc, [-1])

    # EDC in dB
    div = edc.abs().max()
    decay = 10 * torch.log10((edc + eps) / div)

    # Set ranges for T30, T20, etc
    tmp = decay < -5
    ids = tmp.nonzero()
    start_t30 = ids[0]
    tmp = decay < -35
    ids = tmp.nonzero()
    end_t60 = ids[0]

    # fit
    target = decay[start_t30:end_t60]
    x = np.linspace(start_t30, end_t60, end_t60 - start_t30)[:, 0]
    a, b = np.polyfit(x, target, 1)
    t60 = ((-60 - b) / a) / fs  # estimate t60 based on trend
    x = np.linspace(0, decay.shape[-1], decay.shape[-1] - 0)
    trend = a * x + b

    # Set ranges for T30, T20, etc
    tmp = decay < -5
    ids = tmp.nonzero()
    start_t20 = ids[0]
    tmp = decay < -25
    ids = tmp.nonzero()
    end_t20 = ids[0]

    # fit
    target = decay[start_t20:end_t20]
    x = np.linspace(start_t20, end_t20, end_t20 - start_t20)[:, 0]
    a, b = np.polyfit(x, target, 1)
    t20 = ((-60 - b) / a) / fs  # estimate t60 based on trend
    x = np.linspace(0, decay.shape[-1], decay.shape[-1] - 0)
    trend_t20 = a * x + b

    # Set ranges for T30, T20, etc
    tmp = decay <= 0.0
    ids = tmp.nonzero(as_tuple=False)
    start_edt = ids[0]
    tmp = decay < -10
    ids = tmp.nonzero(as_tuple=False)
    end_edt = ids[0]

    # fit
    target = decay[start_edt:end_edt]
    x = np.linspace(start_edt, end_edt, end_edt - start_edt)[:, 0]
    a, b = np.polyfit(x, target, 1)
    edt = ((-60 - b) / a) / fs  # estimate t60 based on trend
    x = np.linspace(0, decay.shape[-1], decay.shape[-1] - 0)
    trend_edt = a * x + b

    h2 = rir ** 2
    # DRR
    # This is following eq 1, of:
    # Emmanouilidou, Dimitra & Gamper, Hannes. (2019). The effect of room acoustics on audio event classification.
    ndir = torch.argmax(rir.abs())
    nw = int(2.5 * fs / 1000)  # DRR at 2.5 milliseconds
    window = [np.maximum(0, ndir - nw), ndir + nw]
    drr = 10 * torch.log10(
        torch.sum(h2[..., window[0]: window[1]]) / torch.sum(h2[..., window[1]::]))

    # C50
    # Eq 2 of
    # Emmanouilidou, Dimitra & Gamper, Hannes. (2019). The effect of room acoustics on audio event classification.
    drops = torch.diff(rir, dim=-1)
    n0 = torch.argmax(drops.abs())  # largest drop
    n50 = n0 + int(50 * fs/1000)
    C50 = 10 * torch.log10(
        torch.sum(h2[..., n0:n50]) / torch.sum(h2[..., n50::]))

    # D50 (in %, not dB)
    # Using same framework as C50
    drops = torch.diff(rir, dim=-1)
    n0 = torch.argmax(drops.abs())  # largest drop
    D50 = torch.sum(h2[..., n0:n50]) / torch.sum(h2[..., n0::])

    # Center time
    idx_start = torch.argmax(rir.abs())
    sub_edc = h2[...,  idx_start::]
    t = torch.linspace(0, (sub_edc.shape[-1] - 1) / fs, sub_edc.shape[-1])
    top = torch.einsum('t, t -> t', [t, sub_edc])
    center_time = torch.sum(top) / torch.sum(sub_edc)

    # Curvature
    #     (T30/T20 - 1) expressed as a percentage, providing an indication of how the slope of the decay curve is changing.
    #     Values from 0 to 5% are typical, higher than 10% is suspicious and may indicate that the room has a two-stage
    #     decay curve. If curvature is negative the results should be treated with caution as they may be in error.
    curvature = t60 / t20 - 1
    if plot:
        print(f'T60 : {t60} [s]')
        print(f'T20 : {t20} [s]')
        print(f'EDT : {edt} [s]')
        print(f'DRR : {drr} [dB]')
        print(f'C50 : {C50} [dB]')
        print(f'D50 : {D50*100} [%]')
        print(f'Curvature : {curvature*100} [%]')
        print(f'Center_time : {center_time} [s]')

        import matplotlib.pyplot as plt
        plt.figure()
        t = torch.linspace(0, (rir.shape[-1] - 1) / fs, rir.shape[-1])
        tmp = 10 * torch.log10(rir**2)
        plt.plot(t, trend, 'k--', linewidth=2, label='fit')
        plt.plot(t, trend_edt, 'm-.', linewidth=2, label='fit-edt')
        plt.plot(t, trend_t20, 'y-.', linewidth=2, label='fit-t20')
        plt.plot(t, tmp.numpy(), 'r', alpha=0.5, label='h^2')
        plt.plot(t, decay, linewidth=2, label='EDC')
        plt.ylim([-80, 10])
        plt.xlabel('Time [s]')
        plt.legend()
        plt.tight_layout()
        plt.show()

    results = {'T60': round(t60, ndigits=2),
               'T20': round(t60, ndigits=2),
               'EDT': round(edt, ndigits=2),
               'DRR': torch.round(drr, decimals=2).item(),
               'C50': torch.round(C50, decimals=2).item(),
               'D50': torch.round(D50, decimals=2).item(),
               'curvature': round(curvature, ndigits=2),
               'ctime:': torch.round(center_time, decimals=4).item()}
    return results

def get_rt_ranges(rt='t30'):
    """ Returns the ranges and factor to compute the RT """
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0
    else:
        print(rt)  # error
    return init, end, factor

def get_metrics(rir: torch.Tensor, fs: int, bands: List[int] = [63, 125, 250, 500, 1000, 2000, 4000, 8000], 
                remove_direct_sound: bool = False, plot: bool = False, plot_band: str = '1000', return_EDC=False):
    """ Computes a standard set of metrics to evaluate impulse responses. The metrics include:
        - RT (reverberation time) [s]:
            - EDT
            - T10
            - T20
            - T30
        - Curvature : T30 / T20 - 1 , in [%]
        - Center time [s]
        - C50 [dB]
        - DRR [dB]
        - D50 [%]
        - EDC [dB] - Full early decay curve

        For the RT metrics, we can optionally remove the direct sound. This finds the largest peak, and takes a window
        of 20 ms around it.

        # polyfit with torch
        https://stackoverflow.com/questions/70701476/how-to-code-pytorch-to-fit-a-different-polynomal-to-every-column-row-in-an-image
    
    """
    if rir.dtype == torch.float32:
        warnings.warn('WARNING: Operating on float32 tensors can lead to numerical accuracy issues.')

    if len(rir.shape) > 1:
        raise ValueError('Acoustic analysis only supports 1D tensors. So no batches or channels.')

    rt = ['edt', 't10', 't20', 't30']
    metrics_labels = ['drr', 'c50', 'd50', 'curvature', 'centertime']
    win_len = 0.02  # In seconds, to remove direct sound
    id_direct = torch.argmax(rir)
    id_direct += int(win_len * fs)
    
    metrics = {}
    for this_rt in rt:
        metrics[this_rt] = []
    for met in metrics_labels:
        metrics[met] = []

    if remove_direct_sound:
        rir_rt = rir[..., id_direct::].detach().clone()
    else:
        rir_rt = rir

    # Filter by bands
    filterbank = FilterByOctaves(order=10, center_frequencies=bands)
    filtered_rir = filterbank(rir)
    filtered_rir_rt = filterbank(rir_rt)

    ########print(filtered_rir.shape)
    trends =[]
    all_edc_db = []

    trends_d = {}
    all_edc_db_d = {}
    rirs_plot = {}
    
    for ii, band in enumerate(bands): 
        this_rir = filtered_rir[..., ii, :]
        this_rir_rt = filtered_rir_rt[..., ii, :]
        
        # EDC
        this_rir_rt /= this_rir_rt.abs().max()
        rirs_plot[f'{band}'] = this_rir_rt.clone()  # this plots h^2 up to 0 dB
        edc = this_rir_rt ** 2
        edc = torch.flip(edc, [-1])
        edc = (1 / edc.shape[-1]) * torch.cumsum(edc, -1)
        edc = torch.flip(edc, [-1])
    
        # EDC in dB
        div = edc.abs().max()
        edc_db = 10 * torch.log10((edc + eps) / div)
        all_edc_db.append(edc_db)
        all_edc_db_d[f'{band}'] = edc_db
    
        # Get ranges for RT calculation
        for this_rt in rt:
            init, end, _ = get_rt_ranges(rt=this_rt)
            
            tmp = edc_db < init
            ids = tmp.nonzero()
            start_idx = ids[0]
            tmp = edc_db < end
            ids = tmp.nonzero()
            end_idx = ids[0] if len(ids) > 0 else len(edc_db)
    
            # fit
            target = edc_db[start_idx:end_idx]
            x = np.linspace(start_idx, end_idx, end_idx - start_idx)[:, 0]
            a, b = np.polyfit(x, target, 1)
            t60 = ((-60 - b) / a) / fs  # estimate t60 based on trend
            x = torch.linspace(0, edc_db.shape[-1], edc_db.shape[-1] - 0)
            trend = a * x #+ b

            metrics[this_rt].append(round(t60, ndigits=2))
            trends.append(trend)
            trends_d[f'{band}_{this_rt}'] = trend

        h2 = this_rir ** 2
        # DRR
        # This is following eq 1, of:
        # Emmanouilidou, Dimitra & Gamper, Hannes. (2019). The effect of room acoustics on audio event classification.
        ndir = torch.argmax(this_rir.abs())
        nw = int(2.5 * fs / 1000)  # DRR at 2.5 milliseconds
        window = [np.maximum(0, ndir - nw), ndir + nw]
        drr = 10 * torch.log10(
            torch.sum(h2[..., window[0]: window[1]]) / (torch.sum(h2[..., window[1]::]) + 1e-10))
        metrics['drr'].append(torch.round(drr, decimals=2).item())
    
        # C50
        # Eq 2 of
        # Emmanouilidou, Dimitra & Gamper, Hannes. (2019). The effect of room acoustics on audio event classification.
        drops = torch.diff(this_rir, dim=-1)
        n0 = torch.argmax(drops.abs())  # largest drop
        n50 = n0 + int(50 * fs/1000)
        C50 = 10 * torch.log10(
            torch.sum(h2[..., n0:n50]) / (torch.sum(h2[..., n50::]) + 1e-10))
        metrics['c50'].append(torch.round(C50, decimals=2).item())
    
        # D50 (in %, not dB)
        # Using same framework as C50
        drops = torch.diff(this_rir, dim=-1)
        n0 = torch.argmax(drops.abs())  # largest drop
        D50 = torch.sum(h2[..., n0:n50]) / torch.sum(h2[..., n0::])
        metrics['d50'].append(torch.round(D50, decimals=2).item())
    
        # Center time
        idx_start = torch.argmax(this_rir.abs())
        sub_edc = h2[...,  idx_start::]
        t = torch.linspace(0, (sub_edc.shape[-1] - 1) / fs, sub_edc.shape[-1])
        top = torch.einsum('t, t -> t', [t, sub_edc])
        center_time = torch.sum(top) / torch.sum(sub_edc)
        metrics['centertime'].append(torch.round(center_time, decimals=2).item())

        # Curvature
        #     (T30/T20 - 1) expressed as a percentage, providing an indication of how the slope of the decay curve is changing.
        #     Values from 0 to 5% are typical, higher than 10% is suspicious and may indicate that the room has a two-stage
        #     decay curve. If curvature is negative the results should be treated with caution as they may be in error.
        curvature = metrics['t30'][ii] / metrics['t20'][ii] - 1
        metrics['curvature'].append(round(curvature, ndigits=2))

        # Append EDC by bands to compute error in EDCs, this can be very memory intensive
        if return_EDC:
            metrics['edc'] = torch.stack(all_edc_db, dim=0)
    
    if plot:
        for k, v in metrics.items():
            print(k, v)

        import matplotlib.pyplot as plt
        plt.figure()

        tmp = 10 * torch.log10(rirs_plot[plot_band]**2)
        t = torch.linspace(0, (tmp.shape[-1] - 1) / fs, tmp.shape[-1])
        styles = ['k--', 'm-.', 'y-.', 'b-.']

        # Find the edc and trends of the desired freq band
        trends_plot, edc_db_plot = [], None
        for k,v in trends_d.items():
            a,b = k.split('_')
            if a == plot_band:
                trends_plot.append(v)
        for k,v in all_edc_db_d.items():
            if k == plot_band:
                edc_db_plot = v
        for trend, style, this_rt in zip(trends_plot, styles, rt):
            #print(trend.shape)
            #print(style)
            plt.plot(t, trend, style, linewidth=2, label=this_rt)
        plt.plot(t, tmp.numpy(), 'r', alpha=0.5, label='h^2')  
        plt.plot(t, edc_db_plot, linewidth=2, label='EDC')
        plt.ylim([-80, 10])
        plt.xlabel('Time [s]')
        plt.ylabel(f'{plot_band} Hz')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #print(len(trends))
        #print(trends_d)

    return metrics

def get_metrics_decayfitnet(rir: torch.Tensor, fs: int, bands: List[int] = [63, 125, 250, 500, 1000, 2000, 4000, 8000], 
                remove_direct_sound: bool = False, plot: bool = False, plot_band: str = '1000', 
                fadeout_length: float = 0.5, return_EDC=False, device='cpu'):
    """ Computes a standard set of metrics to evaluate impulse responses. The metrics include:
        - RT (reverberation time) [s]:
            - EDT
            - T10
            - T20
            - T30
        - Curvature : T30 / T20 - 1 , in [%]
        - Center time [s]
        - C50 [dB]
        - DRR [dB]
        - D50 [%]
        - EDC [dB] - Full early decay curve

        For the RT metrics, we can optionally remove the direct sound. This finds the largest peak, and takes a window
        of 30 ms around it.

        # polyfit with torch
        https://stackoverflow.com/questions/70701476/how-to-code-pytorch-to-fit-a-different-polynomal-to-every-column-row-in-an-image
    
    """
    # Moved import here instead of top of the file, to avoid problems when running in the old triton
    # As it throws an error thet GLIBC is not isntalled
    from DecayFitNet.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
    from DecayFitNet.toolbox.utils import calc_mse
    from DecayFitNet.toolbox.core import discard_last_n_percent, decay_model, PreprocessRIR

    if rir.dtype == torch.float32:
        warnings.warn('WARNING: Operating on float32 tensors can lead to numerical accuracy issues.')

    if len(rir.shape) > 1:
        raise ValueError('Acoustic analysis only supports 1D tensors. So no batches or channels.')

    # Config
    rt = ['edt', 't10', 't20', 't30']
    metrics_labels = ['drr', 'c50', 'd50', 'curvature', 'centertime']
    metrics_labels_decayfitnet = ['t0', 't1', 't2', 'a0', 'a1', 'a2', 'n', 'n_slopes']  # time in seconds, amplitude in linear, single noise term
    win_len = 0.03  # In seconds, to remove direct sound
    id_direct = torch.argmax(rir)
    id_direct += int(win_len * fs)

    # Other way to get the direct sound, seems much better
    drops = torch.diff(rir, dim=-1)
    id_direct = torch.argmax(drops.abs())  # largest drop
    id_direct += int(win_len * fs)

    # For decayfitnet
    n_slopes = 0
    detect_onset = True
    fadeout_length = fadeout_length  # in seconds, to remove trailing zeros
    # This is quite important to get a good estimate
    
    metrics = {}
    for this_rt in rt:
        metrics[this_rt] = []
    for met in metrics_labels:
        metrics[met] = []
    for met in metrics_labels_decayfitnet:
        metrics[met] = []

    if remove_direct_sound:
        rir_rt = rir[..., id_direct::].detach().clone()
    else:
        rir_rt = rir
    rir_decayfitnet = rir_rt.clone()
    if len(rir_decayfitnet.shape) < 2:
        rir_decayfitnet = rir_decayfitnet[None, ...]

    # Filter by bands
    filterbank = FilterByOctaves(order=10, center_frequencies=bands)
    filtered_rir = filterbank(rir)
    filtered_rir_rt = filterbank(rir_rt)

    ########print(filtered_rir.shape)
    trends =[]
    all_edc_db = []

    trends_d = {}
    all_edc_db_d = {}
    rirs_plot = {}
    
    for ii, band in enumerate(bands): 
        this_rir = filtered_rir[..., ii, :]
        this_rir_rt = filtered_rir_rt[..., ii, :]
        
        # EDC
        this_rir_rt /= this_rir_rt.abs().max()
        rirs_plot[f'{band}'] = this_rir_rt.clone()  # this plots h^2 up to 0 dB
        edc = this_rir_rt ** 2
        edc = torch.flip(edc, [-1])
        edc = (1 / edc.shape[-1]) * torch.cumsum(edc, -1)
        edc = torch.flip(edc, [-1])
    
        # EDC in dB
        div = edc.abs().max()
        edc_db = 10 * torch.log10((edc + eps) / div)
        all_edc_db.append(edc_db)
        all_edc_db_d[f'{band}'] = edc_db
    
        # Get ranges for RT calculation
        for this_rt in rt:
            init, end, _ = get_rt_ranges(rt=this_rt)
            
            tmp = edc_db < init
            ids = tmp.nonzero()
            start_idx = ids[0]
            tmp = edc_db < end
            ids = tmp.nonzero()
            end_idx = ids[0]
    
            # fit
            target = edc_db[start_idx:end_idx]
            x = np.linspace(start_idx, end_idx, end_idx - start_idx)[:, 0]
            a, b = np.polyfit(x, target, 1)
            t60 = ((-60 - b) / a) / fs  # estimate t60 based on trend
            x = torch.linspace(0, edc_db.shape[-1], edc_db.shape[-1] - 0)
            trend = a * x #+ b

            metrics[this_rt].append(round(t60, ndigits=2))
            trends.append(trend)
            trends_d[f'{band}_{this_rt}'] = trend

        h2 = this_rir ** 2
        # DRR
        # This is following eq 1, of:
        # Emmanouilidou, Dimitra & Gamper, Hannes. (2019). The effect of room acoustics on audio event classification.
        ndir = torch.argmax(this_rir.abs())
        nw = int(2.5 * fs / 1000)  # DRR at 2.5 milliseconds
        window = [np.maximum(0, ndir - nw), ndir + nw]
        drr = 10 * torch.log10(
            torch.sum(h2[..., window[0]: window[1]]) / (torch.sum(h2[..., window[1]::]) + 1e-10))
        metrics['drr'].append(torch.round(drr, decimals=2).item())
    
        # C50
        # Eq 2 of
        # Emmanouilidou, Dimitra & Gamper, Hannes. (2019). The effect of room acoustics on audio event classification.
        drops = torch.diff(this_rir, dim=-1)
        n0 = torch.argmax(drops.abs())  # largest drop
        n50 = n0 + int(50 * fs/1000)
        C50 = 10 * torch.log10(
            torch.sum(h2[..., n0:n50]) / (torch.sum(h2[..., n50::]) + 1e-10))
        metrics['c50'].append(torch.round(C50, decimals=2).item())
    
        # D50 (in %, not dB)
        # Using same framework as C50
        drops = torch.diff(this_rir, dim=-1)
        n0 = torch.argmax(drops.abs())  # largest drop
        D50 = torch.sum(h2[..., n0:n50]) / torch.sum(h2[..., n0::])
        metrics['d50'].append(torch.round(D50, decimals=2).item())
    
        # Center time
        idx_start = torch.argmax(this_rir.abs())
        sub_edc = h2[...,  idx_start::]
        t = torch.linspace(0, (sub_edc.shape[-1] - 1) / fs, sub_edc.shape[-1])
        top = torch.einsum('t, t -> t', [t, sub_edc])
        center_time = torch.sum(top) / torch.sum(sub_edc)
        metrics['centertime'].append(torch.round(center_time, decimals=2).item())

        # Curvature
        #     (T30/T20 - 1) expressed as a percentage, providing an indication of how the slope of the decay curve is changing.
        #     Values from 0 to 5% are typical, higher than 10% is suspicious and may indicate that the room has a two-stage
        #     decay curve. If curvature is negative the results should be treated with caution as they may be in error.
        curvature = metrics['t30'][ii] / metrics['t20'][ii] - 1
        metrics['curvature'].append(round(curvature, ndigits=2))

        # Append EDC by bands to compute error in EDCs, this can be very memory intensive
        if return_EDC:
            metrics['edc'] = torch.stack(all_edc_db, dim=0)
    
    # ===============================================================================
    # DEcayFitNet processing
    if plot:
        rir_preprocessing = PreprocessRIR(sample_rate=fs, filter_frequencies=bands)
        # Schroeder integration, analyse_full_rir: if RIR onset should be detected, set this to False
        true_edc, __ = rir_preprocessing.schroeder(rir_decayfitnet, analyse_full_rir=~detect_onset)
        time_axis = (torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) / fs)
        # Permute into [n_bands, n_batches, n_samples]
        true_edc = true_edc.permute(1, 0, 2)

    # Delete potential fade-out windows
    #print(rir_decayfitnet.shape)
    if fadeout_length > 0:
        rir_decayfitnet = rir_decayfitnet[:, 0:round(-fadeout_length * fs)].to(device)
    #print(rir_decayfitnet.shape)
    
    # Analyze with DecayFitNet
    decayfitnet = DecayFitNetToolbox(n_slopes=n_slopes, sample_rate=fs, filter_frequencies=bands, device=device)
    estimated_parameters_decayfitnet, norm_vals_decayfitnet = decayfitnet.estimate_parameters(rir_decayfitnet, analyse_full_rir=~detect_onset)
    #print(estimated_parameters_decayfitnet[2].shape)
    for ii in range(3): 
        metrics[f't{ii}'] = np.round(estimated_parameters_decayfitnet[0][:,ii], decimals=3).tolist()
        metrics[f'a{ii}'] = np.round(estimated_parameters_decayfitnet[1][:,ii], decimals=5).tolist()
    metrics[f'n'] = estimated_parameters_decayfitnet[2][:].squeeze().tolist()

    # Count active slopes
    a = estimated_parameters_decayfitnet[0] > 0
    metrics[f'n_slopes'] = torch.from_numpy((estimated_parameters_decayfitnet[0]) > 0).sum(dim=-1).tolist()

    # ===============================================================================
    # Plotting
    if plot:
        for k, v in metrics.items():
            print(k, v)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,2, figsize=(18,6))
        axs = axs.flatten()

        tmp = 10 * torch.log10(rirs_plot[plot_band]**2)
        t = torch.linspace(0, (tmp.shape[-1] - 1) / fs, tmp.shape[-1])
        styles = ['k--', 'm-.', 'y-.', 'b-.']

        # Find the edc and trends of the desired freq band
        trends_plot, edc_db_plot = [], None
        for k,v in trends_d.items():
            a,b = k.split('_')
            if a == plot_band:
                trends_plot.append(v)
        for k,v in all_edc_db_d.items():
            if k == plot_band:
                edc_db_plot = v
        for trend, style, this_rt in zip(trends_plot, styles, rt):
            #print(trend.shape)
            #print(style)
            axs[0].plot(t, trend, style, linewidth=2, label=this_rt)
        axs[0].plot(t, tmp.numpy(), 'r', alpha=0.5, label='h^2')  
        axs[0].plot(t, edc_db_plot, linewidth=2, label='EDC')
        axs[0].set_ylim([-80, 10])
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel(f'{plot_band} Hz')
        axs[0].legend()

        #print(len(trends))
        #print(trends_d)

    # Decayfitnet plot
        fitted_edc_decayfitnet = decay_model(torch.from_numpy(estimated_parameters_decayfitnet[0]),
                                        torch.from_numpy(estimated_parameters_decayfitnet[1]),
                                        torch.from_numpy(estimated_parameters_decayfitnet[2]),
                                        time_axis=time_axis,
                                        compensate_uli=True,
                                        backend='torch')

        if True:
            # Discard last 5% for MSE evaluation
            true_edc = discard_last_n_percent(true_edc, 5)
            fitted_edc_decayfitnet = discard_last_n_percent(fitted_edc_decayfitnet, 5)
            time_axis_excl5 = time_axis[0:round(0.95 * len(time_axis))]  # discard last 5 percent of plot time axis
        else:
            time_axis_excl5 = time_axis

        # Calculate MSE between true EDC and fitted EDC
        mse_per_frequencyband = calc_mse(true_edc, fitted_edc_decayfitnet)

        # Plot
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for band_idx in range(true_edc.shape[0]):
            axs[1].plot(time_axis_excl5, 10 * torch.log10(true_edc[band_idx, 0, :].squeeze()),
                    colors[band_idx], label='Measured EDC, {} Hz'.format(bands[band_idx]))
            axs[1].plot(time_axis_excl5, 10 * torch.log10(fitted_edc_decayfitnet[band_idx, 0, :].squeeze()),
                    colors[band_idx] + '--', label='DecayFitNet fit, {} Hz'.format(bands[band_idx]))
        tmp = 10 * torch.log10(rir_decayfitnet[0]**2)
        print(f'tmp.shape {tmp.shape}')
        print(f'time_axis_excl5.shape {time_axis_excl5.shape}')
        #axs[1].plot(time_axis_excl5, tmp.numpy(), 'r', alpha=0.5, label='h^2')  

        axs[1].set_xlabel('time [s]')
        axs[1].set_ylabel('energy [dB]')
        axs[1].set_ylim([-80, 10])
        #axs[1].subplots_adjust(right=0.6)
        axs[1].legend(loc='upper right', bbox_to_anchor=(1.8, 1))
        axs[1].set_title('DecayFitNet')
        
        plt.tight_layout()
        plt.show() 


    return metrics

def get_beamforming_matrix(w_pattern: str = 'maxre', sph_order: int = 2, polygon_sides=6, 
                           rotate_alternative_convention=False, do_plot=False):
    # Define a grid of azimuth values in 
    tmp = torch.linspace(0, 360, steps=polygon_sides)  # 6 steps = polygon_sides directions
    grid = torch.zeros(tmp.shape[-1], 2)
    grid[:, 0] = tmp
    grid = torch.deg2rad(grid)
    grid[:, 1] = utils.ele2colat(grid[:, 1])

    # Define pattern for W
    if w_pattern.lower() == "cardioid":
        c_n = spa.sph.cardioid_modal_weights(sph_order)
    elif w_pattern.lower() == "hypercardioid":  # max-DI
        c_n = spa.sph.hypercardioid_modal_weights(sph_order)
    elif w_pattern.lower() == "maxre":
        c_n = spa.sph.maxre_modal_weights(sph_order, True)  # works with amplitude compensation and without!
    elif w_pattern.lower() == "butter":
        c_n = spa.butterworth_modal_weights(sph_order, 1, 5)  # not that good 
        
    # Compute Y, W, and G, using a spatial filterbank with the selected directivity pattern
    if rotate_alternative_convention:
        [W, Y] = spa.sph.design_sph_filterbank(sph_order, grid[:,0] -3*np.pi/2, grid[:,1], c_n, 'real', mode='perfect')  # For MRAS scenes, different convention
    else:
        [W, Y] = spa.sph.design_sph_filterbank(sph_order, grid[:,0], grid[:,1], c_n, 'real', mode='perfect' )
    Y = torch.from_numpy(Y).double()
    W = torch.from_numpy(W).double()
    G = torch.eye(W.shape[0]).double()

    # Plot the patter if needed
    if do_plot:
        print(f'Matrix Y of shape: {Y.shape}')
        print(f'Matrix G of shape: {G.shape}')
        print(f'Matrix W of shape: {W.shape}')

        w_nm = spa.sph.repeat_per_order(c_n) * spa.sph.sh_matrix(sph_order, np.pi/4, np.pi/4, 'real')
        spa.plot.sh_coeffs(w_nm)    

    return W, grid[:,0]

def mean_absolute_proportional_error(y_hat: torch.Tensor, y: torch.Tensor, reduction=True):
    eps = 1e-12
    tmp = y_hat - y
    tmp = tmp / (y + eps)
    tmp = tmp.abs()

    if reduction:
        tmp = tmp.mean()
    return tmp

def mean_absolute_proportional_error_no_zero(y_hat: torch.Tensor, y: torch.Tensor, reduction=True):
    """ Same as MAPE, but we ignore those where the GT is 0.0"""
    eps = 1e-12
    non_zeros = y != 0.0

    if non_zeros.sum() > 0:
        tmp = torch.zeros_like(y_hat)
        tmp[non_zeros] = y_hat[non_zeros] - y[non_zeros]
        tmp[non_zeros] = tmp[non_zeros] / (y[non_zeros] + eps)
        tmp = tmp.abs()
        #tmp = tmp[non_zeros]

        if reduction:
            denom = non_zeros.sum()
            tmp /= denom
        return tmp
    else:
        return None

def mean_absolute_proportional_error_threshold(y_hat: torch.Tensor, y: torch.Tensor, threshold=0.1, reduction=True):
    """ Same as MAPE, but here we apply a threshold to those values where the GT is 0.0
    For example, if the is 0.0, we set to the threshold = 0.1, so that any prediction up to 0.1, will produce 0.0 error.
    """
    eps = 1e-12
    zeros = y == 0.0

    y[zeros] = threshold
    y_hat[y_hat < threshold] = threshold  

    tmp = y_hat - y
    tmp = tmp / (y + eps)
    tmp = tmp.abs()

    if reduction:
        tmp = tmp.mean()
    return tmp

def mean_absolute_log_error(y_hat: torch.Tensor, y: torch.Tensor, reduction=True):
    eps = 1e-12
    tmp = torch.log((y_hat + eps) / (y + eps))
    tmp = tmp.abs()

    if reduction:
        tmp = tmp.mean()
    return tmp
                    
def error_for_metrics(metrics_estimate: Dict, metrics_target: Dict):
    """ Computes the error (distance) between two sets of metrics.
        The distance funxtions used are:
        - RT (reverberation time) [s]:
            - EDT - MAPE
            - T10 - MAPE
            - T20 - MAPE
            - T30 - MAPE
        - Curvature : T30 / T20 - 1 , in [%] - MAE
        - Center time [s] - MARE
        - C50 [dB] - MAE
        - DRR [dB] - MAE
        - D50 [%] - MAPE
        - EDC [dB] - MAE

        MARE is the mean absolute relative error.
        """

    tags = ['edt', 't10', 't20', 't30', 'curvature', 'centertime', 'c50', 'drr', 'd50', 'edc']
    dist = ['mape', 'mape', 'mape', 'mape', 'l1', 'mape', 'l1', 'l1', 'mape', 'l1']
    #dist = ['male', 'male', 'male', 'male', 'l1', 'male', 'l1', 'l1', 'male', 'l1']  I am still not convinced if this makes 100% sense

    error = {}
    for ii, tag in enumerate(tags):
        if tag not in metrics_target or tag not in metrics_estimate:
            continue
        target = torch.tensor(metrics_target[tag])
        estimate = torch.tensor(metrics_estimate[tag])

        if dist[ii] == 'mape':
            tmp = mean_absolute_proportional_error(estimate, target)
        elif dist[ii] == 'mape_nozero':
            tmp = mean_absolute_proportional_error_no_zero(estimate, target)
        elif dist[ii] == 'male':
            tmp = mean_absolute_log_error(estimate, target)
        elif dist[ii] == 'l1':
            fn = torch.nn.L1Loss()
            tmp = fn(estimate, target)
        else:
            raise ValueError(f'ERROR: Unrecognized distance fuction {dist[ii]}')
        error[tag] = tmp
        
    return error

def error_mstft(estimate: torch.Tensor, target: torch.Tensor, weights_cSTFT=[1.0, 0.0, 0.0, 0.0]) -> torch.Tensor:
    """ Computes the Multi STFT using the real, imag, and maginute parts """
    loss_fn = losses.MultiComplexSTFTLoss(device=estimate.device, weights_cSTFT=weights_cSTFT)
    error = loss_fn(estimate, target)
    return error

def error_early_reflections_with_peakfinding(estimate: torch.Tensor, target: torch.Tensor, fs: int, early_t: float = 0.05) -> torch.Tensor:
    """ Computes the early reflections L1 loss, lookign at the first 50 milliseconds in time domain centered aroudn the max abs value .
    Here, the early reflection window is centered around the largest peak (which should be the direct sound) of the target RIR, for the 
    first channel (assuming its the omni channel).
    Args:
        early_t_ms: float - time for early relfection in seconds"""

    loss_fn = torch.nn.L1Loss().to(estimate.device)
    
    #peak_estimate = estimate.abs().argmax().item()
    peak_target = target[0, :].abs().argmax().item()
    #start_estimate = np.maximum(0, peak_estimate - int(early_t/2 * fs))
    start_target = np.maximum(0, peak_target - int(early_t/2 * fs))
    
    error = loss_fn(estimate[:, start_target:start_target + int(early_t/2 * fs)], target[:, start_target:start_target + int(early_t/2 * fs)])
    return error

def error_early_reflections(estimate: torch.Tensor, target: torch.Tensor, fs: int, early_t: float = 0.05) -> torch.Tensor:
    """ Computes the early reflections L1 loss, lookign at the first 50 milliseconds in time domain centered aroudn the max abs value
    Args:
        early_t_ms: float - time for early relfection in seconds"""
    loss_fn = torch.nn.L1Loss().to(estimate.device)
    
    error = loss_fn(estimate[:, 0:int(early_t * fs)], target[:, 0:int(early_t * fs)])
    return error

def error_peak_reflections(estimate: torch.Tensor, target: torch.Tensor, fs: int, early_t: float = 0.05, topk: int = 20, distance: str = 'l1') -> torch.Tensor:
    """ Computes the error for the topk early reflections (peaks). The error is based on timing (samples), and not on magnitude.
    Args:
        early_t_ms: float - time for early relfection in seconds
        topk: int - How many peaks to find 
        distance: {'l1', 'l2'} - distance function """

    estimate_values, estimate_indices = torch.topk(estimate[:, 0:int(early_t * fs)]**2, k=topk, dim=-1)
    target_values, target_indices = torch.topk(target[:, 0:int(early_t * fs)]**2, k=topk, dim=-1)

    if distance == 'l1':
        loss_fn = torch.nn.L1Loss().to(estimate.device)
    else:
        loss_fn = torch.nn.MSELoss().to(estimate.device)
    error = loss_fn(estimate_indices.to(torch.float32), target_indices.to(torch.float32))

    if distance == 'l2':
        error = torch.sqrt(error)
    return error / fs  # Error in seconds

def test_filterbank():
    import plots
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    fs = 48000
    sig = torch.randn((1, fs))
    sig /= sig.abs().max()
    filterbank = FilterByOctaves(order=10)
    resp = filterbank.get_filterbank_impulse_response()
    plots.plot_fft(resp[0, :, :], fs=fs, title=f'Filter response')
    for ii in range(resp.shape[-2]):
        plots.plot_fft(resp[:, ii, :], fs=fs, title=f'Filter response {filterbank.get_center_frequencies()[ii]}')

    tic = time.time()
    out = filterbank(sig)
    toc = time.time()

    plots.plot_fft(sig, fs=fs, title='Sig')
    for ii in range(out.shape[-2]):
        plots.plot_fft(out[:, ii, :], fs=fs, title=f'Filtered {filterbank.get_center_frequencies()[ii]}')
    #time.strftime()
    print(f'Elapsed time = {(toc - tic)}')

def test_reverberation_time():
    import plots
    import math
    from scipy import signal
    from random import sample
    from utils import seed_everything

    seed_everything(1111, 'balanced')
    fs = 48000
    t = 2.0
    noise_snr = 70
    lamda = 28  # bigger lamba --> shorter RIR, use 7 for about ~ 1.0 t60
    density = 0.5  # how many early reflections are zero
    scale_direct = 3.5  # Improve DRR by scaling the direct part

    # Late reverb
    tsteps = torch.arange(0, t, step=1 / fs)
    late = torch.randn(1, math.floor(fs * t))
    late /= late.abs().max()
    decay = 1 * torch.exp(-lamda * tsteps)
    rir = late * decay  # this is late reverb

    # Simulate early reflections
    idx = torch.arange(0, math.floor(fs * 0.05))  # 50 ms
    idx_early = idx.shape[-1]
    idx_zeros = sample(idx.tolist(), math.floor(density * idx.shape[-1]))
    rir.index_fill_(-1, torch.tensor(idx_zeros), 0.0)
    rir[:, 0:idx_early] = rir[:, 0:idx_early]

    # Set direct
    idx = torch.arange(0, math.floor(fs * 0.0025))  # somewhere in the first 2.5 ms
    idx_direct = sample(idx.tolist(), 1)[0]
    rir[:, idx_direct] = 1

    # Scale the direct path
    idx_wn = math.floor(fs * 0.0025)
    window_size = np.minimum(idx_direct, idx_wn) + idx_wn
    window_fn = torch.from_numpy(signal.windows.hann(window_size))
    window_neg = 1 - window_fn
    he = rir[:, np.maximum(0, idx_direct - idx_wn): idx_direct + idx_wn]
    he = (scale_direct * window_fn * he) + (window_neg * he)
    rir[:, np.maximum(0, idx_direct - idx_wn) : idx_direct + idx_wn] = he



    # Background noise with desired snr
    alpha = torch.sqrt(torch.sqrt(torch.mean(tsteps ** 2)) / 10 ** (noise_snr / 10))  # rms power for the sig
    noise = torch.randn(1, tsteps.shape[-1], dtype=torch.float32) * alpha

    rir = rir + noise
    rir /= rir.abs().max()

    early = rir[:, 0:idx_early]
    plots.plot_waveform(early, sample_rate=fs, title='Early reflections')
    plots.plot_waveform(rir, sample_rate=fs, title='Late reverb')

    #rir = torch.cat([early, rir], dim=-1)
    rir = rir[0, :]
    get_reverberation_time(rir, fs, plot=True)


def test_metrics():
    import plots
    import math
    from scipy import signal
    from random import sample
    from utils import seed_everything

    seed_everything(1111, 'balanced')
    fs = 48000
    t = 2.0
    noise_snr = 70
    lamda = 28  # bigger lamba --> shorter RIR, use 7 for about ~ 1.0 t60
    density = 0.5  # how many early reflections are zero
    scale_direct = 3.5  # Improve DRR by scaling the direct part

    # Late reverb
    tsteps = torch.arange(0, t, step=1 / fs)
    late = torch.randn(1, math.floor(fs * t))
    late /= late.abs().max()
    decay = 1 * torch.exp(-lamda * tsteps)
    rir = late * decay  # this is late reverb

    # Simulate early reflections
    idx = torch.arange(0, math.floor(fs * 0.05))  # 50 ms
    idx_early = idx.shape[-1]
    idx_zeros = sample(idx.tolist(), math.floor(density * idx.shape[-1]))
    rir.index_fill_(-1, torch.tensor(idx_zeros), 0.0)
    rir[:, 0:idx_early] = rir[:, 0:idx_early]

    # Set direct
    idx = torch.arange(0, math.floor(fs * 0.0025))  # somewhere in the first 2.5 ms
    idx_direct = sample(idx.tolist(), 1)[0]
    rir[:, idx_direct] = 1

    # Scale the direct path
    idx_wn = math.floor(fs * 0.0025)
    window_size = np.minimum(idx_direct, idx_wn) + idx_wn
    window_fn = torch.from_numpy(signal.windows.hann(window_size))
    window_neg = 1 - window_fn
    he = rir[:, np.maximum(0, idx_direct - idx_wn): idx_direct + idx_wn]
    he = (scale_direct * window_fn * he) + (window_neg * he)
    rir[:, np.maximum(0, idx_direct - idx_wn) : idx_direct + idx_wn] = he



    # Background noise with desired snr
    alpha = torch.sqrt(torch.sqrt(torch.mean(tsteps ** 2)) / 10 ** (noise_snr / 10))  # rms power for the sig
    noise = torch.randn(1, tsteps.shape[-1], dtype=torch.float32) * alpha

    rir = rir + noise
    rir /= rir.abs().max()

    early = rir[:, 0:idx_early]
    plots.plot_waveform(early, sample_rate=fs, title='Early reflections')
    plots.plot_waveform(rir, sample_rate=fs, title='Late reverb')

    #rir = torch.cat([early, rir], dim=-1)
    rir = rir[0, :]
    metrics = get_metrics(rir, fs, plot=True)
    print(metrics)


def OLD_early_reflections():
    # Simuilate early refelctions
    density = 0.8  # how many early relfections are not zero
    early = (torch.rand(size=(1, math.floor(fs * 0.05))) * 2) - 1  # 50 ms for ealy reflections
    early /= early.abs().max()
    idx = torch.arange(0, early.shape[-1])
    idx_nonzero = sample(idx.tolist(), math.floor(density * early.shape[-1]))
    idx_nonzero = torch.tensor(idx_nonzero)
    zeros = torch.zeros_like(early)
    zeros[:, idx_nonzero] = early[:, idx_nonzero]
    early = zeros
    # Add decay to early relfections
    lamda = 3  # bigger lamba --> shorter RIR
    t = torch.linspace(0, 2, steps=early.shape[-1])
    decay = 1 * torch.exp(-lamda * t)
    early = early * decay



import time
def test_scipy_backend():
    # Create an instance of the class with the 'scipy' backend
    filter = FilterByOctaves(backend='scipy')
    # Create a random tensor to process
    tensor = torch.randn(1, 48000)
    # Measure the execution time of the forward method
    start_time = time.time()
    output = filter.forward(tensor)
    end_time = time.time()
    # Check that the output has the correct shape
    print(output.shape)
    assert output.shape ==  (1, 7, 48000), 'ERROR, wrong shape'
    # Print the execution time
    print(f'Scipy backend execution time: {end_time - start_time} seconds')
    
def test_pytorch_backend():
    # Create an instance of the class with the 'pytorch' backend
    filter = FilterByOctaves(backend='pytorch')
    # Create a random tensor to process
    tensor = torch.randn(1, 48000)
    # Measure the execution time of the forward method
    start_time = time.time()
    output = filter.forward(tensor)
    end_time = time.time()
    print(output.shape)
    # Check that the output has the correct shape
    assert output.shape ==  (1, 7, 48000), 'ERROR, wrong shape'
    # Print the execution time
    print(f'Pytorch backend execution time: {end_time - start_time} seconds')
    



if __name__ == '__main__':
    #test_scipy_backend()
    #test_pytorch_backend()
    #exit(0)
    #test_filterbank()
    #test_reverberation_time()
    test_metrics()

