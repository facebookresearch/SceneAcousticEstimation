# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
import warnings
import ot
import ptwt
import pywt
from typing import List, Union

##
# Useful repos:
# https://github.com/salu133445/dan/blob/master/src/dan/losses.py
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
#
##

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

def discriminator_loss(prediction: torch.Tensor, target_is_real: bool, loss_type: str = 'minmax'):
    target_real = 1.0
    target_fake = 0.0

    if target_is_real:
        target_tensor = torch.tensor(target_real)
    else:
        target_tensor = torch.tensor(target_fake)
    target_tensor = target_tensor.expand_as(prediction).to(prediction.get_device())

    if loss_type == 'minmax':
        fn = nn.BCEWithLogitsLoss().to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'non-sat':
        fn = nn.BCEWithLogitsLoss().to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'ls':
        fn = nn.MSELoss().to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'hinge':
        if target_is_real:
            loss = -torch.mean(torch.min(torch.zeros_like(prediction), prediction - 1))
        else:
            loss = -torch.mean(torch.min(torch.zeros_like(prediction), -prediction - 1))
    elif loss_type == 'wass':
        # NOTE: This is missing the Gradient Penalty, so it does not work very well
        if target_is_real:
            loss = -torch.mean(prediction)
        else:
            loss = torch.mean(prediction)

    return loss

def generator_loss(prediction, loss_type='minmax'):
    eps = 1e-8

    if loss_type == 'minmax':
        fn = nn.BCEWithLogitsLoss().to(prediction.get_device())
        target_real = 1.0
        target_tensor = torch.tensor(target_real)
        target_tensor = target_tensor.expand_as(prediction).to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'non-sat':
        loss = -torch.mean(torch.log(torch.sigmoid(prediction + eps)))
    elif loss_type == 'ls':
        loss = torch.mean(torch.pow(prediction - 1, 2))
    elif loss_type == 'hinge':
        loss = -torch.mean(prediction)
    elif loss_type == 'wass':
        loss = -torch.mean(prediction)

    return loss

class Wasserstein2dLoss(nn.Module):
    """ Computes the 2d Sliced Wassertein distance. 
    This is done by n random 1d projections, and then computing then computing the 1d earth mover distance there.
    We use the POT library here. This is slow because this does not support batches, or gpu, or anything.
    NOTE: Maybe it would be better to define the random projection vectors once and fix them for all data.

    """
    def __init__(self, n_projections=50):
        super(Wasserstein2dLoss, self).__init__()
        self.n_projections = n_projections

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(output.shape) < 3:
            output = output[None, None, ...]
            target = target[None, None, ...]
        elif len(output.shape) < 4:
            output = output[None, ...]
            target = target[None, ...]
        
        if len(output.shape) != 4:
            raise ValueError(f'WRONG shape for tensors, I found {output.shape}')
        
        batch, chan = output.shape[0], output.shape[1]
        distance = []
        for b in range(batch):
            for c in range(chan):
                this_distance = ot.sliced_wasserstein_distance(output[b,c].detach().cpu().numpy(), 
                                                               target[b,c].detach().cpu().numpy())
                                                               #projections=self.n_projections)
                distance.append(this_distance)

        distance = torch.tensor(distance).mean()
        return distance

def bandpass_filter(image, min_radius, max_radius, do_plot=False, apply_windowing=True, do_padding=True, 
                    use_soft_masks=True, trans_width=10):
    ''' Perform a 2D bandpass filter on an image
    
    :param image: A 2D image, in your case, it will be a 2D slice of a single color of an image
    :param min_radius: The minimum frequency to cut off
    :param max_radius: The maximum frequency to cut off 
    '''

    if len(image.shape) > 2:
        raise ValueError("The image should be 2D.")
        
    if do_padding:
        # Create padded image for reducing spectral leakage
        h_pad, w_pad = image.size(0) // 2, image.size(1) // 2
        #image_padded = torch.nn.functional.pad(image, (w_pad, w_pad, h_pad, h_pad))
        image = torch.nn.functional.pad(image[None,...], (w_pad, w_pad, h_pad, h_pad), mode='constant')[0]
        # mode (str) – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

    if apply_windowing:
        # Apply Hann window
        x_win = torch.hann_window(image.size(1), dtype=torch.float, device=image.device)
        y_win = torch.hann_window(image.size(0), dtype=torch.float, device=image.device)
        #x_win = torch.blackman_window(image.size(1), dtype=torch.float, device=image.device)
        #y_win = torch.blackman_window(image.size(0), dtype=torch.float, device=image.device)
        window = y_win.unsqueeze(1) * x_win.unsqueeze(0)
        image = image * window


    # Perform 2D fft
    image_fft = torch.fft.fft2(image, norm='ortho')
    image_fft = torch.fft.fftshift(image_fft)

    # Calculate the frequency domain mask
    h, w = image.shape[-2:]
    x = torch.arange(-w // 2, w - w // 2).float()
    y = torch.arange(-h // 2, h - h // 2).float()
    x, y = torch.meshgrid(x, y) # 2D grid of x, y
    radius = torch.sqrt(x**2+y**2)  # Distance to center
    #print(mask.to(torch.float32).dtype)

    if use_soft_masks:
        lower_bound = (radius >= min_radius) & (radius < (min_radius + trans_width))
        upper_bound = (radius <= max_radius) & (radius > (max_radius - trans_width))
        bound = lower_bound | upper_bound
        mask = torch.where(lower_bound, (radius - min_radius) / trans_width, torch.tensor([0.]).to(image.device))
        mask += torch.where(upper_bound, (max_radius - radius) / trans_width, torch.tensor([0.]).to(image.device))
        mask = torch.where(bound, mask, ((radius >= min_radius) & (radius <= max_radius)).float())
    else:
        mask = ((radius >= min_radius) & (radius <= max_radius)).float()
    

    if False and use_soft_masks:
        x_win = torch.hann_window(image.size(1), dtype=torch.float, device=image.device)
        y_win = torch.hann_window(image.size(0), dtype=torch.float, device=image.device)
        #x_win = torch.blackman_window(mask.size(1), dtype=torch.float, device=image.device)
        #y_win = torch.blackman_window(mask.size(0), dtype=torch.float, device=image.device)
        window = y_win.unsqueeze(1) * x_win.unsqueeze(0)
        mask = window * mask

    # Perform filtering in frequency domain
    filtered_fft = mask * image_fft.clone()

    # Inverse 2D FFT to get back the spatial domain image
    filtered_fft_shift = torch.fft.ifftshift(filtered_fft)
    filtered_image = torch.fft.ifft2(filtered_fft_shift, norm='ortho')

    # Remove padding if needed
    if do_padding:  
        filtered_image = filtered_image[h_pad:image.size(0)-h_pad, w_pad:image.size(1)-w_pad]

    if do_plot:
        print(f'image_fft.real.min = {image_fft.real.min()}')
        print(f'image_fft.real.max = {image_fft.real.max()}')
        print(f'filtered_fft.real.min = {filtered_fft.real.min()}')
        print(f'filtered_fft.real.max = {filtered_fft.real.max()}')
        fig, axs = plt.subplots(1,4, figsize=(15,4))
        axs = axs.flatten()
        axs[0].imshow(np.log10(image_fft.abs() + 1e-10))
        axs[0].set_title('FFT log magnitue')
        axs[0].grid(False)
        axs[1].imshow(image_fft.angle())
        axs[1].set_title('FFT phase')
        axs[1].grid(False)
        axs[2].imshow(mask)
        axs[2].set_title('mask')
        axs[2].grid(False)
        axs[3].imshow(np.log10(filtered_fft.abs() + 1e-10))
        axs[3].set_title('log filtered_fft mag')
        axs[3].grid(False)
        plt.tight_layout()
        plt.show()

    return torch.abs(filtered_image)
   
class Filterbank2dLoss(nn.Module):
    """ Computes the a loss based on a 2d filterbank. 
    The idea is to have multiple views of the image, where each view is a bandpassed version of the image.
    This is the same idea behind the filterbank anaysis of the RIRs.
    However, this is very hacky and there are some issues, e.g. aliasing.

    Here we do the fitlering in frequency domain, by creating some masks.
    """
    def __init__(self, filters=[(0,5), (0,20), (10, 30), (20, 60), (50, 200)], apply_windowing=True, do_padding=True, 
                 use_soft_masks=True, trans_width=40, criterion='l1', return_breakdown=False, debug=False):
        super(Filterbank2dLoss, self).__init__()
        self.filters = filters
        self.apply_windowing = apply_windowing
        self.do_padding = do_padding
        self.use_soft_masks = use_soft_masks
        self.trans_width = trans_width
        self.return_breakdown = return_breakdown
        self.debug = debug

        self.window = None
        self.radius = None
        self.masks = None
        
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f'ERROR, criterion should l1 or l2, not {criterion}')

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if len(output.shape) < 3:
            output = output[None, None, ...]
            target = target[None, None, ...]
        elif len(output.shape) < 4:
            output = output[None, ...]
            target = target[None, ...]
        
        if len(output.shape) != 4:
            raise ValueError(f'WRONG shape for tensors, I found {output.shape}')
        loss = []

         # Create padded image for reducing spectral leakage
        if self.do_padding: 
            h_pad, w_pad = output.size(-2) // 2, output.size(-1) // 2
            output = torch.nn.functional.pad(output, (w_pad, w_pad, h_pad, h_pad), mode='replicate')
            target = torch.nn.functional.pad(target, (w_pad, w_pad, h_pad, h_pad), mode='replicate')
            self.h_pad, self.w_pad = h_pad, w_pad
            # mode (str) – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            if self.debug:
                if torch.any(torch.isnan(output)):
                    print(f'I found NaNs in padding output')
                if torch.any(torch.isnan(target)):
                    print(f'I found NaNs in padding target')

                
        # Apply Hann window
        if self.apply_windowing:  
            if self.window is None:
                x_win = torch.hann_window(output.size(-1), dtype=torch.float, device=output.device)
                y_win = torch.hann_window(output.size(-2), dtype=torch.float, device=output.device)
                self.window = y_win.unsqueeze(1) * x_win.unsqueeze(0)
            output = output * self.window
            target = target * self.window
            if self.debug:
                if torch.any(torch.isnan(output)):
                    print(f'I found NaNs in windowing output')
                if torch.any(torch.isnan(target)):
                    print(f'I found NaNs in windowing target')

        # Compute template for frequency domain masks
        if self.radius is None:
            h, w = output.shape[-2:]
            x = torch.arange(-w // 2, w - w // 2).float()
            y = torch.arange(-h // 2, h - h // 2).float()
            x, y = torch.meshgrid(x, y)  # 2D grid of x, y
            self.radius = torch.sqrt(x**2+y**2).to(output.device)  # Distance to center
            if self.debug:
                if torch.any(torch.isnan(self.radius)):
                    print(f'I found NaNs in radius')

        # Compute frequency domain masks
        if self.masks is None:
            self.masks = []
            for min_radius,max_radius in self.filters:
                if self.use_soft_masks:
                    if min_radius == 0:  # when low passing, we do not want soft masks near the middle
                        lower_bound = (self.radius >= min_radius) & (self.radius < (min_radius + 0))
                    else:
                        lower_bound = (self.radius >= min_radius) & (self.radius  < (min_radius + self.trans_width))
                    upper_bound = (self.radius  <= max_radius) & (self.radius  > (max_radius - self.trans_width))
                    bound = lower_bound | upper_bound
                    mask = torch.where(lower_bound, (self.radius  - min_radius) / self.trans_width, torch.tensor([0.], device=output.device).to(output.device))
                    mask += torch.where(upper_bound, (max_radius - self.radius ) / self.trans_width, torch.tensor([0.], device=output.device).to(output.device))
                    mask = torch.where(bound, mask, ((self.radius  >= min_radius) & (self.radius  <= max_radius)).float())
                else:
                    mask = ((self.radius >= min_radius) & (self.radius <= max_radius)).float()    
                self.masks.append(mask.clone())
                if self.debug:
                    if torch.any(torch.isnan(mask)):
                        print(f'I found NaNs in  mask')
        
        # Perform 2D fft
        output_fft = torch.fft.fft2(output, norm='ortho')
        output_fft = torch.fft.fftshift(output_fft)
        target_fft = torch.fft.fft2(target, norm='ortho')
        target_fft = torch.fft.fftshift(target_fft)

        if self.debug:
            if torch.any(torch.isnan(output_fft)):
                print(f'I found NaNs in output_fft')
            if torch.any(torch.isnan(target_fft)):
                print(f'I found NaNs in target_fft')

        # Perform filtering in frequency domain
        for mask in self.masks:
            filtered_output_fft = mask * output_fft.clone()
            filtered_target_fft = mask * target_fft.clone()

            if self.debug:
                if torch.any(torch.isnan(filtered_output_fft)):
                    print(f'I found NaNs in filtered_output_fft')
                if torch.any(torch.isnan(filtered_target_fft)):
                    print(f'I found NaNs in filtered_target_fft')

            # Inverse 2D FFT to get back the spatial domain image
            filtered_output_fft = torch.fft.ifftshift(filtered_output_fft)
            filtered_output = torch.fft.ifft2(filtered_output_fft, norm='ortho')
            filtered_target_fft = torch.fft.ifftshift(filtered_target_fft)
            filtered_target = torch.fft.ifft2(filtered_target_fft, norm='ortho')
            if self.debug:
                if torch.any(torch.isnan(filtered_output)):
                    print(f'I found NaNs in filtered_output')
                if torch.any(torch.isnan(filtered_target)):
                    print(f'I found NaNs in filtered_target')

            # Remove padding if needed
            if self.do_padding:  
                filtered_output = filtered_output[..., h_pad:output.size(-2)-self.h_pad, w_pad:output.size(-1)-self.w_pad]
                filtered_target = filtered_target[..., h_pad:output.size(-2)-self.h_pad, w_pad:output.size(-1)-self.w_pad]
                if self.debug:
                    if torch.any(torch.isnan(filtered_output)):
                        print(f'I found NaNs in filtered_output')
                    if torch.any(torch.isnan(filtered_target)):
                        print(f'I found NaNs in filtered_target')

            tmp = self.criterion(filtered_output, filtered_target)
            if self.debug:
                if torch.any(torch.isnan(tmp)):
                    print(f'I found NaNs in final loss')
            loss.append(tmp)

        if self.return_breakdown:
            return loss  # Return loss for each level, as a list
        else:
            return torch.tensor(loss).mean()  # Average across all filters

class WaveletTransformLoss(nn.Module):
    """ Computes a loss based on the wavelet transform of images.
    The WT decomposes the image into multiple levels, and for each level we get 3 high passed versions.
    One for horizontal, vertical, and diagonal frequencies.
    There is also a low pass version.
    So if using 3 levels, using 128x128 inputs, the WT returns:  
        - low pass @ 16x16
        - high pass horizontal, high pass vertical, high pass diagonal @ 16x16  
        - high pass horizontal, high pass vertical, high pass diagonal @ 32,32
        - high pass horizontal, high pass vertical, high pass diagonal @ 64,64
    """
    def __init__(self, levels=3, criterion='l1', return_breakdown=False):
        super(WaveletTransformLoss, self).__init__()
        self.levels = levels
        self.return_breakdown = return_breakdown
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f'ERROR, criterion should l1 or l2, not {criterion}')

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        output_stacks = []
        target_stacks = []

        coefficients_output = ptwt.wavedec2(output, pywt.Wavelet("haar"), level=self.levels, mode="constant")
        coefficients_target = ptwt.wavedec2(target, pywt.Wavelet("haar"), level=self.levels, mode="constant")

        loss = []
        loss.append(self.criterion(coefficients_output[0], coefficients_target[0]))
        for i in range(1, self.levels):
            output_stacks = torch.stack(coefficients_output[i], dim=0)
            target_stacks = torch.stack(coefficients_target[i], dim=0)
            loss.append(self.criterion(output_stacks, target_stacks))

        if self.return_breakdown:
            return loss  # Return loss for each level, as a list
        else:
            return torch.tensor(loss).mean()  # Average across all filters

class SlopedL1Loss(nn.Module):
    def __init__(self, slope=0.1, apply_before_L1=True):
        super(SlopedL1Loss, self).__init__()
        self.slope = slope
        self.apply_before_L1 = apply_before_L1

    def forward(self, output, target):
        if self.apply_before_L1:
            mask1 = (output > -1) & (output < 1) 
            mask2 = ~mask1
            diff = (mask1.type(torch.float32) * output + mask2.type(torch.float32) * self.slope * output) - target
            return torch.mean(diff.abs())
        else:
            raise NotImplementedError

class ComplexSTFTLoss(nn.Module):
    """ General loss to compare complex spectrograms. This can work with the real, imaginary, magnitude, or phase, using
    either L1 or L2 norms. This is a flexible loss that covers many cases, depending on the weights assigned to each part.

    For example, if using only real, imaginary, and magnitude, we get the RI+MAG loss.
    Equation 3 of:
    https://zqwang7.github.io/publications/spl2021_magnitude_phase_compensation.pdf"""
    def __init__(self, distance='L1', weights=[1.0, 1.0, 1.0, 1.0]):
        super(ComplexSTFTLoss, self).__init__()
        if distance == 'L1':
            self.distance = nn.L1Loss()
        elif distance == 'L2':
            self.distance = nn.MSELoss()
        else:
            raise ValueError('ERROR, distance function should be L1 or L2')
        self.weights = weights

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tmp_real, tmp_imag, tmp_mag, tmp_phase = 0, 0, 0, 0
        if self.weights[0] != 0.0:
            tmp_real = self.distance(x.real, y.real)
        if self.weights[1] != 0.0:
            tmp_imag = self.distance(x.imag, y.imag)
        if self.weights[2] != 0.0:
            tmp_mag = self.distance(x.abs(), y.abs())
        if self.weights[3] != 0.0:
            tmp_phase = self.distance(x.angle(), y.angle())

        return self.weights[0] * tmp_real + \
               self.weights[1] * tmp_imag + \
               self.weights[2] * tmp_mag + \
               self.weights[3] * tmp_phase
        
class MultiComplexSTFTLoss(nn.Module):
    """ Multi-resolution STFT loss
    Computes the ComplexSTFTLoss loss of the STFT at different n_fft sizes.
    """

    def __init__(self, n_ffts=(2048, 1024, 512, 256, 128, 64), hop_size=0.75, alpha=1.0, device='cpu', distance='L1',
                 return_breakdown=False, weights_cSTFT: List = (1.0, 1.0, 1.0, 0.0)):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_size = hop_size
        self.alpha = alpha
        self.weights_cSTFT = weights_cSTFT
        self.return_breakdown = return_breakdown

        self.crit = ComplexSTFTLoss(distance=distance, weights=weights_cSTFT)
        self.spec_transforms = [torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                                  win_length=n_fft,
                                                                  hop_length=math.floor(n_fft * self.hop_size),
                                                                  power=None,
                                                                  normalized=False).to(device)
                                for n_fft in n_ffts]

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(output.shape) == 2 or len(
            output.shape) == 3, f'Tensors should be [n_channels, timesteps] with or without batch. Shape = {output.shape}'
        assert output.shape == target.shape, 'Tensors should have the same shape.'

        losses = []
        for spectrogram in self.spec_transforms:
            if spectrogram.n_fft > output.shape[-1]:
                print(
                    f'Warning: Audio too small for n_fft ({spectrogram.n_fft} > {output.shape[-1]}). Skipping this size.')
                continue

            spec_output = spectrogram(output)
            spec_target = spectrogram(target)

            loss = self.crit(spec_output, spec_target)
            losses.append(loss)

        if self.return_breakdown:
            return sum(losses), losses
        return sum(losses)


def unit_test_image_losses():
    import numpy as np
    import time
    """ Quick and dirty test for the 3 image similatiry metrics/losses"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    b,c,h,w = 32,9,128,128
    atol = 1e-8
    output = torch.randn([b,c,h,w], device=device)
    target = torch.randn([b,c,h,w], device=device)

    times = []
    loss_FB = Filterbank2dLoss(debug=False)
    for loss_fn in [WaveletTransformLoss(levels=3),
                    Wasserstein2dLoss(),
                    loss_FB,
                    loss_FB,
                    loss_FB,
                    loss_FB]:
        start_time = time.time()
        loss = loss_fn(output.detach(), output.detach()).item()
        assert np.isclose(loss, 0.0, atol=atol), f'ERROR: {loss_fn} should be 0.0 for same tensor, not {loss}'
        loss = loss_fn(target.detach(), target.detach()).item()
        assert np.isclose(loss, 0.0, atol=atol), f'ERROR: {loss_fn} should be 0.0 for same tensor, not {loss}'
        loss_left = loss_fn(output.detach(), target.detach()).item()
        loss_right = loss_fn(target.detach(), output.detach()).item()
        assert loss_left > 0.0, f'ERROR: {loss_fn} should be > 0.0 for different tensors, not {loss}'
        assert loss_right > 0.0, f'ERROR: {loss_fn} should be > 0.0 for different tensors, not {loss}'
        assert np.isclose(loss_left, loss_right, atol=1e-2), f'ERROR: {loss_fn} should be KINDA symmetrical, not {loss_left} and {loss_right}'
        # Note, this is not really a metric because of the random projections
        # so we used a very relaxed tolerance

        end_time = time.time()
        times.append(end_time - start_time)

    for i, t in enumerate(times):
        print(f'Before breakdownTime taken for iteration {i+1}: {t:.2f} seconds')

    times = []
    # Now with breakdown for each level
    for loss_fn in [WaveletTransformLoss(levels=3, return_breakdown=True),
                    Filterbank2dLoss(return_breakdown=True, debug=False)]:
        start_time = time.time()
        
        loss = loss_fn(output.detach(), output.detach())
        for this_loss in loss:
            assert np.isclose(this_loss.cpu(), 0.0, atol=atol), f'ERROR: {loss_fn} should be 0.0 for same tensor, not {loss}'
        loss = loss_fn(target.detach(), target.detach())
        for this_loss in loss:
            assert np.isclose(this_loss.cpu(), 0.0, atol=atol), f'ERROR: {loss_fn} should be 0.0 for same tensor, not {loss}'
        loss_left = loss_fn(output.detach(), target.detach())
        loss_right = loss_fn(target.detach(), output.detach())
        for this_loss_left, this_loss_right in zip(loss_left, loss_right):
            assert this_loss_left.cpu() > 0.0, f'ERROR: {loss_fn} should be > 0.0 for different tensors, not {this_loss_left}'
            assert this_loss_right.cpu() > 0.0, f'ERROR: {loss_fn} should be > 0.0 for different tensors, not {this_loss_right}'
            assert np.isclose(this_loss_left.cpu(), this_loss_right.cpu(), atol=1e-2), f'ERROR: {loss_fn} should be KINDA symmetrical, not {loss_left} and {loss_right}'


        end_time = time.time()
        times.append(end_time - start_time)
    
    for i, t in enumerate(times):
        print(f'Time taken for iteration {i+1}: {t:.2f} seconds')

    print('>>>>>>>>>>>>>>>>> Unit test success!!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('')
    print('')
    print('')

    return 0

if __name__ == '__main__':
    # Call it like this:
    # (conda-fedora) python -m losses
    unit_test_image_losses()

