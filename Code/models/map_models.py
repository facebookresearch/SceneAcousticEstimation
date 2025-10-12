# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms
from typing import Tuple, Optional, List

class CustomThreshold(nn.Module):
    """ Sets to 0.0 everything outside the range determined by the threshold. 
    Set the threshold to None to ignore. For example:
    min_threshold = 0.1, max_threshold = None
    Will only affect values under 0.1."""
    def __init__(self, min_threshold=0.0, max_threshold=None, channels=[0,1,2]):
        super(CustomThreshold, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.channels = channels

    def forward(self, x: torch.Tensor):
        if self.min_threshold is not None:
            mask_min = x[..., self.channels, :, :] < self.min_threshold
        else:
            mask_min = torch.zeros_like(x[..., self.channels, :, :], device=x.device).to(torch.bool)
        if self.max_threshold is not None:
            mask_max = x[..., self.channels, :, :] > self.max_threshold
        else:
            mask_max = torch.zeros_like(x[..., self.channels, :, :], device=x.device).to(torch.bool)
        mask = torch.logical_or(mask_min, mask_max)
        x[..., self.channels, :, :][mask] = 0.0
        return x

class CustomLinear(nn.Module):
    """ Custom activation function that is linear in the range [-1, 1], but sloped everywhere else.
    This is to reduce the impact of outliers."""
    def __init__(self):
        super(CustomLinear, self).__init__()
        self.slope = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        mask1 = (x > -1) & (x < 1)
        mask2 = ~mask1
        return mask1.type(torch.float32) * x + mask2.type(torch.float32) * self.slope * x

def get_norm_layer(norm_type: str, num_channels: int, num_groups: int):
    """ Reference:
    https://wandb.ai/wandb_fc/GroupNorm/reports/Group-Normalization-in-Pytorch-With-Examples---VmlldzoxMzU0MzMy
    """
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d(num_channels)
    elif norm_type == 'layer':
        norm_layer = nn.GroupNorm(num_channels, num_channels)
    elif norm_type == 'instance':
        norm_layer = nn.GroupNorm(1, num_channels)
    elif norm_type == 'group':
        #raise NotImplementedError('ERROR: Group nomralizaiotn not implemented yet')
        if num_groups == None:
            num_groups = num_channels // 4
        norm_layer = nn.GroupNorm(num_groups, num_channels)
    return norm_layer

def get_activation(activation: str):
    if activation == 'relu':
        activation_layer = nn.ReLU()
    elif activation == 'lrelu':
        activation_layer = nn.LeakyReLU()
    elif activation == 'prelu':
        activation_layer = nn.PReLU()
    elif activation == 'gelu':
        activation_layer = nn.GELU()
    elif activation == 'elu':
        activation_layer = nn.ELU()
    else:
        raise ValueError(f'ERROR, activation function {activation} is not supported.')
    return activation_layer

class DenseConvBlock(nn.Module):
    """ Densely connected block.
    Input is [batch, channels, frequency, timesteps] for spectrograms
    Input is [batch, channels, h, w] for image-like tensors
    """
    def __init__(self, input_channels: int, output_channels: int, n_layers=3, 
                 kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1,
                 use_ConvTransposed=False, normalization='batch', activation='relu', use_conv1x1=True, use_coordconv=False, 
                 use_dropout=False, use_film=False):
        super(DenseConvBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.use_residual = True
        self.use_conv1x1 = use_conv1x1
        self.use_coordconv = use_coordconv

        for layer_idx in range(n_layers):
            tmp = input_channels + layer_idx * output_channels
            print(f'in channels : {tmp}')
            if not use_ConvTransposed:
                if use_coordconv:  # Use CoordConvLayer only at the first layer
                    self.layers.append(CoordConv2d(in_channels=input_channels + layer_idx * output_channels, out_channels=output_channels, kernel_size=kernel,
                                                stride=stride, padding=padding, bias=True))
                    use_coordconv = False
                else:
                    self.layers.append(nn.Conv2d(in_channels=input_channels + layer_idx * output_channels, out_channels=output_channels, kernel_size=kernel,
                                                stride=stride, padding=padding, bias=True))
            else:
                self.layers.append(nn.ConvTranspose2d(in_channels=input_channels + layer_idx * output_channels, out_channels=output_channels, kernel_size=kernel,
                                                      stride=stride, padding=padding, bias=True))
            #self.layers.append(get_norm_layer(normalization, output_channels, None))
            #self.layers.append(get_norm_layer(normalization, tmp, None))
            #self.layers.append(nn.LazyBatchNorm2d())  # This is not working for some reason
            if use_film:
                self.layers.append(FiLM_MLP(output_channels))
            self.layers.append(get_activation(activation))
            if use_dropout:
                self.layers.append(nn.Dropout(0.5))
        
        if self.use_conv1x1:
            self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), stride=stride, padding=(0, 0), bias=True)


        if False:
            raise NotImplementedError()  # This is not ready yet
            if not use_ConvTransposed:
                for ii in range(n_layers):
                    norm = get_norm_layer(normalization, output_channels, None)
                    if use_coordconv:
                        self.layers.append(nn.Sequential(CoordConv2d(in_channels=input_channels * (ii + 1), out_channels=output_channels, kernel_size=kernel,
                                                                    stride=stride, padding=padding, bias=True),
                                                                    norm,
                                                                    nn.ReLU()))
                        use_coordconv = False
                    else:
                        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=input_channels * (ii + 1), out_channels=output_channels, kernel_size=kernel,
                                                                stride=stride, padding=padding, bias=True),
                                                                norm,
                                                                nn.ReLU()))
            else:
                norm = get_norm_layer(normalization, output_channels, None)
                self.layers.append(nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=kernel,
                                                        stride=stride, padding=padding, bias=True),
                                                        norm,
                                                        nn.ReLU()))
                for ii in range(n_layers - 1):
                    norm = get_norm_layer(normalization, output_channels, None)
                    self.layers.append(
                        nn.Sequential(nn.Conv2d(output_channels * (ii + 1) + input_channels, output_channels,
                                                kernel_size=kernel, stride=stride, padding=padding, bias=True),
                                                norm,
                                                nn.ReLU()))

            print(self.layers)
            if self.use_conv1x1:
                self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=(1,1), stride=stride, padding=(0,0), bias=True)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor = None) -> torch.Tensor:
        features = [x]

        ###print('<<<<<<< Dense block forward')
        for layer in self.layers:
            if isinstance(layer, FiLM) or isinstance(layer, FiLM_MLP):
                raise NotImplementedError()
                combined_features = torch.cat(features, dim=-3)
                out = layer(combined_features, conditioning)
            elif isinstance(layer, nn.Conv2d) or isinstance(layer, CoordConv2d) or isinstance(layer, nn.ConvTranspose2d):
                combined_features = torch.cat(features, dim=-3)
                ###print(layer)
                ###print(f'conmbined_features {combined_features.shape}')
                out = layer(combined_features)
                ###print(f'out.shape {out.shape}')
            elif isinstance(layer, nn.ReLU):
                out = layer(out)
                features.append(out)  # Concatenate only outputs after activation
                
                
            #[print(f.shape) for f in features]
            #if isinstance(layer, nn.Conv2d) or isinstance(layer, CoordConv2d):
            #if isinstance(layer, nn.ReLU):
            #    features.append(out)


        #combined_features = torch.cat(features, dim=1)
        if self.use_conv1x1:
            x = self.conv1x1(x)
        out = out + x  # Residual connection
        return out
        return combined_features if not self.use_residual else out

        
    def forward_OLD(self, x):
        features = [x]

        for layer in self.layers:
            tmp = torch.cat(features, dim=-3)  # Concat across channels
            out = layer(tmp)
            features.append(out)

        if self.use_conv1x1:
            x = self.conv1x1(x)
        if self.use_residual:
            out = out + x
        return out

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for ii, layer in enumerate(self.layers):
            self.layers[ii] = layer.to(*args, **kwargs)
        return self

class CoordConv2d(nn.Conv2d):
    # Based on the CoordConv paper:
    #     https://arxiv.org/pdf/1807.03247.pdf
    # This is basically a regular Conv2d layer, but adds positional encodings as extra channels:
    # x = columns from 0 to w, normalized from -1 to 1
    # y = rows from 0 to h, normalized from -1 to 1
    # r (optional) = sqrt(x^2 + y^2)  (so this like a diagonal matrix)
    def __init__(self, *args, with_r=False, **kwargs):
        self.with_r = with_r
        extra_channels = 3 if self.with_r else 2
        print(kwargs.keys())
        kwargs['in_channels'] = kwargs['in_channels'] + extra_channels  # Concat channels for the coordconv layer
        super(CoordConv2d, self).__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Assuming input is [batch, channels, F, frames]  (like a 2d image)
        tmp_x = torch.arange(start=0, end=input.shape[-1], device=input.device)
        tmp_x = 2 * tmp_x / tmp_x.max() - 1  # range [-1, 1]
        x = torch.tile(tmp_x.unsqueeze(0), dims=(input.shape[-2],1)) # [F, frames]
        x = torch.tile(x.unsqueeze(0).unsqueeze(0), dims=(input.shape[0],1,1,1))  # [batch, 1, F, frames]

        tmp_y = torch.arange(start=0, end=input.shape[-2], device=input.device)
        tmp_y = 2 * tmp_y / tmp_y.max() - 1  # range [-1, 1]
        y = torch.tile(tmp_y.unsqueeze(1), dims=(1, input.shape[-1]))  # [F, frames]
        y = torch.tile(y.unsqueeze(0).unsqueeze(0), dims=(input.shape[0], 1, 1, 1))  # [batch, 1, F, frames]

        input = torch.cat([input, y, x], dim=-3)  # Concatenate over channels

        if self.with_r:
            tmp_r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
            input = torch.cat([input, tmp_r], dim=-3)
        return super(CoordConv2d, self).forward(input=input)
    
class ResNetConvBlock(nn.Module):
    """ Implements a basic 2dConv block for ResNet """
    def __init__(self, input_channels, output_channels, 
                 kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1,
                 use_ConvTransposed=False, normalization='batch', activation='relu',
                 use_conv1x1=True, use_coordconv=False, 
                 use_dropout=False, use_film=False):
        super(ResNetConvBlock, self).__init__()

        layers = nn.ModuleList()
        self.use_conv1x1 = use_conv1x1

        if not use_ConvTransposed:
            if use_coordconv:  # Use CoordConvLayer only at the first layer
                layers.append(CoordConv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=True))
                use_coordconv = False
            else:
                layers.append(nn.Conv2d(input_channels, input_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=True))
        else:
            layers.append(nn.ConvTranspose2d(input_channels, input_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=True))
        layers.append(get_norm_layer(normalization, input_channels, None))
        if use_film:
            layers.append(FiLM_MLP(input_channels))
        layers.append(get_activation(activation))
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        if not use_ConvTransposed:
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=True))
        else:
            layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=True))
        layers.append(get_norm_layer(normalization, output_channels, None))
        if use_film:
            layers.append(FiLM_MLP(output_channels))

        if self.use_conv1x1:
            self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=(1,1), stride=stride, padding=(0,0), bias=True)
        self.layers = layers

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor = None) -> torch.Tensor:
        out = x
        for layer in self.layers:
            if isinstance(layer, FiLM) or isinstance(layer, FiLM_MLP):
                out = layer(out, conditioning)
            else:
                out = layer(out)
            #print(f'{layer} {out.shape}')
        if self.use_conv1x1:
            x = self.conv1x1(x)
            #print('1x1')
            #print(x.shape)
        #print('residual')
        out = out + x
        
        return out
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for ii, layer in enumerate(self.layers):
            self.layers[ii] = layer.to(*args, **kwargs)
        return self

class ResamplingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, 
             kernel=(4, 4), stride=(2, 2), padding=(1, 1), groups=1,
             use_ConvTransposed=False):
        super(ResamplingBlock, self).__init__()
    
        layers = nn.ModuleList()
        if not use_ConvTransposed:
            layers.append(nn.Conv2d(input_channels, input_channels, kernel_size=kernel, stride=stride, 
                                    padding=padding, padding_mode='reflect', groups=groups, bias=True))
        else:
            layers.append(nn.ConvTranspose2d(input_channels, input_channels, kernel_size=kernel, stride=stride, 
                                             padding=padding, padding_mode='zeros', groups=groups, bias=True))
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out
        
        return out
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for ii, layer in enumerate(self.layers):
            self.layers[ii] = layer.to(*args, **kwargs)
        return self 

class FiLM(nn.Module):
    def __init__(self, in_dim, gamma_dim=128, beta_dim=128):
        super().__init__()
        #self.gamma = nn.Linear(in_dim, gamma_dim)
        #self.beta = nn.Linear(in_dim, beta_dim)
        self.gamma = nn.LazyLinear(gamma_dim)
        self.beta = nn.LazyLinear(beta_dim)
        self.gamma_fc = nn.Linear(gamma_dim, in_dim)
        self.beta_fc = nn.Linear(beta_dim, in_dim)

    def forward(self, x, conditioning):
        b, c, h, w = x.shape
        conditioning = conditioning.view(-1, conditioning.shape[-3])
        #####print(f'conditioning: {conditioning.shape}')
        gamma = self.gamma(conditioning).sigmoid()
        beta = self.beta(conditioning).sigmoid()
        gamma_fc = self.gamma_fc(gamma)
        beta_fc = self.beta_fc(beta)

        print(f'x.shape {x.shape}')
        print(f'x.view(-1, conditioning.shape[-3]) {x.view(-1, x.shape[-3]).shape}')
        print(f'gamma_fc.shape {gamma_fc.shape}')
        out = x.view(-1, x.shape[-3]) * gamma_fc + beta_fc
        out = out.view(b, c, h, w)
        #####print(f'x.shape {x.shape}')
        #####print(f'x * gamma_fc + beta_fc : {out.shape}')
        return out

class FiLM_MLP(nn.Module):
    def __init__(self, out_dim, activation='elu'):
        super().__init__()
        # 3 layer MLP with shared params
        self.mlp_0 = nn.LazyLinear(out_dim//2)
        self.mlp_1 = nn.LazyLinear(out_dim//2)
        self.mlp_2 = nn.LazyLinear(out_dim * 2)
        if activation == 'relu': # The ISNIDE paper uses tanh
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.tanh()

        # Using 1x1 convs == Linear pixel-wise linear layer
        self.mlp_0 = nn.LazyConv2d(128, kernel_size=1)
        self.mlp_1 = nn.LazyConv2d(128, kernel_size=1)
        self.mlp_2 = nn.LazyConv2d(out_dim * 2, kernel_size=1)

    def forward(self, x, conditioning):
        #print('yolo')
        b, c, h, w = x.shape
        ###conditioning = conditioning.view(-1, conditioning.shape[-3])  # Not neded with pixel-wise linear layers
        #print(f'conditioning: {conditioning.shape}')
        #print(f'x.shape = {x.shape}')
        tmp = self.mlp_0(conditioning)
        tmp = self.activation(tmp)
        tmp = self.mlp_1(tmp)
        tmp = self.activation(tmp)
        tmp = self.mlp_2(tmp)
        #tmp = self.activation(tmp)
        #tmp = tmp.sigmoid()
        gamma = tmp[..., 0:c, :, :]
        beta = tmp[..., c::, :, :]

        ###print(f'x.shape {x.shape}')
        ###print(f'x.view(-1, conditioning.shape[-3]) {x.view(-1, x.shape[-3]).shape}')
        ###print(f'gamma_fc.shape {gamma.shape}')
        #print(f'b, c, h, w  {b} {c} {h} {w}')
        #print(f'gamma.shape {gamma.shape}')
        #print(f'tmp.shape {tmp.shape}')
        out = x * gamma.view(b, c, 1, 1) + beta.view(b, c, 1, 1)
        #out = out.view(b, c, h, w)
        #####print(f'x.shape {x.shape}')
        #####print(f'x * gamma_fc + beta_fc : {out.shape}')
        return out
    
class Encoder(nn.Module):
    def __init__(self, input_channels: int, 
                 channels: List[int] = [16, 32, 64, 128, 256, 512, 512],  # [16, 32, 64, 128, 256, 512, 512], 
                 groups: List[int] = [1, 1, 1, 1, 1, 1, 1], 
                 groups_resamp: List[int] = [1, 1, 1, 1, 2, 4, 4],  
                 normalization: str = 'batch',  # batch
                 activation: str = 'relu',
                 block_type: str = 'resnet',  
                 use_coordconv = True,
                 use_dropout = False,
                 return_skip_connections = True,
                 add_conv1x1_last_layer = False,  # Adds a 1x1 conv to the last layer like a FC, useful for discrimators
                 debug=False):
        """Key parameters:
        blocks = {'resnet', 'dense'}"""
        super(Encoder, self).__init__()
        self.debug = debug

        blocks = nn.ModuleList()
        self.channels = channels
        self.groups = groups
        self.groups_resamp = groups_resamp
        self.use_coordconv = use_coordconv
        self.return_skip_connections = return_skip_connections
        self.add_conv1x1_last_layer = add_conv1x1_last_layer
        assert block_type == 'resnet' or block_type == 'dense', f"ERROR, block type {block_type} is not supported, Use 'resnet' or 'dense'"

        # Main body, increase channels per block
        #channels = [input_channels * 2**i for i in range(7)]
        #channels.append(512)
        self.features_ids = []  # For skip connections
        tmp_channels = [input_channels]
        tmp_channels.extend(channels)
        for i in range(1, len(tmp_channels)):
            if block_type == 'resnet':
                blocks.append(ResNetConvBlock(tmp_channels[i-1], tmp_channels[i], use_conv1x1=True, groups=groups[i-1], normalization=normalization, 
                                              activation=activation, use_coordconv=use_coordconv and i == 1, use_dropout=use_dropout))
            elif block_type == 'dense':
                blocks.append(DenseConvBlock(tmp_channels[i-1], tmp_channels[i], use_conv1x1=True, groups=groups[i-1], normalization=normalization, 
                                             use_coordconv=use_coordconv and i == 1, use_dropout=use_dropout))
            blocks.append(ResamplingBlock(tmp_channels[i],  tmp_channels[i], groups=groups_resamp[i-1]))
            self.features_ids.append(len(blocks) - 2)

        if add_conv1x1_last_layer:
            blocks.append(nn.Conv2d(tmp_channels[-1], 1, kernel_size=1, stride=1, 
                                    padding=0, padding_mode='reflect', groups=1, bias=True))
        self.blocks = blocks


        print(f'Encoder \t features_ids: {self.features_ids}')
        print(f'Encoder \t channels: {self.channels}')
        print(f'Encoder \t groups: {self.groups}')
        print(f'Encoder \t groups_resamp: {self.groups_resamp}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        this_input = x

        for ii in range(len(self.blocks)):
            if self.debug:
                print(f'Encoder input: {this_input.shape}')
            out = self.blocks[ii](this_input)
            if ii in self.features_ids:
                features.append(out)
            this_input = out

        if self.add_conv1x1_last_layer:
            out = out.squeeze([-1, -2, -3])  # Keep only batch dimension

        if self.return_skip_connections:
            return out, features
        else:
            return out

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for ii, block in enumerate(self.blocks):
            self.blocks[ii] = block.to(*args, **kwargs)
        return self

class Decoder(nn.Module):
    def __init__(self, output_channels: int,                  
                 channels: List[int] = [512, 512, 256, 128, 64, 32, 16],  # [512, 512, 256, 128, 64, 32, 16]
                 groups: List[int] = [4, 4, 2, 1, 1, 1, 1], 
                 groups_resamp: List[int] = [512, 512, 256, 128, 64, 32, 16],  # [512, 512, 256, 128, 64, 32, 16], 
                 normalization: str = 'batch',  # batch
                 activation: str = 'relu',
                 block_type: str = 'resnet', 
                 use_dropout = False, use_film=False, use_custom_linear=False,
                 embedding_size=512, debug=False):
        super(Decoder, self).__init__()
        self.debug = debug
        self.use_film = use_film

        blocks = nn.ModuleList()
        self.features_ids = []  # For skip connections
        self.channels = channels
        self.groups = groups
        self.groups_resamp = groups_resamp
        assert block_type == 'resnet' or block_type == 'dense', f"ERROR, block type {block_type} is not supported, Use 'resnet' or 'dense'"
        channels.extend([output_channels])
        
        # Main body, increase channels per block
        ####channels = [embedding_size]
        ####channels.extend([embedding_size // 2**i for i in range(7)]) 
        #print(channels)
        for i in range(0, len(channels) - 1):
            print(i)
            print(channels[i])
            blocks.append(ResamplingBlock(channels[i],  channels[i], use_ConvTransposed=True, groups=groups_resamp[i]))
            if block_type == 'resnet':
                blocks.append(ResNetConvBlock(channels[i]*2, channels[i+1], use_ConvTransposed=True, activation=activation,
                                            use_conv1x1=True, groups=groups[i], normalization=normalization, use_dropout=use_dropout, use_film=use_film))
            elif block_type == 'dense':
                blocks.append(DenseConvBlock(channels[i]*2, channels[i+1], use_ConvTransposed=True, activation=activation,
                                            use_conv1x1=True, groups=groups[i], normalization=normalization, use_dropout=use_dropout, use_film=use_film))
            self.features_ids.append(len(blocks) - 1)

        # Add activation layer to limit the impact of outliers
        if use_custom_linear:
            blocks.append(CustomLinear())

        self.blocks = blocks
        print(f'Decoder \t features_ids: {self.features_ids}')
        print(f'Decoder \t channels: {self.channels}')
        print(f'Decoder \t groups: {self.groups}')
        print(f'Decoder \t groups_resamp: {self.groups_resamp}')

    def forward(self, x: torch.Tensor, encoder_features: List[torch.Tensor], conditioning: torch.Tensor = None) -> torch.Tensor:
        output = x
        if encoder_features is not None:
            encoder_id = len(encoder_features) - 1  # Reverse ids for encoder features
        else:
            encoder_id = None
        for ii in range(len(self.blocks)):
            if self.debug:
                print(f'{encoder_id}  / {ii}')
            layer = self.blocks[ii]
            if ii in self.features_ids:
                if self.debug:
                    tmp = f'Decoder input/encoder features: {output.shape} \t {encoder_features[encoder_id].shape}'
                    print(tmp)
                if encoder_id is not None:
                    output = torch.cat([output, encoder_features[encoder_id]], dim=-3)  # concat across channels
                    encoder_id -= 1
            if self.use_film and isinstance(layer, ResNetConvBlock):
                #print('before Film -----------------------------------------------')
                #print(f'output.shape {output.shape}')
                #print(f'conditioning.shape {conditioning.shape}')
                output = layer(output, conditioning)
            else:
                output = layer(output)
        return output

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for ii, block in enumerate(self.blocks):
            self.blocks[ii] = block.to(*args, **kwargs)
        return self

class Bottleneck(nn.Module):
    def __init__(self, input_channels: int, 
                 debug=False):
        super(Bottleneck, self).__init__()
        self.debug = debug

        blocks = nn.ModuleList()

        # Main body, increase channels per block
        self.features_ids = []  # For skip connections
        tmp_channels = [input_channels]
        for i in range(0, len(tmp_channels)):
            blocks.append(nn.Conv2d(tmp_channels[i], tmp_channels[i], kernel_size=1))
            blocks.append(nn.ReLU())
            blocks.append(FiLM_MLP(tmp_channels[i]))
            
        self.blocks = blocks
        self.channels = tmp_channels
        print(f'Bottleneck \t channels: {self.channels}')
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        this_input = x

        for ii in range(len(self.blocks)):
            if self.debug:
                print(f'Bottleneck input: {this_input.shape}')
                print(f'Bottleneck layer: {self.blocks[ii]}')
            if isinstance(self.blocks[ii], FiLM) or isinstance(self.blocks[ii], FiLM_MLP):
                out = self.blocks[ii](this_input, conditioning)
            else:
                out = self.blocks[ii](this_input)
            this_input = out
        
        if self.debug:
            print(f'Bottleneck output {this_input.shape}')

        return this_input

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for ii, block in enumerate(self.blocks):
            self.blocks[ii] = block.to(*args, **kwargs)
        return self
        
class UnetBasic(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, normalization: str = 'batch', 
                 use_dropout=False, use_custom_linear=False, block_type='resnet', debug=False):
        super(UnetBasic, self).__init__()
        self.debug = debug

        self.encoder = Encoder(input_channels=input_channels, normalization=normalization, use_dropout=use_dropout, block_type=block_type, debug=debug)
        self.decoder = Decoder(output_channels=output_channels, normalization=normalization,  use_dropout=use_dropout, 
                               use_custom_linear=use_custom_linear, block_type=block_type, debug=debug)

    def forward(self, x):
        y, encoder_features = self.encoder(x)  # [batch, feature_maps, freqs=1, timesteps=1]

        if self.debug:
            print('FEATURESSSSSSSS from encoder')
            for aa in encoder_features:
                print(aa.shape)
        if self.debug:
            print(f'Output encoder: {y.shape}')
        y = self.decoder(y, encoder_features)  # [batch, feature_maps, freqs, timesteps]
        if self.debug:
            print(f'Output Decoder: {y.shape}')

        return y

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self

class UnetBottleNeck(nn.Module):
    """ Same Basic Unet but with an extra block in the bottleneck"""
    def __init__(self, input_channels: int, output_channels: int, normalization: str = 'batch', 
                 use_dropout=False, use_custom_linear=False, block_type='resnet', debug=False):
        super(UnetBottleNeck, self).__init__()
        self.debug = debug
        bottleneck_channels = 512

        self.encoder = Encoder(input_channels=input_channels, normalization=normalization, use_dropout=use_dropout, block_type=block_type, debug=debug)
        self.decoder = Decoder(output_channels=output_channels, normalization=normalization,  use_dropout=use_dropout, 
                               use_custom_linear=use_custom_linear, block_type=block_type, debug=debug)
        
        if block_type == 'resnet':
            self.bottleneck = ResNetConvBlock(bottleneck_channels, bottleneck_channels, use_conv1x1=True, groups=1, normalization=normalization, 
                                              activation='relu', use_coordconv=False, use_dropout=use_dropout)
        elif block_type == 'dense':
            self.bottleneck = DenseConvBlock(bottleneck_channels, bottleneck_channels, use_conv1x1=True, groups=1, normalization=normalization, 
                                             use_coordconv=False, use_dropout=use_dropout)

    def forward(self, x):
        y, encoder_features = self.encoder(x)  # [batch, feature_maps, freqs=1, timesteps=1]

        if self.debug:
            print('FEATURESSSSSSSS from encoder')
            for aa in encoder_features:
                print(aa.shape)
        if self.debug:
            print(f'Output encoder: {y.shape}')

        y = self.bottleneck(y)

        y = self.decoder(y, encoder_features)  # [batch, feature_maps, freqs, timesteps]
        if self.debug:
            print(f'Output Decoder: {y.shape}')

        return y

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self

class UnetFilm(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck_channels: int = 512, normalization: str = 'batch', 
                 use_dropout=False, use_film_decoder=False, block_type='resnet', debug=True):
        super(UnetFilm, self).__init__()
        self.debug = debug
        self.use_film_decoder = use_film_decoder

        self.encoder = Encoder(input_channels=input_channels, normalization=normalization,  use_dropout=use_dropout, block_type=block_type,  debug=debug)
        self.decoder = Decoder(output_channels=output_channels, normalization=normalization,  use_dropout=use_dropout, 
                               use_film=use_film_decoder, block_type=block_type, debug=debug)
        self.bottleneck = Bottleneck(bottleneck_channels)
        
    def forward(self, x, conditioning):
        y, encoder_features = self.encoder(x)  # [batch, feature_maps, freqs=1, timesteps=1]

        if self.debug:
            print('FEATURESSSSSSSS from encoder')
            for aa in encoder_features:
                print(aa.shape)
        if self.debug:
            print(f'Output encoder: {y.shape}')

        y = self.bottleneck(y, conditioning)
        if self.debug:
            print(f'AFter bottlenck y.shape {y.shape}')

        if self.use_film_decoder:
            y = self.decoder(y, encoder_features, conditioning)  # [batch, feature_maps, freqs, timesteps]
        else:        
            y = self.decoder(y, encoder_features)  # [batch, feature_maps, freqs, timesteps]
        if self.debug:
            print(f'Output Decoder: {y.shape}')

        return y

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self
    
class UnetConcatBottleNeck(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck_channels: int = 512, conditioning_channels: int = 512,
                 normalization: str = 'batch', use_bias_linear_projection=False, use_dropout=False, debug=True):
        super(UnetConcatBottleNeck, self).__init__()
        self.debug = debug
        self.bottleneck_channels = bottleneck_channels
        self.conditioning_channels = conditioning_channels

        self.encoder = Encoder(input_channels=input_channels, normalization=normalization,  use_dropout=use_dropout, debug=debug)
        self.decoder = Decoder(output_channels=output_channels, normalization=normalization,  
                               channels = [bottleneck_channels+(0*512), 512, 256, 128, 64, 32, 16],
                               use_dropout=use_dropout, use_film=False, debug=debug)
        self.linear = nn.LazyLinear(bottleneck_channels, bias=use_bias_linear_projection)

    def forward(self, x, conditioning):
        y, encoder_features = self.encoder(x)  # [batch, feature_maps, freqs=1, timesteps=1]

        if self.debug:
            print('FEATURESSSSSSSS from encoder')
            for aa in encoder_features:
                print(aa.shape)
        if self.debug:
            print(f'Output encoder: {y.shape}')

        # Linear projection of concatenated latent and conditioning
        #print(f'y.shape {y.shape}')
        #print(f'conditioning.shape {conditioning.shape}')
        latent = torch.concat([y, conditioning], dim=-3)
        #print(f'latent.shape {latent.shape}')
        #print(f'self.bottleneck_channels {self.bottleneck_channels}')
        y = self.linear(latent.view(-1, self.bottleneck_channels + self.conditioning_channels))
        y = y.view(-1, self.bottleneck_channels, 1 ,1)

        if self.debug:
            print(f'AFter bottlenck y.shape {y.shape}')

        # Decoder      
        #print(f'y.shape {y.shape}')
        y = self.decoder(y, encoder_features)  # [batch, feature_maps, freqs, timesteps]
        if self.debug:
            print(f'Output Decoder: {y.shape}')

        return y

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self
        
def test_unet():
    """ Quick test for the UNET model.
    This overfits 2 synthetic data samples."""
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchinfo import summary

    import features

    learning_rate = 0.001
    datapoints, batch, epochs = 2, 2, 1000000
    resolution = 128
    input_shape = [2, 8, resolution, resolution]  # encoder input, maps output
    output_shape = [2, 512, 1, 1]  # bottleneck, decoder input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    use_film = True
    use_concat_bottle = True
    use_dropout = False
    debug = False
    use_stft = True  # for conditioning of Film

    x = torch.randn(input_shape, dtype=dtype).to(device)
    y = torch.randn(output_shape, dtype=dtype).to(device)
    encoder_features = [torch.randn([2, 16, 128, 128], dtype=dtype, device=device),
                       torch.randn([2, 32, 64, 64], dtype=dtype, device=device),
                       torch.randn([2, 64, 32, 32], dtype=dtype, device=device),
                       torch.randn([2, 128, 16, 16], dtype=dtype, device=device),
                       torch.randn([2, 256, 8, 8], dtype=dtype, device=device),
                       torch.randn([2, 512, 4, 4], dtype=dtype, device=device),
                       torch.randn([2, 512, 2, 2], dtype=dtype, device=device)]
    if use_film:
        conditioning = torch.randn([2, 512, 1, 1],  dtype=dtype).to(device)
    audio = torch.randn([2, 1, 24000], dtype=dtype).to(device)
    
    data = torch.utils.data.TensorDataset(x, x)
    dataloader = DataLoader(data, batch_size=batch)
    #model = Encoder(input_channels=input_shape[-3], inner_channels=16, debug=debug).to(device)
    #model = Decoder(output_channels=input_shape[-3], inner_channels=16, debug=debug).to(device)
        
    if use_film:
        #model = UnetFilm(input_channels=input_shape[-3], output_channels=input_shape[-3], use_dropout=use_dropout, debug=debug).to(device)
        model = UnetFilm(input_channels=input_shape[-3], output_channels=input_shape[-3], use_dropout=use_dropout, use_film_decoder=True, debug=debug).to(device)
    elif use_concat_bottle:
        conditioning_channels = 32
        model = UnetConcatBottleNeck(input_channels=input_shape[-3], output_channels=input_shape[-3], 
                                     bottleneck_channels=512, conditioning_channels=conditioning_channels, 
                                     use_bias_linear_projection=False, use_dropout=use_dropout, debug=debug).to(device)
        stft_tform = features.Feature_MelPlusPhase(n_mels=128, hop=188, nfft=1024).to(device)
        stft_encoder = Encoder(input_channels=1, use_coordconv=False, use_dropout=use_dropout, 
                               channels=[16, 32, 64, 128, 256, 512, conditioning_channels],
                               return_skip_connections=False, debug=debug).to(device)
    else:
        model = UnetBasic(input_channels=input_shape[-3], output_channels=input_shape[-3], use_dropout=use_dropout, debug=debug).to(device)
        
    if use_stft:
         stft_tform = features.Feature_MelPlusPhase(n_mels=128, hop=188, nfft=1024).to(device)
         stft_encoder = Encoder(input_channels=audio.shape[-2], use_coordconv=False, use_dropout=use_dropout, return_skip_connections=False, debug=debug).to(device)
    
    #loss_f = MSELossComplex()
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_film:
            # To test UnetFilm
        summary(model, input_data=[x, conditioning],
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=10, dtypes=[dtype])
    elif use_concat_bottle:
        # To test concat bottleneck model
        stft = stft_tform(audio)
        summary(stft_encoder, input_data=stft.to(device),
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=2, dtypes=[dtype])

        conditioning = stft_encoder(stft)
        print(f'conditionig : {conditioning.shape}')
        print(f'x : {x.shape}')
        summary(model, input_data=[x, conditioning],
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=10, dtypes=[dtype])
    else:
        # To test Encoder only, or full Unet
        summary(model, input_data=[x],
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['depth'], depth=10, dtypes=[dtype])


    if False:
        # To test the decoder only
        summary(model, input_data=[y, encoder_features],
            col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
            row_settings=['depth'], depth=10, dtypes=[dtype])

    if debug:
        return 0 
        
    # Train
    model.train()
    for epoch in range(epochs):
        for ctr, (x, target) in enumerate(dataloader):
            # x, target = x.to(device), target.to(device)
            model.zero_grad()

            if use_stft:
                stft = stft_tform(audio.to(device))
                conditioning = stft_encoder(stft)

            if use_concat_bottle:
                stft = stft_tform(audio.to(device))
                conditioning = stft_encoder(stft)

            if not use_film and not use_concat_bottle:
                out = model(x)
            else:
                #out = model(x, conditioning[0:1, ...])
                out = model(x, conditioning)

            loss = loss_f(out, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0 and ctr == 0:
                print('Epoch: {} / {} , loss {:.8f}'.format(epoch, epochs, loss.item()))
                # print('outputs : {}'.format(out.detach().cpu().numpy()))

    model.eval()
    out = model(x)
    print('')
    print('outputs : {}'.format(out.detach().cpu().numpy()))
    a = out.detach().cpu().numpy()
    b = target.detach().cpu().numpy()
    print('target : {}'.format(b))
    assert np.allclose(a, b, atol=1e-7), 'Wrong outputs'

    print('Unit test completed.')

if __name__=='__main__':
    test_unet()
