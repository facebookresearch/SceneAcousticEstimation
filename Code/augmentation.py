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
import torchvision
from typing import Tuple, Optional, Dict, List, Union

import utils

def get_bounding_box(map: torch.Tensor, debug=False):
    """ Finds the bounding box of non NaN pixles in the floormap or acumap.
    Returns a tensor of shape [channels, 4],
    Wich a bbox (xmin, ymin, xmax, ymax) for each channel . """
    if len(map.shape) < 3:  # Add channel dimension if needed
        map = map[None]

    bboxes = []
    for i in range(map.shape[0]):
        #non_nans = torch.nonzero(~torch.isnan(map[i,:,:]))
        non_zeros = torch.nonzero(map[i,:,:])
        xmin = non_zeros[:,1].min()
        ymin = non_zeros[:,0].min()
        xmax = non_zeros[:,1].max()
        ymax = non_zeros[:,0].max()
        bbox = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        bboxes.append(bbox)
        if debug:
            print(f"Bounding boxes for each channel: {bbox}")
    return torch.stack(bboxes, dim=0)

def center_bounding_box(bbox: torch.Tensor, map_size_meters: float, grid_lims=[128, 128], debug=False):
    """ Translates a bouding box to the de center of the grid.
        grd_lims is the size of the grid in pixels, so 128, for 128x128. """
    assert len(bbox.shape) == 1 and bbox.shape[0] == 4, f'ERROR, worng shape for bbox, I was expeting [4] but found {bbox.shape}'

    tmp = torch.tensor(grid_lims)
    #map_size_meters = tmp.abs().sum().item()  # NOTE: this is fixed from the data, the total area of the 128,128 map
    #map_size_meters = 20

    xmin = bbox[0].item()
    ymin = bbox[1].item()
    xmax = bbox[2].item()
    ymax = bbox[3].item()

    # Translate to center
    new_xmin = xmin - ((xmax + xmin) / 2 - grid_lims[0]//2)
    new_ymin = ymin - ((ymax + ymin) / 2 - grid_lims[1]//2)
    new_xmax = xmax - ((xmax + xmin) / 2 - grid_lims[0]//2)
    new_ymax = ymax - ((ymax + ymin) / 2 - grid_lims[1]//2)
    new_box = torch.tensor([new_xmin, new_ymin, new_xmax, new_ymax], dtype=torch.int64, device=bbox.device)
    translation = torch.tensor([((xmax + xmin) / 2 - grid_lims[0]//2), ((ymax + ymin) / 2 - grid_lims[0]//2)],  device=bbox.device)
    translation = torch.tensor([new_xmin - xmin, new_ymin - ymin],  device=bbox.device).floor()

    # Parse pixel coords to meters
    pixel_size_x = map_size_meters / grid_lims[0]  # meters per pixel in x direction
    pixel_size_y = map_size_meters / grid_lims[1]  # meters per pixel in y direction
    translation_meters = translation * torch.tensor([pixel_size_x, pixel_size_y],  device=bbox.device)  # element wise multiplication

    hipo = math.sqrt(math.ceil((xmax - xmin)/2)**2 + math.ceil((ymax - ymin)/2)**2)
    if debug:
        print(f"Translated boxes for each channel: {new_box}, translation: {translation}, hipo: {hipo}")
    return new_box, translation, translation_meters

def ramdon_translate_bounding_box(bbox: torch.Tensor,map_size_meters: float,  grid_lims=[128, 128], debug=False):
    """ Translates a bounding box but making sure the translated box stays within the valid range """
    assert len(bbox.shape) == 1 and bbox.shape[0] == 4, f'ERROR, worng shape for bbox, I was expeting [4] but found {bbox.shape}'
    xmin = bbox[0].item()
    ymin = bbox[1].item()
    xmax = bbox[2].item()
    ymax = bbox[3].item()

    tmp = torch.tensor(grid_lims)
    #map_size_meters = tmp.abs().sum().item() 
    #map_size_meters = 20

    translation = [np.random.rand() * (grid_lims[0] - xmax + xmin) - xmin,
                    np.random.rand() * (grid_lims[1] - ymax + ymin) - ymin]
    translation[1] = -translation[1]  # different conventions
    translation = torch.tensor(translation, device=bbox.device).floor()
    
    # Parse pixel coords to meters
    pixel_size_x = map_size_meters/ grid_lims[0]  # meters per pixel in x direction
    pixel_size_y = map_size_meters / grid_lims[1]  # meters per pixel in y direction
    translation_meters = translation * torch.tensor([pixel_size_x, pixel_size_y],  device=bbox.device)  # element wise multiplication

    if debug:
        print(f" translation: {translation}")
    return translation, translation_meters
    

class RandomRotation(nn.Module):
    """ Performns a random rotation of image-like data (floormaps, acumaps) and sources and receivers.
    There are no hard constraints on the translations, so the maps or source can go outside the boundary of the floorplan."""
    def __init__(self, angle: float = None, mode='per_batch', device='cpu', debug=False):
        super(RandomRotation, self).__init__()
        self.angle = np.pi / 4
        self.debug = debug
        self.mode = mode
        self.device = device

        assert mode in ['per_example', 'per_batch'], f"ERROR, mode for RandomRotation shoulde be ['per_example', 'per_batch'], but found {mode}"
        
        if angle is not None:
            self.angle = angle
            
    def set_angle(self):
        if self.mode == 'per_example':
            raise NotImplementedError()
        elif self.mode == 'per_batch':
            self.angle = np.random.rand(1)[0] * 2*np.pi
        self.rot_mat = utils.get_rotation_matrix(0.0, 0.0, self.angle, device=self.device)
            
    def forward(self, src: torch.Tensor, rec: torch.Tensor, floormap: torch.Tensor, acumap: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        self.set_angle()
        
        # Add batch if needed to src and rec
        do_reshape = False
        if len(src.shape) < 2:
            do_reshape = True
            src = src[None, ...] 
        if len(rec.shape) < 2:
            do_reshape = True
            rec = rec[None, ...]

        # Rotate src and recs (3d points)
        if self.debug:
            print(f'src.shape {src.shape}')
            print(f'self.rot_mat.shape { self.rot_mat.shape}')
        src_rotated = torch.einsum('ik,kj->ij', src.to(torch.float64), self.rot_mat.T).to(torch.float32)
        rec_rotated = torch.einsum('ik,kj->ij', rec.to(torch.float64), self.rot_mat.T).to(torch.float32)

        # Rotated floormap and acumap (images)
        if self.debug:
            print(f'floormap.shape {floormap.shape}')
        floormap = torchvision.transforms.functional.rotate(floormap, -1 * np.rad2deg(self.angle), fill=0.0)
        if self.debug:
            print(f'floormap.shape {acumap.shape}')
        acumap = torchvision.transforms.functional.rotate(acumap, -1 * np.rad2deg(self.angle), fill=np.nan)

        if do_reshape:
            src_rotated = src_rotated.squeeze(0)
            rec_rotated = rec_rotated.squeeze(0)

        return src_rotated, rec_rotated, floormap, acumap

class RandomTranslation(nn.Module):
    """ Applies a random transaltion to  image-like data (floormaps, acumaps) and sources and receivers.
    There are no hard constraints on the translations, so the maps or source can go outside the boundary of the floorplan."""
    def __init__(self, translation: List[float] = None, mode='per_batch', xlim_translation: List[float]=[-5, 5], xlim_max: List[float]=[-10, 10], debug=False):
        super(RandomTranslation, self).__init__()
        self.debug = debug
        self.mode = mode
        self.xlim_max = np.abs(np.array(xlim_max)).sum()
        self.xlim_translation = np.array(xlim_translation)

        assert mode in ['per_example', 'per_batch', 'fixed'], f"ERROR, mode for RandomRotation shoulde be ['per_example', 'per_batch', 'fixed'], but found {mode}"
        
        if translation is not None:
            self.translation = translation
            
    def set_translation(self):
        if self.mode == 'per_example':
            raise NotImplementedError()
        elif self.mode == 'per_batch':
            if self.debug:
                tmp = np.random.rand(2)
                print(tmp)
                tmp = np.abs(self.xlim_translation).sum()
                print(tmp)
                tmp = np.random.rand(2) * np.abs(self.xlim_translation).sum() - (np.abs(self.xlim_translation).sum() // 2) 
                print(tmp)
                print(f'translate {tmp}')
                self.translation = tmp
                #self.translation = tmp.tolist()
                #self.translation = np.array([1.0, 2.0])
                #self.translation = np.array([5.0, 5.0])
                #self.translation = np.array([0.0, 0.0])
            self.translation = np.random.rand(2) * np.abs(self.xlim_translation).sum() - (np.abs(self.xlim_translation).sum() // 2) 
            
    def forward(self, src: torch.Tensor, rec: torch.Tensor, floormap: torch.Tensor, acumap: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        image_size = floormap.shape[-1]
        self.set_translation()  # This is in meters, that we then have to transform to pixle coords, using the image_siez (resolution) and known boundaries
        
        # Add batch if needed to src and rec
        do_reshape = False
        if len(src.shape) < 2:
            do_reshape = True
            src = src[None, ...] 
        if len(rec.shape) < 2:
            do_reshape = True
            rec = rec[None, ...]

        # Translate src and recs (3d points), with flipped axis due to different conventions
        if self.debug:
            print(f'src.shape {src.shape}')
        src[:, 0] = src[:, 0] + self.translation[1]
        src[:, 1] = src[:, 1] - self.translation[0]
        rec[:, 0] = rec[:, 0] + self.translation[1]
        rec[:, 1] = rec[:, 1] - self.translation[0]

        # Map translation values (in meters, for source and rec), to pixel values (for floormaps and acumaps) 
        pixel_translation = self.translation * image_size / self.xlim_max
        pixel_translation = pixel_translation

        if self.debug:
            print(f'self.translation {self.translation}')
            print(f'self.translation * image_size {self.translation * image_size}')
            print(f'self.xlim_max { self.xlim_max}')
            print(f'pixel_translation {pixel_translation}')
            
        # Rotated floormap and acumap (images)
        if self.debug:
            print(f'floormap.shape {floormap.shape}')
        floormap = torchvision.transforms.functional.affine(floormap, translate=pixel_translation.tolist(), angle=0.0, shear=0.0, scale=1.0)
        if self.debug:
            print(f'floormap.shape {acumap.shape}')
        acumap = torchvision.transforms.functional.affine(acumap, translate=pixel_translation.tolist(), fill=np.nan, angle=0.0, shear=0.0, scale=1.0)

        if do_reshape:
            src = src.squeeze(0)
            rec = rec.squeeze(0)
        return src, rec, floormap, acumap
    
class RandomTranslationWithBoundingBox(nn.Module):
    def __init__(self, bbox_channel=1, xlim_max: List[float]=[-10, 10], debug=False):
        super(RandomTranslationWithBoundingBox, self).__init__()
        self.debug = debug
        self.bbox_channel = bbox_channel  # Chanel from floormap to use to compute bounding box, should be the full floormap for best results
        self.xlim_max = np.abs(np.array(xlim_max)).sum()

    def forward(self,
                src: Union[torch.Tensor, List[torch.Tensor]], rec:  Union[torch.Tensor, List[torch.Tensor]], 
                floormap: torch.Tensor, acumap: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                
        # Cast to list if needed, this is to support multiple src, recs per item
        if isinstance(src, torch.Tensor):
            src = [src]
        if isinstance(rec, torch.Tensor):
            rec = [rec]

        # Add batch if needed
        do_reshape = False
        if len(src[0].shape) < 2:
            do_reshape = True
            for jj in range(len(src)):
                src[jj] = src[jj][None, ...] 
        if len(rec[0].shape) < 2:
            do_reshape = True
            for jj in range(len(rec)):
                rec[jj] = rec[jj][None, ...] 
        if len(floormap.shape) < 4:
            do_reshape = True
            floormap = floormap[None, ...]
        if len(acumap.shape) < 4:
            do_reshape = True
            acumap = acumap[None, ...]
        if mask is not None and len(mask.shape) < 4:
            do_reshape = True
            mask = mask[None, ...]

        # Find and center bounding boxes
        translations = []
        translations_meters = []
        for i in range(floormap.shape[0]):
            bbox = get_bounding_box(map=floormap[i, self.bbox_channel, :, :])
            centered_box, center_translation, center_translation_meters = center_bounding_box(bbox[0], self.xlim_max, debug=self.debug)
            new_translation, new_translation_meters = ramdon_translate_bounding_box(centered_box, self.xlim_max, debug=self.debug)
            translation = center_translation + new_translation
            translation_meters = center_translation_meters + new_translation_meters
            translations.append(translation)
            translations_meters.append(translation_meters)
        
        # Translate src and recs (3d points), with flipped axis due to different conventions
        ###print(f'translations {translations}')
        ###print(f'translations_meters {translations_meters}')
        translations_meters = torch.stack(translations_meters, dim=0)
        if self.debug:
            print(f'src.shape {src[0].shape}')
            print(f'translations_meters.shape {translations_meters.shape}')
        for jj in range(len(src)):
            src[jj][:, 0] = src[jj][:, 0] + translations_meters[:, 1]
            src[jj][:, 1] = src[jj][:, 1] - translations_meters[:, 0]
            rec[jj][:, 0] = rec[jj][:, 0] + translations_meters[:, 1]
            rec[jj][:, 1] = rec[jj][:, 1] - translations_meters[:, 0]

        if self.debug:
            print(f'random {np.random.rand()}')
            print(f'self.translations {translations}')
            print(f'self.translations_meters {translations_meters}')
            print(f'self.xlim_max { self.xlim_max}')
             
        # Translated floormap and acumap (images)
        if self.debug:
            print(f'floormap.shape {floormap.shape}')
            print(f'floormap.shape {acumap.shape}')
        for i in range(floormap.shape[0]):
            floormap[i] = torchvision.transforms.functional.affine(floormap[i], translate=translations[i].tolist(), angle=0.0, shear=0.0, scale=1.0)
            acumap[i] = torchvision.transforms.functional.affine(acumap[i], translate=translations[i].tolist(), fill=np.nan, angle=0.0, shear=0.0, scale=1.0)
            if mask is not None:
                #print(f'mask.shape {mask.shape}')
                mask[i] = torchvision.transforms.functional.affine(mask[i], translate=translations[i].tolist(), fill=0.0, angle=0.0, shear=0.0, scale=1.0)
                #coso = [32, 32]
                #mask[i] = torchvision.transforms.functional.affine(mask[i], translate=coso, fill=0.0, angle=0.0, shear=0.0, scale=1.0)
                #mask[i, :, 10:100, 10:100] = 0.0
                #mask[i] = torch.zeros_like(acumap[i])
                #print(f'mask.shape {mask.dtype}')
        
        if do_reshape:
            for jj in range(len(src)):
                src[jj] = src[jj].squeeze(0)
                rec[jj] = rec[jj].squeeze(0)
            floormap = floormap.squeeze(0)
            acumap = acumap.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        # When lists are not needed, return single tensor
        if len(src) == 1:
            src = src[0]
        if len(rec) == 1:
            rec = rec[0]
        
        if mask is not None:
            return src, rec, floormap, acumap, mask
        else:
            return src, rec, floormap, acumap

class RandomCenteredRotationWithBoundingBox(nn.Module):
    def __init__(self, bbox_channel=1, xlim_max: List[float]=[-10, 10], return_rot_angles: bool = False, debug=False):
        super(RandomCenteredRotationWithBoundingBox, self).__init__()
        self.debug = debug
        self.return_rot_angles = return_rot_angles
        self.bbox_channel = bbox_channel  # Chanel from floormap to use to compute bounding box
        self.xlim_max = np.abs(np.array(xlim_max)).sum()
    

    def get_rotation(self, bbox: torch.Tensor, grid_lims=[128, 128], debug=False):
        """ Returns a translation matrix and rotation angle, that this valid given the grid and the hipothenuze.
        This guarantees that the rotated bounding box will be within the grid limits"""
        xmin = bbox[0].item()
        ymin = bbox[1].item()
        xmax = bbox[2].item()
        ymax = bbox[3].item()
        hipo = math.sqrt(math.ceil((xmax - xmin)/2)**2 + math.ceil((ymax - ymin)/2)**2)
        limit = grid_lims[0] // 2

        if hipo < limit:
            angle = np.random.rand(1)[0] * 2*np.pi
        else:
            angle = np.random.choice([0, np.pi/2, 2*np.pi/2, 3*np.pi/2])

        rot_mat = utils.get_rotation_matrix(0.0, 0.0, angle, device=bbox.device)

        if debug:
            print(f'ROTATION angle: {angle}')
        return angle, rot_mat

    def forward(self, src: Union[torch.Tensor, List[torch.Tensor]], rec: Union[torch.Tensor, List[torch.Tensor]], 
                floormap: torch.Tensor, acumap: torch.Tensor, mask: torch.Tensor = None, 
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        # Cast to list if needed, this is to support multiple src, recs per item
        if isinstance(src, torch.Tensor):
            src = [src]
        if isinstance(rec, torch.Tensor):
            rec = [rec]
        image_size = floormap.shape[-1]
        
        # Add batch if needed
        do_reshape = False
        if len(src[0].shape) < 2:
            do_reshape = True
            for jj in range(len(src)):
                src[jj] = src[jj][None, ...] 
        if len(rec[0].shape) < 2:
            do_reshape = True
            for jj in range(len(rec)):
                rec[jj] = rec[jj][None, ...] 
        if len(floormap.shape) < 4:
            do_reshape = True
            floormap = floormap[None, ...]
        if len(acumap.shape) < 4:
            do_reshape = True
            acumap = acumap[None, ...]
        if mask is not None and len(mask.shape) < 4:
            do_reshape = True
            mask = mask[None, ...]

        # Find and center boxes, rotate, translate
        translations = []
        translations_meters = []
        rot_mats = []
        non_nans = ~torch.isnan(acumap)
        for i in range(floormap.shape[0]):
            bbox = get_bounding_box(map=floormap[i, self.bbox_channel, :, :])
            centered_box, center_translation, center_translation_meters = center_bounding_box(bbox[0], self.xlim_max, debug=self.debug)
            angle, rot_mat = self.get_rotation(centered_box, debug=self.debug)

            floormap[i] = torchvision.transforms.functional.affine(floormap[i], translate=center_translation.tolist(), fill=0.0, angle=0.0, shear=0.0, scale=1)
                                                                   #interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            acumap[i] = torchvision.transforms.functional.affine(acumap[i], translate=center_translation.tolist(), fill=np.nan, angle=0.0, shear=0.0, scale=1.0)
            if mask is not None:
                mask[i] = torchvision.transforms.functional.affine(mask[i], translate=center_translation.tolist(), fill=0.0, angle=0.0, shear=0.0, scale=1.0)

            
            floormap[i] = torchvision.transforms.functional.affine(floormap[i], translate=[0,0], fill=0.0, angle=1 * np.rad2deg(angle), shear=0.0, scale=1.0)
                                                                   #interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            ##acumap[i][non_nans[i]] = 0.0  # Remove nans so that the interpolation works   # this does not work
            acumap[i] = torchvision.transforms.functional.affine(acumap[i], translate=[0,0], fill=np.nan, angle=1 * np.rad2deg(angle), shear=0.0, scale=1.0)
                                                                 #interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            ##acumap[i][non_nans[i]] = np.nan  # Remove nans so that the interpolation works
            if mask is not None:
                mask[i] = torchvision.transforms.functional.affine(mask[i], translate=[0,0], fill=0.0, angle=1 * np.rad2deg(angle), shear=0.0, scale=1.0,
                                                                   interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            
            if False:
                floormap[i] = torchvision.transforms.functional.affine(floormap[i], translate=center_translation.tolist(), angle=1 * np.rad2deg(angle), shear=0.0, scale=1.0)
                acumap[i] = torchvision.transforms.functional.affine(acumap[i], translate=center_translation.tolist(), fill=np.nan, angle=1 * np.rad2deg(angle), shear=0.0, scale=1.0)
                if mask is not None:
                    mask[i] = torchvision.transforms.functional.affine(mask[i], translate=center_translation.tolist(), fill=0.0, angle=1 * np.rad2deg(angle), shear=0.0, scale=1.0)
            translations.append(center_translation)
            translations_meters.append(center_translation_meters)
            rot_mats.append(rot_mat)
        
        # Translate src and recs (3d points), with flipped axis due to different conventions
        translations_meters = torch.stack(translations_meters, dim=0)
        if self.debug:
            print(f'src[0].shape {src[0].shape}')
            print(f'translations_meters.shape {translations_meters.shape}')
        for jj in range(len(src)):
            src[jj][:, 0] = src[jj][:, 0] + translations_meters[:, 1]
            src[jj][:, 1] = src[jj][:, 1] - translations_meters[:, 0]
            rec[jj][:, 0] = rec[jj][:, 0] + translations_meters[:, 1]
            rec[jj][:, 1] = rec[jj][:, 1] - translations_meters[:, 0]

        for jj in range(len(src)):
            for i in range(floormap.shape[0]):
                src[jj][i] = torch.einsum('ik,kj->ij', src[jj][i:i+1].to(torch.float64), rot_mats[i].T).to(torch.float32)
                rec[jj][i]  = torch.einsum('ik,kj->ij', rec[jj][i:i+1].to(torch.float64), rot_mats[i].T).to(torch.float32)

        if self.debug:
            print(f'random {np.random.rand()}')
            print(f'self.translations {translations}')
            print(f'self.translations_meters {translations_meters}')
            print(f'self.xlim_max { self.xlim_max}')
             
        if do_reshape:
            for jj in range(len(src)):
                src[jj] = src[jj].squeeze(0)
                rec[jj] = rec[jj].squeeze(0)
            floormap = floormap.squeeze(0)
            acumap = acumap.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        # When lists are not needed, return single tensor
        if len(src) == 1:
            src = src[0]
        if len(rec) == 1:
            rec = rec[0]

        if mask is not None:
            return src, rec, floormap, acumap, mask
        elif self.return_rot_angles:
            return src, rec, floormap, acumap, angle
        else:
            return src, rec, floormap, acumap
        

class CombinedAugmentation(nn.Module):
    def __init__(self, bbox_channel=1, xlim: List[float]=[-10, 10], return_rot_angles: bool = False):
        super(CombinedAugmentation, self).__init__()
        self.return_rot_angles = return_rot_angles
        self.rotator = RandomCenteredRotationWithBoundingBox(bbox_channel=bbox_channel, xlim_max=xlim, return_rot_angles=return_rot_angles)
        self.translator = RandomTranslationWithBoundingBox(bbox_channel=bbox_channel, xlim_max=xlim)
    
    def forward(self, src: torch.Tensor, rec: torch.Tensor, floormap: torch.Tensor, acumap: torch.Tensor, 
        mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,  torch.Tensor):
        if False and mask is not None:
            src, rec, floormap, acumap, mask = self.rotator(src, rec, floormap, acumap, mask)
            src, rec, floormap, acumap, mask = self.translator(src, rec, floormap, acumap, mask)
            return src, rec, floormap, acumap, mask
        elif self.return_rot_angles:
            #src, rec, floormap, acumap, src_pose = self.rotator(src, rec, floormap, acumap, src_pose=src_pose)   # DEBUGGING return to this
            src, rec, floormap, acumap, src_rot_angle = self.rotator(src, rec, floormap, acumap)
            #src_pose = 0.0
            src, rec, floormap, acumap = self.translator(src, rec, floormap, acumap)
            return src, rec, floormap, acumap, src_rot_angle
        else:
            src, rec, floormap, acumap = self.rotator(src, rec, floormap, acumap) 
            src, rec, floormap, acumap = self.translator(src, rec, floormap, acumap) 
            return src, rec, floormap, acumap 


class FloormapAugmentation(nn.Module):
    """ This applies 3 types of acugmentation to the floormap:
    Random displacement
    Fat walls
    Delete walls"""
    def __init__(self, channels_to_augment=[0], kernel_size=(5,5), active_displacement=5, num_kernels=10, stride=1, device='cpu'):
        super(FloormapAugmentation, self).__init__()
        self.active_displacement = active_displacement
        self.kernel_size = kernel_size
        self.channels_to_augment = channels_to_augment
        self.num_kernels = num_kernels
        self.stride = stride
        self.device = device

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        padding = (kernel_size[0] - stride) // 2
        padding = 0
        conv_random_displacement = torch.nn.ConvTranspose2d(in_channels=1, out_channels=self.num_kernels, kernel_size=kernel_size, 
                                                   stride=stride, padding=padding, bias=False).to(self.device)
        self.conv_random_displacement = conv_random_displacement

    def forward_lines(self, x: torch.Tensor):
        # Here we use lines instead of single pixels
        # looks ok, but the image grows too much, or the walls look fat
        if len(x.shape) < 4:
            x = x[None, ...]
            remove_batch = True
        else:
            remove_batch = False

        for i in range(self.num_kernels):

            # Generate a straight line kernel
            random_kernel = torch.zeros(self.kernel_size[0], self.kernel_size[1]).to(self.device)

            # Decide the line orientation (0 for Vertical, 1 for Horizontal, 2 for Diagonal)
            line_orientation = torch.randint(0, 3, (1,), device=self.device)

            # Draw a line on the random_kernel tensor
            if line_orientation == 0:    # Vertical
                line_position_v = torch.randint(0, self.kernel_size[1], (1,), device=self.device)
                random_kernel[:, line_position_v] = 1
            elif line_orientation == 1:  # Horizontal
                line_position_h = torch.randint(0, self.kernel_size[0], (1,), device=self.device)
                random_kernel[line_position_h, :] = 1
            else:                        # Diagonal
                for j in range(min(self.kernel_size)):
                    random_kernel[j, j] = 1

            random_kernel = random_kernel.view(1, 1, self.kernel_size[0], self.kernel_size[1])
            self.conv_random_displacement.weight.data[:, i, :, :] = random_kernel

        output = self.conv_random_displacement(x[..., self.channels_to_augment, :, :])

        rand_indices = torch.randint(low=0, high=self.num_kernels, 
                                    size=(x.shape[0], 1, x.shape[-2], x.shape[-1]), device=self.device)

        output = torch.gather(output, 1, rand_indices)
        output = (output >= 0.5).float()

        x[..., self.channels_to_augment, :, :] = output

        if remove_batch:
            x = x.squeeze(0)
        return x

    def forward(self, x: torch.Tensor):
        if len(x.shape) < 4:
            x = x[None, ...]
            remove_batch = True
        else:
            remove_batch = False
        a = self.kernel_size[0]*self.kernel_size[1]

        for i in range(self.num_kernels):
            random_kernel = torch.cat((torch.ones(self.active_displacement), torch.zeros(a-self.active_displacement)))
            random_kernel = random_kernel[torch.randperm(random_kernel.nelement())].to(self.device)
            random_kernel = random_kernel.view(1,1,self.kernel_size[0],self.kernel_size[1])
            random_kernel = random_kernel / random_kernel.sum()
            self.conv_random_displacement.weight.data[:, i, :, :] = random_kernel
    
        output = self.conv_random_displacement(x[..., self.channels_to_augment, :, :])

        # Generate indices with same shape as target (1,128,128) 
        rand_indices = torch.randint(low=0, high=self.num_kernels, 
                                     size=(x.shape[0],1,x.shape[-2],x.shape[-1]), device=self.device)

        # We need to add an additional dimension so they can be used in gather
        #rand_indices = rand_indices.unsqueeze(0)

        # Gather elements.
        #print(output.shape)
        #print(rand_indices.shape)
        output = torch.gather(output, 1, rand_indices)

        output = (output >= 0.5).float()

        x[..., self.channels_to_augment, :, :] = output

        if remove_batch:
            x = x.squeeze(0)
        return x

        # Now `output` tensor has shape [1, 128, 128] and is obtained by selecting a 
        # random channel from `tensor` for each 'pixel'.

    def forward_gpt(self, x: torch.Tensor):
        if len(x.shape) < 4:
            x = x[None, ...]
        a = self.kernel_size[0]*self.kernel_size[1]

        for i in range(self.num_kernels):
            random_kernel = torch.cat((torch.ones(self.active_displacement), torch.zeros(a-self.active_displacement)))
            random_kernel = random_kernel[torch.randperm(random_kernel.nelement())].to(self.device)
            random_kernel = random_kernel.view(1,1,self.kernel_size[0],self.kernel_size[1])
            random_kernel = random_kernel / random_kernel.sum()
            self.conv_random_displacement.weight.data[i, :, :, :] = random_kernel

        output = self.conv_random_displacement(x[:, self.channels_to_augment, :, :])

        # Create a tensor of random indices in the channel dimension
        random_indices = torch.randint(self.num_kernels, size=(x.size(0), 1, x.size(2), x.size(3))).long().to(self.device)

        # Use advanced indexing to get one channel randomly for each pixel
        output = output[torch.arange(output.size(0))[:,None,None,None], random_indices, :, :]

        # Flatten the output in the channel dimension
        output = output.squeeze(1)

        # Threshold to create binary output
        output = (output >= 0.5).float()

        x_buffer = x.clone()
        print(x.shape)
        print(output.shape)
        x_buffer[:, self.channels_to_augment, :, :] = output

        return x_buffer

    def forward_OLD(self, x: torch.Tensor):
        # Create a random displacement kernel
        # So this is zeros, with a few ones, at random positions
        b = x.shape[0] if len(x.shape) > 3 else 1
        a = self.kernel_size[0] * self.kernel_size[1]

        for i in range(self.num_kernels):
            random_kernel = torch.cat((torch.ones(self.active_displacement), torch.zeros(a-self.active_displacement)))
            random_kernel = random_kernel[torch.randperm(random_kernel.nelement())].to(self.device)
            random_kernel = random_kernel.view(1,1,self.kernel_size[0],self.kernel_size[1])
            random_kernel = random_kernel / random_kernel.sum()
            self.conv_random_displacement.weight.data[i, :, :, :] = random_kernel

        output = self.conv_random_displacement(x[..., self.channels_to_augment, :, :])
        print(output.shape)
        #output = output[torch.randint(self.num_kernels,(len(self.channels_to_augment),)), :, :]  # select a random output 
        #output = torch.mean(output, dim=-3, keepdim=True)

        # Create a tensor of random indices in the channel dimension
        random_indices = torch.randint(self.num_kernels, size=(b, 1, x.size(-2), x.size(-1)), device=self.device).long()

        # Use advanced indexing to get one channel randomly for each pixel
        print(random_indices.shape)
        b = torch.arange(output.size(-3))[:,None,None,None]
        print(b.shape)
        output = output[b, random_indices, :, :]

        # Flatten the output in the channel dimension
        output = output.squeeze(1)

#        output = (output >= 0.5).float()  # binarize

        print(output.shape)
        x[..., self.channels_to_augment, :, :] = output

        return x 
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.conv_random_displacement = self.conv_random_displacement.to(*args, **kwargs)
        return self


class FloormapAugmentation_OLD(nn.Module):
    """ This applies 3 types of acugmentation to the floormap:
    Random displacement
    Fat walls
    Delete walls"""
    def __init__(self, channels_to_augment=[0], kernel_size=(5,5), active_displacement=5, num_kernels=10, device='cpu'):
        super(FloormapAugmentation, self).__init__()
        self.active_displacement = active_displacement
        self.kernel_size = kernel_size
        self.channels_to_augment = channels_to_augment
        self.num_kernels = num_kernels
        self.device = device


        conv_random_displacement = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, 
                                                   stride=1, padding='same', bias=False)
        conv_random_displacement.weight.data = torch.zeros(kernel_size)
        self.conv_random_displacement = conv_random_displacement

    def forward(self, x: torch.Tensor):
                # Create a random displacement kernel
        # So this is zeros, with a few ones, at random positions
        a = self.kernel_size[0] * self.kernel_size[1]
        random_kernel = torch.cat((torch.ones(self.active_displacement), torch.zeros(a-self.active_displacement)))
        random_kernel = random_kernel[torch.randperm(random_kernel.nelement())]
        random_kernel = random_kernel.view(self.kernel_size) / random_kernel.sum()
        self.conv_random_displacement.weight.data = random_kernel[None, None, ...]

        output = self.conv_random_displacement(x[..., self.channels_to_augment, :, :])
        output = (output >= 0.5).float()  # binarize
        print(output.shape)
        x[..., self.channels_to_augment, :, :] = output

        return x 
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.conv_random_displacement = self.conv_random_displacement.to(*args, **kwargs)
        return self


def add_floormap_displacement(floormap: torch.Tensor):
    """Applies random dispalcement to a floormap. The displacement is done with a convotuliton with a random kernel."""
    kernel = torch.random


def test_random_displacement():
    a = torch.rand((1,4,128,128))

    aug = FloormapAugmentation()
    b = aug(a)
    print(b.shape)

if __name__ == '__main__':
    test_random_displacement()
