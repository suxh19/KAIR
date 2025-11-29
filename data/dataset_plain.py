import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from typing import Dict, Any, List, Optional, Union, cast


class DatasetPlain(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt: Dict[str, Any]):
        super(DatasetPlain, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt: Dict[str, Any] = opt
        self.n_channels: int = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size: int = self.opt['H_size'] if self.opt['H_size'] else 64

        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        paths_H: Optional[List[str]] = util.get_image_paths(opt['dataroot_H'])
        paths_L: Optional[List[str]] = util.get_image_paths(opt['dataroot_L'])
        
        assert paths_H is not None, 'Error: H path is empty.'
        assert paths_L is not None, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        
        self.paths_H: List[str] = paths_H
        self.paths_L: List[str] = paths_L
        
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index: int) -> Dict[str, Any]:

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self) -> int:
        return len(self.paths_H)
