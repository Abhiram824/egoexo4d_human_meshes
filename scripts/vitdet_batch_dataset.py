"""
Batched variant of hamer's ViTDetDataset (hamer/hamer/datasets/vitdet_dataset.py):
crops one hand per item from bbox coordinates, but takes a list of per-item
images/paths instead of a single shared image, so it can back a DataLoader
across many frames at once (used by scripts/run_batched_hamer.py). Kept here
rather than inside the vendored hamer/ clone so hamer/ can stay an unmodified
mirror of upstream geopavlakos/hamer.

Images are read lazily per __getitem__ call rather than pre-loaded into
memory upfront -- with 002-006 track_preds worth of hand crops across an
18k-frame cooking take (~14GB/camera), eager loading was the original cause
of stage-2 OOMs.
"""
from typing import Dict, List

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from hamer.datasets.utils import (convert_cvimg_to_tensor,
                                   expand_to_aspect_ratio,
                                   generate_image_patch_cv2)


class ViTDetBatchDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 imgs_cv2: List[np.array],
                 boxes: np.array,
                 right: np.array,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        # each entry is either a path-like (lazy-loaded below) or an
        # already-decoded np.array; never pre-read here
        self.imgs_cv2 = imgs_cv2

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # boxes are [x1, y1, x2, y2] per item; convert to center+scale, and
        # pad by rescale_factor so the crop has margin around the hand
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        # HaMeR's crops are always right-hand convention; left hands are
        # flipped into that space here and un-flipped later by the caller
        right = self.right[idx].copy()
        flip = right == 0

        # lazy load: read from disk only when this item is actually fetched
        _raw = self.imgs_cv2[idx]
        cvimg = cv2.imread(str(_raw)) if not isinstance(_raw, np.ndarray) else _raw.copy()
        if True:
            # Blur image to avoid aliasing artifacts when the crop is
            # downsampled substantially to the model's fixed patch size
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = self.right[idx].copy()
        return item
