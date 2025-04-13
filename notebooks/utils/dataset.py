from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

class xView2Dataset(Dataset):
    '''
    A custom PyTorch Dataset class designed for the xView2 dataset, supporting both training (with target masks)
    and inference (without masks).
    The dataset expects pre-disaster and post-disaster image pairs, 
    with optional corresponding target masks.
    '''
    def __init__(self,
                 png_path: str,
                 target_path: callable = None,
                 transform: callable = None,
                 image_transform: callable = None,
                 inference = False):

        
        self.png_path = png_path
        self.target_path = target_path
        self.transform = transform
        self.image_transform = image_transform
        self.inference = inference

        

        # get all pre-disaster images:
        self.pre_images = sorted(self.png_path.glob("*_pre_disaster.png"))
        
        self.pairs = [] #

        for pre_img_path in self.pre_images:
            post_img_path = self.png_path / pre_img_path.name.replace("_pre_disaster", "_post_disaster")

            if self.inference: 
                if post_img_path.exists():
                    self.pairs.append((pre_img_path, post_img_path))
            else: 
                # Target path only for training and validation not for inference
                if self.target_path is None:
                    raise ValueError("target_path must be provided when not in inference mode")
                    
                post_target_path = self.target_path / pre_img_path.name.replace("_pre_disaster", "_post_disaster")
                pre_target_path = self.target_path / pre_img_path.name

                if post_img_path.exists() and post_target_path.exists() and pre_target_path.exists():
                    self.pairs.append((pre_img_path, post_img_path, pre_target_path, post_target_path))

        assert len(self.pairs) > 0, "No matching image-pairs found!"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        if self.inference:
            pre_img_path, post_img_path = self.pairs[index]

            # Load images
            pre_img = Image.open(pre_img_path).convert("RGB")
            post_img = Image.open(post_img_path).convert("RGB")

            # Convert to numpy arrays
            pre_img = np.array(pre_img, dtype=np.float32) / 255.0
            post_img = np.array(post_img, dtype=np.float32) / 255.0

            # Convert to Tensor
            pre_img = torch.tensor(pre_img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
            post_img = torch.tensor(post_img).permute(2, 0, 1)
            
            if self.image_transform:
                pre_img = self.image_transform(pre_img)
                post_img = self.image_transform(post_img)

            return pre_img, post_img, pre_img_path.name, post_img_path.name

        else:
            pre_img_path, post_img_path, pre_target_path, post_target_path = self.pairs[index]

            # load images and target masks with 
            
            pre_img = Image.open(pre_img_path).convert("RGB")
            post_img = Image.open(post_img_path).convert("RGB")
            pre_target_mask = Image.open(pre_target_path).convert('L')
            post_target_mask = Image.open(post_target_path).convert('L')

            # convert to numpy arrays
            pre_img = np.array(pre_img, dtype=np.float32) / 255.0
            post_img = np.array(post_img, dtype=np.float32) / 255.0
            pre_target_mask = np.array(pre_target_mask, dtype=np.float32)
            post_target_mask = np.array(post_target_mask, dtype=np.float32)

            # convert to Tensor
            pre_img = torch.tensor(pre_img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
            post_img = torch.tensor(post_img).permute(2, 0, 1)
            pre_target_mask = torch.tensor(pre_target_mask).unsqueeze(0)  # (H, W) → (1, H, W)
            post_target_mask = torch.tensor(post_target_mask).unsqueeze(0)

            # Transformation (optional)
            if self.transform:
                stack = torch.cat([pre_img, post_img, pre_target_mask, post_target_mask], dim=0)  # (8, H, W)
                stack = self.transform(stack)

                pre_img, post_img, pre_target_mask, post_target_mask = stack[:3], stack[3:6], stack[6:7], stack[7:8]
            
            if self.image_transform:
                # Normalization only for images and not the masks
                pre_img = self.image_transform(pre_img)
                post_img = self.image_transform(post_img)

            return pre_img, post_img, pre_target_mask, post_target_mask 

    
def collate_fn(batch):
      
    pre_imgs, post_imgs, pre_masks, post_masks = zip(*batch)

    # Stapeln der Tensoren entlang der Batch-Dimension (erste Dimension)
    pre_imgs = torch.stack(pre_imgs, dim=0)
    post_imgs = torch.stack(post_imgs, dim=0)
    pre_masks = torch.stack(pre_masks, dim=0)
    post_masks = torch.stack(post_masks, dim=0)

    return pre_imgs, post_imgs, pre_masks, post_masks

    
def collate_fn_test(batch):
    pre_imgs, post_imgs, pre_names, post_names = zip(*batch)
    # Stapeln der Tensoren entlang der Batch-Dimension (erste Dimension)
    pre_imgs = torch.stack(pre_imgs, dim=0)
    post_imgs = torch.stack(post_imgs, dim=0)

    return pre_imgs, post_imgs, pre_names, post_names
def transform():
    """Transform für Bilder & Masken"""
    return v2.Compose([
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomVerticalFlip(p = 0.5),
        #v2.RandomRotation(degrees=15),
       # v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.ToDtype(torch.float32, scale=True)  # Automatische Skalierung auf [0,1]
    ])

def image_transform():
    """Nur für RGB-Bilder"""
    return v2.Compose([
       # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

