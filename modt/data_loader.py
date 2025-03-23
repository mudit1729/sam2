# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Callable

class NuImagesTrackingDataset(Dataset):
    """
    Dataset for loading nuImages data for memory-based object tracking.
    
    This dataset loads clips (sequences of images) from the nuImages dataset,
    with each sample containing a keyframe and its associated frames.
    """
    
    def __init__(
        self,
        nuim,  # NuImages instance
        dataroot: str,
        clip_length: int = 13,
        transform: Optional[Callable] = None,
        return_tensors: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            nuim: NuImages instance (initialized from nuimages.NuImages)
            dataroot: Root directory where image files are stored
            clip_length: Maximum number of frames in each clip
            transform: Optional transform to apply to images
            return_tensors: Whether to return PyTorch tensors (if False, returns PIL images)
        """
        self.nuim = nuim
        self.dataroot = dataroot
        self.clip_length = clip_length
        self.transform = transform
        self.return_tensors = return_tensors
        
        # Build a mapping from sample tokens to their sample_data records.
        self.sample_to_sample_data = {}
        for sample in self.nuim.sample:
            sample_token = sample['token']
            self.sample_to_sample_data[sample_token] = []
        
        # Populate the mapping.
        for sd in self.nuim.sample_data:
            sample_token = sd['sample_token']
            if sample_token in self.sample_to_sample_data:
                self.sample_to_sample_data[sample_token].append(sd)
        
        # Store sample tokens as a list for indexing.
        self.sample_tokens = list(self.sample_to_sample_data.keys())
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a clip (sequence of images) and its annotations.
        
        Args:
            idx: Dataset index
            
        Returns:
            Dict containing clip images, annotations, and metadata
        """
        sample_token = self.sample_tokens[idx]
        sample = self.nuim.get('sample', sample_token)
        
        # Get all sample_data records for this sample.
        sample_datas = self.sample_to_sample_data[sample_token]
        
        # Sort by the absolute time difference from the keyframe.
        key_timestamp = sample['timestamp']
        sample_datas.sort(key=lambda sd: abs(sd['timestamp'] - key_timestamp))
        
        # Limit to clip_length.
        sample_datas = sample_datas[:self.clip_length]
        
        # Find the keyframe camera token (usually the first in the sorted list).
        key_camera_token = sample_datas[0]['token'] if sample_datas else None
        
        # Load each image in the clip. We assume each sample_data record has a 'filename' key.
        clip_images = []
        for sd in sample_datas:
            img_path = os.path.join(self.dataroot, sd['filename'])
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                raise FileNotFoundError(f"Could not open image {img_path}: {e}")
            
            if self.transform:
                image = self.transform(image)
            
            # Convert to tensor if requested
            if self.return_tensors and not isinstance(image, torch.Tensor):
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)
                
            clip_images.append(image)
        
        # Get annotations for the keyframe.
        # nuim.list_anns returns a tuple: (object_tokens, surface_tokens).
        object_tokens, surface_tokens = self.nuim.list_anns(sample_token)
        object_annotations = [self.nuim.get('object_ann', token) for token in object_tokens]
        surface_annotations = [self.nuim.get('surface_ann', token) for token in surface_tokens]
        
        annotations = {
            'object': object_annotations,
            'surface': surface_annotations
        }
        
        # Include additional metadata
        meta = {
            'sample_token': sample_token,
            'key_camera_token': key_camera_token,
            'timestamps': [sd['timestamp'] for sd in sample_datas]
        }
        
        return {'clip': clip_images, 'annotations': annotations, 'meta': meta}


def create_nuimages_dataloader(
    nuim,
    dataroot: str,
    batch_size: int = 4,
    clip_length: int = 13,
    transform: Optional[Callable] = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for the nuImages dataset.
    
    Args:
        nuim: NuImages instance
        dataroot: Root directory where image files are stored
        batch_size: Batch size for DataLoader
        clip_length: Maximum number of frames in each clip
        transform: Optional transform to apply to images
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = NuImagesTrackingDataset(
        nuim,
        dataroot=dataroot,
        clip_length=clip_length,
        transform=transform,
    )
    
    # Define a custom collate function to handle variable-length clips
    def collate_fn(batch):
        clips = [item['clip'] for item in batch]
        annotations = [item['annotations'] for item in batch]
        meta = [item['meta'] for item in batch]
        
        return {
            'clip': clips,
            'annotations': annotations,
            'meta': meta
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return dataloader