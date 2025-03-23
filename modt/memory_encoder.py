# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMemoryEncoder(nn.Module):
    """
    Memory encoder that combines detections with patch embeddings to create memory vectors.
    
    This module encodes detection results into memory representations that can be 
    used for tracking objects across frames.
    """
    
    def __init__(
        self,
        embed_dim: int,
        memory_dim: int = None,
        detection_dim: int = 256,
        max_objects: int = 20,
        use_box_features: bool = True,
        use_confidence: bool = True,
    ):
        """
        Initialize the memory encoder.
        
        Args:
            embed_dim: Dimension of the input patch embeddings
            memory_dim: Dimension of output memory vectors (defaults to embed_dim)
            detection_dim: Dimension of detection features
            max_objects: Maximum number of objects to track
            use_box_features: Whether to use bounding box features
            use_confidence: Whether to use detection confidence
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim if memory_dim is not None else embed_dim
        self.detection_dim = detection_dim
        self.max_objects = max_objects
        self.use_box_features = use_box_features
        self.use_confidence = use_confidence
        
        # Feature dimension for combined input
        box_feat_dim = 4 if use_box_features else 0
        conf_dim = 1 if use_confidence else 0
        
        # Encoder to combine detection features with patch embeddings
        self.encoder = nn.Sequential(
            nn.Linear(detection_dim + box_feat_dim + conf_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim),
        )
        
        # Attention for selecting relevant patch embeddings
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=detection_dim,
            num_heads=8,
            dropout=0.1,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=True,
        )
        
        # Norms for inputs
        self.norm_patches = nn.LayerNorm(embed_dim)
        self.norm_detections = nn.LayerNorm(detection_dim)
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        detection_features: torch.Tensor,
        boxes: torch.Tensor = None,
        confidence: torch.Tensor = None,
        image_size: tuple = None,
    ) -> torch.Tensor:
        """
        Forward pass of the memory encoder.
        
        Args:
            patch_embeddings: Patch embeddings from the vision transformer [B, N_p, C_p]
            detection_features: Features from object detection [B, N_o, C_d]
            boxes: Bounding boxes [B, N_o, 4] in (x, y, w, h) format
            confidence: Detection confidence scores [B, N_o, 1]
            image_size: Image size (H, W) for normalizing boxes
            
        Returns:
            Memory vectors [B, N_o, C_m] for tracked objects
        """
        batch_size, num_objects = detection_features.shape[:2]
        
        # Normalize inputs
        patches_norm = self.norm_patches(patch_embeddings)
        detections_norm = self.norm_detections(detection_features)
        
        # Attend to relevant patch embeddings based on detection features
        attn_output, _ = self.patch_attention(
            query=detections_norm,
            key=patches_norm,
            value=patches_norm,
        )
        
        # Format inputs for the encoder
        encoder_inputs = [detection_features]
        
        # Add box features if enabled
        if self.use_box_features and boxes is not None:
            if image_size is not None:
                # Normalize box coordinates to [0, 1]
                h, w = image_size
                norm_boxes = boxes.clone()
                norm_boxes[..., 0] /= w  # x
                norm_boxes[..., 1] /= h  # y
                norm_boxes[..., 2] /= w  # width
                norm_boxes[..., 3] /= h  # height
                encoder_inputs.append(norm_boxes)
            else:
                encoder_inputs.append(boxes)
        
        # Add confidence if enabled
        if self.use_confidence and confidence is not None:
            encoder_inputs.append(confidence)
        
        # Combine inputs
        combined_input = torch.cat(encoder_inputs, dim=-1)
        
        # Encode to memory vectors
        memory_vectors = self.encoder(combined_input)
        
        return memory_vectors