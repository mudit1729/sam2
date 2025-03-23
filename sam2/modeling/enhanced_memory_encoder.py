# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d


class DetectionProcessor(nn.Module):
    """
    Process detection results into a format suitable for memory encoding.
    """
    def __init__(
        self,
        hidden_dim=256,
        output_dim=256,
        max_detections=100,
    ):
        super().__init__()
        
        # Project detection features to the right dimension
        self.detection_proj = nn.Linear(hidden_dim, output_dim)
        
        # For combining detection features with spatial information
        self.pos_embed = nn.Linear(4, output_dim)  # Box coordinates (normalized)
        
        self.max_detections = max_detections
        self.output_dim = output_dim
        
    def forward(self, detections):
        """
        Process detection results to create detection features.
        """
        pred_boxes = detections['pred_boxes']  # [batch_size, num_detections, 4]
        pred_logits = detections['pred_logits']  # [batch_size, num_detections, num_classes+1]
        obj_scores = detections['obj_scores']  # [batch_size, num_detections, 1]
        
        batch_size, num_detections = pred_boxes.shape[:2]
        
        # Limit number of detections if needed
        if num_detections > self.max_detections:
            # Sort by object score and keep top-k
            _, indices = torch.sort(obj_scores.squeeze(-1), dim=1, descending=True)
            indices = indices[:, :self.max_detections]
            batch_indices = torch.arange(batch_size, device=pred_boxes.device).unsqueeze(1)
            batch_indices = batch_indices.expand(-1, self.max_detections)
            
            pred_boxes = pred_boxes[batch_indices, indices]
            pred_logits = pred_logits[batch_indices, indices]
            obj_scores = obj_scores[batch_indices, indices]
            num_detections = self.max_detections
        
        # Convert class logits to probabilities
        class_probs = F.softmax(pred_logits, dim=-1)
        
        # Process box coordinates - create positional embeddings
        box_pos_embed = self.pos_embed(pred_boxes)
        
        # Project detection features
        detection_features = self.detection_proj(class_probs)
        
        # Combine with positional information
        detection_features = detection_features + box_pos_embed
        
        # Weight by object confidence
        detection_features = detection_features * obj_scores
        
        return detection_features


class EnhancedMemoryEncoder(nn.Module):
    """
    Enhanced memory encoder that combines patch embeddings with detection results.
    """
    def __init__(
        self,
        out_dim,  # Output dimension of memory vectors
        patch_encoder,  # The original memory encoder for processing patch embeddings
        position_encoding,  # Position encoding module
        detection_processor,  # Processor for detection results
        in_dim=256,  # Input dimension of patch features
        fusion_type='concat',  # How to fuse detections with patch embeddings: 'concat', 'add', or 'attention'
    ):
        super().__init__()
        
        # For processing patch embeddings
        self.patch_encoder = patch_encoder
        
        # For processing detection results
        self.detection_processor = detection_processor
        
        # Position encoding for the memory vectors
        self.position_encoding = position_encoding
        
        # Fusion type: how to combine patch embeddings with detections
        self.fusion_type = fusion_type
        
        # For fusing patch features with detection features
        if fusion_type == 'concat':
            # Concatenate and project
            self.fusion_layer = nn.Linear(in_dim + detection_processor.output_dim, out_dim)
        elif fusion_type == 'add':
            # Project to same dimension and add
            self.patch_proj = nn.Linear(in_dim, out_dim)
            self.detection_proj = nn.Linear(detection_processor.output_dim, out_dim)
        elif fusion_type == 'attention':
            # Use attention to fuse
            self.attn = nn.MultiheadAttention(in_dim, 8, batch_first=True)
            self.norm = nn.LayerNorm(in_dim)
            self.detection_proj = nn.Linear(detection_processor.output_dim, in_dim)
            self.output_proj = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
            
        # Final memory vector generation
        self.memory_generator = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        
        self.out_dim = out_dim

    def forward(
        self,
        patch_embeddings: torch.Tensor,  # Patch embeddings from encoder [B, C, H, W]
        detections: Dict[str, torch.Tensor],  # Detection results from detection head
        prev_memory_vector: Optional[torch.Tensor] = None,  # Previous memory vector (optional)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the enhanced memory encoder.
        
        Args:
            patch_embeddings: Patch embeddings from the encoder
            detections: Detection results from detection head
            prev_memory_vector: Optional previous memory vector for temporal consistency
            
        Returns:
            Dictionary containing memory vector and positional encoding
        """
        batch_size = patch_embeddings.shape[0]
        device = patch_embeddings.device
        
        # Process patch embeddings
        # We're assuming patch_embeddings are already processed appropriately
        patch_features = patch_embeddings  # [B, C, H, W]
        
        # Process detection results
        detection_features = self.detection_processor(detections)  # [B, num_detections, C]
        
        # Fuse patch features with detection features
        if self.fusion_type == 'concat':
            # Average pool patch features to get a single vector per batch
            pooled_patch_features = F.adaptive_avg_pool2d(patch_features, 1).squeeze(-1).squeeze(-1)
            # Average pool detection features to get a single vector per batch
            pooled_detection_features = detection_features.mean(dim=1)
            # Concatenate and project
            combined_features = torch.cat([pooled_patch_features, pooled_detection_features], dim=1)
            memory_features = self.fusion_layer(combined_features)
            
        elif self.fusion_type == 'add':
            # Average pool patch features to get a single vector per batch
            pooled_patch_features = F.adaptive_avg_pool2d(patch_features, 1).squeeze(-1).squeeze(-1)
            # Average pool detection features to get a single vector per batch
            pooled_detection_features = detection_features.mean(dim=1)
            # Project to same dimension and add
            memory_features = self.patch_proj(pooled_patch_features) + self.detection_proj(pooled_detection_features)
            
        elif self.fusion_type == 'attention':
            # Reshape patch features to sequence
            h, w = patch_features.shape[2], patch_features.shape[3]
            patch_seq = patch_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            # Project detection features
            detection_seq = self.detection_proj(detection_features)  # [B, num_detections, C]
            # Concatenate sequences
            combined_seq = torch.cat([patch_seq, detection_seq], dim=1)  # [B, H*W+num_detections, C]
            # Apply self-attention
            combined_seq = self.norm(combined_seq)
            combined_seq = combined_seq + self.attn(combined_seq, combined_seq, combined_seq)[0]
            # Mean pool to get memory features
            memory_features = combined_seq.mean(dim=1)  # [B, C]
            memory_features = self.output_proj(memory_features)  # [B, out_dim]
        
        # Add previous memory vector if provided (temporal consistency)
        if prev_memory_vector is not None:
            memory_features = memory_features + 0.1 * prev_memory_vector
        
        # Final memory vector generation
        memory_vector = self.memory_generator(memory_features)  # [B, out_dim]
        
        # Generate positional encoding for the memory vector
        memory_pos_enc = torch.zeros(batch_size, self.out_dim, device=device)
        
        return {
            "memory_vector": memory_vector,
            "memory_pos_enc": memory_pos_enc,
        }