# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .memory_attention import EnhancedMemoryAttention
from .memory_encoder import EnhancedMemoryEncoder
from .detr_decoder import DETRDecoder
from .detection_head import DetectionHead, DetectionRefinementHead

class SAM2EnhancedModel(nn.Module):
    """
    Enhanced SAM2 base model integrating memory-based tracking components.
    
    This model extends SAM2 with memory feedback and object tracking capabilities.
    """
    
    def __init__(
        self,
        image_encoder,  # SAM2's vision transformer encoder
        mask_decoder,   # SAM2's mask decoder
        embed_dim: int = 256,
        memory_dim: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        num_classes: int = 80,
        use_refinement: bool = True,
    ):
        """
        Initialize the enhanced SAM2 model.
        
        Args:
            image_encoder: SAM2's vision transformer encoder
            mask_decoder: SAM2's mask decoder
            embed_dim: Embedding dimension
            memory_dim: Memory dimension
            num_heads: Number of attention heads
            num_decoder_layers: Number of decoder layers
            num_queries: Number of object queries
            num_classes: Number of object classes
            use_refinement: Whether to use detection refinement
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.num_classes = num_classes
        self.use_refinement = use_refinement
        
        # Get feature dimensions from the image encoder
        self.patch_embed_dim = image_encoder.patch_embed_dim
        self.high_res_embed_dim = 64  # From low-level features
        
        # Memory attention module
        self.memory_attention = EnhancedMemoryAttention(
            embed_dim=self.patch_embed_dim,
            num_heads=num_heads,
            memory_dim=memory_dim,
            dropout=0.1,
        )
        
        # DETR decoder
        self.detr_decoder = DETRDecoder(
            embed_dim=self.patch_embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=0.1,
            num_queries=num_queries,
            hidden_dim=embed_dim,
        )
        
        # Detection head
        self.detection_head = DetectionHead(
            hidden_dim=embed_dim,
            num_classes=num_classes,
            num_box_params=4,
        )
        
        # Detection refinement head (optional)
        if use_refinement:
            self.detection_refinement = DetectionRefinementHead(
                hidden_dim=embed_dim,
                high_res_dim=self.high_res_embed_dim,
                num_classes=num_classes,
                num_box_params=4,
            )
        
        # Memory encoder
        self.memory_encoder = EnhancedMemoryEncoder(
            embed_dim=self.patch_embed_dim,
            memory_dim=memory_dim,
            detection_dim=embed_dim,
            use_box_features=True,
            use_confidence=True,
        )
    
    def forward(
        self,
        image: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tracking_state: Optional[Dict] = None,
        image_size: Optional[Tuple[int, int]] = None,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass of the enhanced SAM2 model.
        
        Args:
            image: Input image [B, 3, H, W]
            memory: Memory vectors from previous frames [B, M, memory_dim] (optional)
            tracking_state: Current tracking state (optional)
            image_size: Original image size (H, W) (optional)
            points: Prompt points [B, N, 2] (optional)
            boxes: Prompt boxes [B, N, 4] (optional)
            masks: Prompt masks [B, 1, H, W] (optional)
            
        Returns:
            Dict with detection and segmentation results
        """
        # Extract image features
        try:
            features = self.image_encoder(image)
            # DINOv2 returns patch_features directly (not in a dict)
            if isinstance(features, torch.Tensor):
                patch_features = features  # [B, N, C]
                high_res_features = None   # Not available in this case
            else:
                # Extract from dictionary
                patch_features = features["patch_features"]  # [B, N, C]
                high_res_features = features.get("high_res_features", None)  # [B, C, H, W]
        except Exception as e:
            # Log the error for debugging
            print(f"Error in image_encoder: {e}")
            print(f"Image shape: {image.shape}")
            if isinstance(self.image_encoder, torch.nn.Module):
                print(f"Encoder type: {type(self.image_encoder)}")
            raise
        
        # Apply memory attention if memory is provided
        if memory is not None and memory.shape[1] > 0:
            memory_enhanced_features = self.memory_attention(patch_features, memory)
        else:
            memory_enhanced_features = patch_features
        
        # Apply DETR decoder to get detection features
        # Ensure the input format is correct - DETR decoder expects either a tensor
        # or a dict with 'patch_features' key
        if isinstance(memory_enhanced_features, torch.Tensor):
            # Wrap in a dict to match the expected format
            detection_features = self.detr_decoder({"patch_features": memory_enhanced_features})
        else:
            detection_features = self.detr_decoder(memory_enhanced_features)
        
        # Generate detections
        detections = self.detection_head(detection_features)
        
        # Refine detections if enabled
        if self.use_refinement and high_res_features is not None:
            if image_size is None:
                # Estimate image size from features
                h, w = high_res_features.shape[2:4]
                image_size = (h * 4, w * 4)  # Assuming 4x downsampling
            
            refined = self.detection_refinement(
                detection_features=detection_features,
                high_res_features=high_res_features,
                init_boxes=detections["boxes"],
                image_size=image_size,
            )
            detections.update(refined)
        
        # Generate masks if prompt points or boxes are provided
        if points is not None or boxes is not None or masks is not None:
            # Use SAM2's mask decoder to generate masks
            mask_results = self.mask_decoder(
                image_embeddings=patch_features,
                point_coords=points,
                point_labels=None,  # Foreground by default
                boxes=boxes,
                mask_input=masks,
            )
            detections.update(mask_results)
        
        # Generate memory vectors for detected objects
        class_probs = F.softmax(detections["class_logits"], dim=-1)
        bg_prob = class_probs[..., 0]  # Assuming class 0 is background
        confidence = 1.0 - bg_prob.unsqueeze(-1)
        
        memory_vectors = self.memory_encoder(
            patch_embeddings=patch_features,
            detection_features=detection_features,
            boxes=detections["boxes"] if "refined_boxes" not in detections else detections["refined_boxes"],
            confidence=confidence,
            image_size=image_size,
        )
        
        detections["memory_vectors"] = memory_vectors
        
        return detections