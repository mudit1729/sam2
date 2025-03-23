# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """
    Object detection head that predicts bounding boxes and classes.
    
    This module takes detection features from the DETR decoder and outputs
    object classes and bounding box coordinates.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 80,
        num_box_params: int = 4,  # (x, y, w, h)
    ):
        """
        Initialize the detection head.
        
        Args:
            hidden_dim: Dimension of the input features
            num_classes: Number of object classes (including background)
            num_box_params: Number of box parameters (4 for x,y,w,h)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_box_params = num_box_params
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Bounding box regression head
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_box_params),
            nn.Sigmoid(),  # Box coordinates are normalized to [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass of the detection head.
        
        Args:
            x: Detection features [B, num_queries, hidden_dim]
            
        Returns:
            Dict with class logits and box predictions
        """
        # Predict class logits
        class_logits = self.class_head(x)
        
        # Predict box coordinates normalized to [0, 1]
        boxes = self.box_head(x)
        
        return {
            "class_logits": class_logits,
            "boxes": boxes,
        }


class DetectionRefinementHead(nn.Module):
    """
    Refinement head for improving small object detection.
    
    This module uses high-resolution features to refine initial object detections,
    especially for small objects that might be missed in the main detection head.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        high_res_dim: int = 64,
        num_classes: int = 80,
        num_box_params: int = 4,
    ):
        """
        Initialize the refinement head.
        
        Args:
            hidden_dim: Dimension of detection features
            high_res_dim: Dimension of high-resolution features
            num_classes: Number of object classes
            num_box_params: Number of box parameters
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.high_res_dim = high_res_dim
        
        # Project high-resolution features to match detection features
        self.high_res_proj = nn.Conv2d(high_res_dim, hidden_dim, kernel_size=1)
        
        # RoI feature extraction for refinement
        self.roi_align = nn.ModuleList([
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        ])
        
        # Refinement heads
        self.class_refine = nn.Linear(hidden_dim, num_classes)
        self.box_refine = nn.Linear(hidden_dim, num_box_params)
    
    def forward(
        self,
        detection_features: torch.Tensor,
        high_res_features: torch.Tensor,
        init_boxes: torch.Tensor,
        image_size: tuple,
    ) -> dict:
        """
        Forward pass of the refinement head.
        
        Args:
            detection_features: Detection features [B, num_queries, hidden_dim]
            high_res_features: High-resolution features [B, C, H, W]
            init_boxes: Initial box predictions [B, num_queries, 4] (x,y,w,h format, normalized)
            image_size: Original image size (H, W) for de-normalizing boxes
            
        Returns:
            Dict with refined class logits and box predictions
        """
        batch_size, num_queries = init_boxes.shape[:2]
        H, W = image_size
        
        # Project high-resolution features
        high_res_proj = self.high_res_proj(high_res_features)
        
        # Extract RoI features for each box
        boxes_unnorm = init_boxes.clone()
        boxes_unnorm[..., 0] *= W  # x
        boxes_unnorm[..., 1] *= H  # y
        boxes_unnorm[..., 2] *= W  # width
        boxes_unnorm[..., 3] *= H  # height
        
        # Convert x,y,w,h to x1,y1,x2,y2 format for ROI align
        roi_boxes = torch.zeros_like(boxes_unnorm)
        roi_boxes[..., 0] = boxes_unnorm[..., 0] - boxes_unnorm[..., 2] / 2  # x1
        roi_boxes[..., 1] = boxes_unnorm[..., 1] - boxes_unnorm[..., 3] / 2  # y1
        roi_boxes[..., 2] = boxes_unnorm[..., 0] + boxes_unnorm[..., 2] / 2  # x2
        roi_boxes[..., 3] = boxes_unnorm[..., 1] + boxes_unnorm[..., 3] / 2  # y2
        
        # Extract and process ROI features
        roi_features = roi_boxes  # This is a simplification - real implementation would use torchvision.ops.roi_align
        
        # Apply refinement heads
        refined_classes = self.class_refine(detection_features)
        refined_boxes = self.box_refine(detection_features) + init_boxes  # Residual connection
        
        # Normalize refined boxes to [0, 1]
        refined_boxes = torch.sigmoid(refined_boxes)
        
        return {
            "refined_class_logits": refined_classes,
            "refined_boxes": refined_boxes,
        }