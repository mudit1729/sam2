# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple multi-layer perceptron with ReLU activations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ObjectDetectionHead(nn.Module):
    """
    Detection head for DETR-style object detection.
    Takes decoder outputs and predicts bounding boxes and class labels.
    """
    def __init__(
        self,
        hidden_dim=256,
        num_classes=1,  # Default for tracking is 1 class
        num_box_params=4,  # [x, y, w, h]
        num_layers=3,
    ):
        super().__init__()
        
        # Class prediction head
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background class
        
        # Bounding box prediction head
        self.bbox_embed = MLP(hidden_dim, hidden_dim, num_box_params, num_layers)
        
        # Object presence/confidence score head
        self.obj_score_embed = nn.Linear(hidden_dim, 1)
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, decoder_output):
        """
        Args:
            decoder_output: output from the DETR decoder [num_queries, batch_size, hidden_dim]
            
        Returns:
            dict containing class predictions, box predictions, and object scores
        """
        # Transpose if needed: [num_queries, batch_size, hidden_dim] -> [batch_size, num_queries, hidden_dim]
        if decoder_output.shape[1] != decoder_output.shape[0]:
            decoder_output = decoder_output.permute(1, 0, 2)
        
        # Predict classes, boxes, and object confidence scores
        output_class = self.class_embed(decoder_output)  # [batch_size, num_queries, num_classes+1]
        output_bbox = self.bbox_embed(decoder_output).sigmoid()  # [batch_size, num_queries, 4], normalized to [0,1]
        output_obj_scores = self.obj_score_embed(decoder_output)  # [batch_size, num_queries, 1]
        
        return {
            'pred_logits': output_class,
            'pred_boxes': output_bbox,
            'obj_scores': output_obj_scores,
        }


class ObjectRefinementHead(nn.Module):
    """
    Refinement head for small objects.
    Takes initial detections and high-resolution features to improve small object detection.
    """
    def __init__(
        self,
        hidden_dim=256,
        small_object_threshold=0.05,  # Relative size threshold to define small objects
        refinement_factor=2.0,  # How much to refine small object features
    ):
        super().__init__()
        
        # Feature refinement layers
        self.refine_features = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Double input: detections + high-res features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Box refinement for small objects
        self.box_refine = MLP(hidden_dim, hidden_dim, 4, 2)
        
        self.small_object_threshold = small_object_threshold
        self.refinement_factor = refinement_factor
        
    def forward(self, detections, high_res_vectors, prev_memory_vector=None):
        """
        Args:
            detections: output from the object detection head
            high_res_vectors: high-resolution features for refinement
            prev_memory_vector: optional previous memory vector for temporal consistency
            
        Returns:
            refined detections
        """
        pred_boxes = detections['pred_boxes']  # [batch_size, num_queries, 4]
        pred_logits = detections['pred_logits']  # [batch_size, num_queries, num_classes+1]
        obj_scores = detections['obj_scores']  # [batch_size, num_queries, 1]
        
        # Identify small objects based on area (width * height)
        box_areas = pred_boxes[..., 2] * pred_boxes[..., 3]  # [batch_size, num_queries]
        is_small_object = box_areas < self.small_object_threshold  # [batch_size, num_queries]
        is_small_object = is_small_object.unsqueeze(-1)  # [batch_size, num_queries, 1]
        
        # Extract high-res features for each detection
        # This is a simplified version - in practice, you'd use ROI pooling/align
        # to extract features at the specific box locations
        batch_size, num_queries = pred_boxes.shape[:2]
        
        # For simplicity, we'll assume high_res_vectors is already aligned with detections
        # In practice, you would implement ROI pooling/align here
        
        # Concatenate detection features with high-res features
        if high_res_vectors.shape[0] != batch_size or high_res_vectors.shape[1] != num_queries:
            # Reshape high_res_vectors to match detections shape if needed
            high_res_vectors = high_res_vectors.view(batch_size, num_queries, -1)
            if high_res_vectors.shape[2] != pred_logits.shape[2]:
                # Adapt dimensions if needed (simplified)
                high_res_vectors = F.adaptive_avg_pool1d(
                    high_res_vectors.transpose(1, 2), pred_logits.shape[2]
                ).transpose(1, 2)
        
        # Concatenate along feature dimension
        combined_features = torch.cat([pred_logits, high_res_vectors], dim=-1)
        
        # Apply refinement
        refined_features = self.refine_features(combined_features)
        
        # Refine boxes for small objects
        box_refinements = self.box_refine(refined_features).sigmoid()
        
        # Only apply refinements to small objects
        refined_boxes = torch.where(
            is_small_object.expand_as(pred_boxes),
            pred_boxes * (1.0 + self.refinement_factor * (box_refinements - 0.5)),  # Apply refinement with scaling
            pred_boxes  # Keep original boxes for non-small objects
        )
        
        # Ensure boxes stay within [0, 1] range
        refined_boxes = torch.clamp(refined_boxes, 0.0, 1.0)
        
        # Return refined detections
        return {
            'pred_logits': pred_logits,  # Keep original class predictions
            'pred_boxes': refined_boxes,  # Update with refined boxes
            'obj_scores': obj_scores,     # Keep original scores
        }