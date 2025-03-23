# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class ObjectTracker(nn.Module):
    """
    Object tracker module for video sequences.
    
    This module maintains object states across frames and handles object tracking
    using the memory-based mechanism.
    """
    
    def __init__(
        self,
        memory_dim: int = 256,
        max_objects: int = 20,
        similarity_threshold: float = 0.7,
        max_age: int = 10,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize the object tracker.
        
        Args:
            memory_dim: Dimension of memory vectors
            max_objects: Maximum number of objects to track
            similarity_threshold: Threshold for feature similarity matching
            max_age: Maximum frames an object can be missing before termination
            min_hits: Minimum number of detections before track is established
            iou_threshold: IoU threshold for box matching
        """
        super().__init__()
        self.memory_dim = memory_dim
        self.max_objects = max_objects
        self.similarity_threshold = similarity_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Initialize memory bank (batch_size, max_objects, memory_dim)
        self.register_buffer("empty_memory", torch.zeros(1, 1, memory_dim))
    
    def initialize_tracks(self, batch_size: int = 1) -> Dict:
        """
        Initialize empty tracking state.
        
        Args:
            batch_size: Batch size for tracking state
            
        Returns:
            Dict with initialized tracking state
        """
        device = self.empty_memory.device
        
        # Create empty tracking state
        tracking_state = {
            "memory": torch.zeros(batch_size, 0, self.memory_dim, device=device),
            "boxes": torch.zeros(batch_size, 0, 4, device=device),
            "obj_ids": torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            "active": torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            "age": torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            "hits": torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            "next_obj_id": torch.ones(batch_size, dtype=torch.long, device=device),
        }
        
        return tracking_state
    
    def update(self, tracking_state: Dict, detections: Dict) -> Dict:
        """
        Update tracking state with new detections.
        
        Args:
            tracking_state: Current tracking state
            detections: New detections (class_logits, boxes, memory_vectors)
            
        Returns:
            Updated tracking state
        """
        batch_size = tracking_state["memory"].shape[0]
        device = tracking_state["memory"].device
        
        # Extract new detections
        new_boxes = detections["boxes"]  # [B, N, 4]
        new_memory = detections["memory_vectors"]  # [B, N, memory_dim]
        new_scores = F.softmax(detections["class_logits"], dim=-1).max(dim=-1)[0]  # [B, N]
        
        updated_state = {}
        
        for b in range(batch_size):
            # Get current tracks for this batch item
            curr_memory = tracking_state["memory"][b]  # [T, memory_dim]
            curr_boxes = tracking_state["boxes"][b]  # [T, 4]
            curr_obj_ids = tracking_state["obj_ids"][b]  # [T]
            curr_active = tracking_state["active"][b]  # [T]
            curr_age = tracking_state["age"][b]  # [T]
            curr_hits = tracking_state["hits"][b]  # [T]
            next_obj_id = tracking_state["next_obj_id"][b]  # scalar
            
            # Get new detections for this batch item
            batch_new_boxes = new_boxes[b]  # [N, 4]
            batch_new_memory = new_memory[b]  # [N, memory_dim]
            batch_new_scores = new_scores[b]  # [N]
            
            # Match new detections to existing tracks
            matched_indices, unmatched_tracks, unmatched_detections = self._match_detections(
                curr_boxes, batch_new_boxes, curr_active, curr_memory, batch_new_memory
            )
            
            # Update matched tracks
            for track_idx, det_idx in matched_indices:
                curr_memory[track_idx] = batch_new_memory[det_idx]
                curr_boxes[track_idx] = batch_new_boxes[det_idx]
                curr_active[track_idx] = True
                curr_age[track_idx] = 0
                curr_hits[track_idx] += 1
            
            # Update unmatched tracks
            for track_idx in unmatched_tracks:
                if curr_active[track_idx]:
                    curr_age[track_idx] += 1
                    if curr_age[track_idx] > self.max_age:
                        curr_active[track_idx] = False
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                if len(curr_obj_ids) < self.max_objects:
                    # Add new track
                    curr_memory = torch.cat([curr_memory, batch_new_memory[det_idx:det_idx+1]], dim=0)
                    curr_boxes = torch.cat([curr_boxes, batch_new_boxes[det_idx:det_idx+1]], dim=0)
                    curr_obj_ids = torch.cat([curr_obj_ids, next_obj_id.unsqueeze(0)], dim=0)
                    curr_active = torch.cat([curr_active, torch.tensor([True], device=device)], dim=0)
                    curr_age = torch.cat([curr_age, torch.zeros(1, dtype=torch.long, device=device)], dim=0)
                    curr_hits = torch.cat([curr_hits, torch.ones(1, dtype=torch.long, device=device)], dim=0)
                    next_obj_id += 1
            
            # Update tracking state for this batch item
            if b == 0:
                # Initialize tensors for the updated state
                updated_state["memory"] = curr_memory.unsqueeze(0)
                updated_state["boxes"] = curr_boxes.unsqueeze(0)
                updated_state["obj_ids"] = curr_obj_ids.unsqueeze(0)
                updated_state["active"] = curr_active.unsqueeze(0)
                updated_state["age"] = curr_age.unsqueeze(0)
                updated_state["hits"] = curr_hits.unsqueeze(0)
                updated_state["next_obj_id"] = next_obj_id.unsqueeze(0)
            else:
                # Append tensors for this batch item
                updated_state["memory"] = torch.cat(
                    [updated_state["memory"], curr_memory.unsqueeze(0)], dim=0
                )
                updated_state["boxes"] = torch.cat(
                    [updated_state["boxes"], curr_boxes.unsqueeze(0)], dim=0
                )
                updated_state["obj_ids"] = torch.cat(
                    [updated_state["obj_ids"], curr_obj_ids.unsqueeze(0)], dim=0
                )
                updated_state["active"] = torch.cat(
                    [updated_state["active"], curr_active.unsqueeze(0)], dim=0
                )
                updated_state["age"] = torch.cat(
                    [updated_state["age"], curr_age.unsqueeze(0)], dim=0
                )
                updated_state["hits"] = torch.cat(
                    [updated_state["hits"], curr_hits.unsqueeze(0)], dim=0
                )
                updated_state["next_obj_id"] = torch.cat(
                    [updated_state["next_obj_id"], next_obj_id.unsqueeze(0)], dim=0
                )
        
        return updated_state
    
    def _match_detections(
        self,
        tracks_boxes: torch.Tensor,
        detections_boxes: torch.Tensor,
        tracks_active: torch.Tensor,
        tracks_memory: torch.Tensor,
        detections_memory: torch.Tensor,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using both feature similarity and IoU.
        
        Args:
            tracks_boxes: Current track boxes [T, 4]
            detections_boxes: New detection boxes [N, 4]
            tracks_active: Active status of tracks [T]
            tracks_memory: Memory vectors for current tracks [T, memory_dim]
            detections_memory: Memory vectors for new detections [N, memory_dim]
            
        Returns:
            Tuple of (matched_indices, unmatched_tracks, unmatched_detections)
        """
        if len(tracks_boxes) == 0 or len(detections_boxes) == 0:
            return [], list(range(len(tracks_boxes))) if len(tracks_boxes) > 0 else [], \
                   list(range(len(detections_boxes))) if len(detections_boxes) > 0 else []
        
        num_tracks = len(tracks_boxes)
        num_detections = len(detections_boxes)
        
        # Calculate feature similarity matrix
        similarity_matrix = torch.mm(
            F.normalize(tracks_memory, dim=1),
            F.normalize(detections_memory, dim=1).transpose(0, 1)
        )
        
        # Calculate IoU matrix
        iou_matrix = torch.zeros((num_tracks, num_detections), device=tracks_boxes.device)
        for t in range(num_tracks):
            for d in range(num_detections):
                iou_matrix[t, d] = self._box_iou(tracks_boxes[t], detections_boxes[d])
        
        # Combine similarity and IoU
        combined_matrix = similarity_matrix * 0.7 + iou_matrix * 0.3
        
        # Only consider active tracks
        if not tracks_active.all():
            combined_matrix[~tracks_active] = -1
        
        # Match detections to tracks using greedy algorithm
        matched_indices = []
        unmatched_tracks = []
        unmatched_detections = list(range(num_detections))
        
        while True:
            # Find highest value in the matrix
            flat_idx = torch.argmax(combined_matrix.view(-1))
            if combined_matrix.view(-1)[flat_idx] < self.similarity_threshold:
                break
                
            # Convert flat index to matrix indices
            track_idx = flat_idx // num_detections
            det_idx = flat_idx % num_detections
            
            # Add to matches
            matched_indices.append((track_idx.item(), det_idx.item()))
            
            # Remove matched track and detection from consideration
            combined_matrix[track_idx, :] = -1
            combined_matrix[:, det_idx] = -1
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)
        
        # Get unmatched tracks
        matched_track_indices = [t for t, _ in matched_indices]
        unmatched_tracks = [t for t in range(num_tracks) if t not in matched_track_indices and tracks_active[t]]
        
        return matched_indices, unmatched_tracks, unmatched_detections
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between two boxes in x,y,w,h format.
        
        Args:
            box1: First box [4] in x,y,w,h format
            box2: Second box [4] in x,y,w,h format
            
        Returns:
            IoU value
        """
        # Convert to x1,y1,x2,y2 format
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
        
        # Get intersection rectangle coordinates
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        # Calculate intersection area
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Calculate union area
        b1_area = box1[2] * box1[3]
        b2_area = box2[2] * box2[3]
        union_area = b1_area + b2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou