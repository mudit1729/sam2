# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator, Union
import os
import cv2
import json
from pathlib import Path

from .tracker import ObjectTracker

class MODTVideoPredictor:
    """
    Video predictor for the Memory-based Object Detection and Tracking (MODT) model.
    
    This class handles video frame processing, object tracking state management,
    and result visualization for the enhanced SAM2 model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tracker: ObjectTracker = None,
        device: torch.device = None,
    ):
        """
        Initialize the video predictor.
        
        Args:
            model: Enhanced SAM2 model
            tracker: Object tracker (created if not provided)
            device: Device to run inference on
        """
        self.model = model
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(device)
        
        # Create tracker if not provided
        if tracker is None:
            tracker = ObjectTracker(
                memory_dim=model.memory_dim,
                max_objects=20,
                similarity_threshold=0.7,
            ).to(device)
        self.tracker = tracker
    
    @classmethod
    def from_pretrained(cls, model_id: str, device: torch.device = None) -> "MODTVideoPredictor":
        """
        Create a video predictor from a pretrained model.
        
        Args:
            model_id: Pretrained model identifier
            device: Device to run inference on
            
        Returns:
            Initialized video predictor
        """
        # This is a placeholder - in a real implementation, this would load the model
        # from a pretrained checkpoint
        raise NotImplementedError("Loading from pretrained model is not yet implemented")
    
    def init_state(self, video_path: str = None, max_objects: int = 20) -> Dict:
        """
        Initialize tracking state for a video.
        
        Args:
            video_path: Path to video file (optional)
            max_objects: Maximum number of objects to track
            
        Returns:
            Initialized tracking state
        """
        tracking_state = self.tracker.initialize_tracks(batch_size=1)
        
        # Add video information if provided
        if video_path is not None:
            # Open video and get properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            tracking_state["video_info"] = {
                "path": video_path,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
            }
        
        # Add additional state information
        tracking_state["current_frame"] = torch.tensor([0], device=self.device)
        tracking_state["processing_complete"] = torch.tensor([False], device=self.device)
        tracking_state["max_objects"] = max_objects
        
        return tracking_state
    
    def add_new_points_or_box(
        self,
        tracking_state: Dict,
        frame_idx: int,
        obj_id: int = None,
        points: torch.Tensor = None,
        labels: torch.Tensor = None,
        box: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> Tuple[int, List[int], torch.Tensor, Dict]:
        """
        Add new points or box for tracking.
        
        Args:
            tracking_state: Current tracking state
            frame_idx: Frame index to process
            obj_id: Object ID (generated if not provided)
            points: Point prompts [N, 2]
            labels: Point labels [N] (1 for foreground, 0 for background)
            box: Box prompt [4] (x1, y1, x2, y2)
            mask: Mask prompt [H, W]
            
        Returns:
            Tuple of (frame_idx, object_ids, masks, detections)
        """
        # Load frame from video
        video_path = tracking_state["video_info"]["path"]
        frame = self._load_video_frame(video_path, frame_idx)
        frame_tensor = self._preprocess_frame(frame).to(self.device)
        
        # Format prompts
        if points is not None:
            points = points.unsqueeze(0).to(self.device)  # [1, N, 2]
            if labels is not None:
                labels = labels.unsqueeze(0).to(self.device)  # [1, N]
        
        if box is not None:
            box = box.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 4]
        
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]
        
        # Run model inference
        with torch.no_grad():
            image_size = (tracking_state["video_info"]["height"], tracking_state["video_info"]["width"])
            results = self.model(
                image=frame_tensor,
                memory=tracking_state["memory"],
                tracking_state=tracking_state,
                image_size=image_size,
                points=points,
                boxes=box,
                masks=mask,
            )
        
        # Update tracking state with new detections
        if "masks" in results:
            masks = results["masks"]
            scores = results.get("iou_scores", torch.ones(1, 1, device=self.device))
            masks_np = masks.cpu().numpy()[0]  # [N, H, W]
            
            # Initialize object ID if not provided
            if obj_id is None:
                obj_id = tracking_state["next_obj_id"][0].item()
                tracking_state["next_obj_id"][0] += 1
            
            # Create memory vectors for the mask
            if "memory_vectors" not in results:
                # If no memory vectors were generated by the model, create dummy ones
                # In a real implementation, you would extract features from the mask region
                memory_dim = tracking_state["memory"].shape[-1] if tracking_state["memory"].shape[1] > 0 else self.model.memory_dim
                memory_vector = torch.zeros(1, 1, memory_dim, device=self.device)
            else:
                memory_vector = results["memory_vectors"][:1]  # Take only the first one if multiple
            
            # Compute bounding box from mask
            if "boxes" not in results:
                boxes = []
                for mask in masks_np:
                    box = self._mask_to_box(mask)
                    boxes.append(box)
                box_tensor = torch.tensor(boxes, device=self.device).unsqueeze(0)  # [1, N, 4]
            else:
                box_tensor = results["boxes"][:1]  # Take only the first one if multiple
            
            # Update tracking state
            batch_idx = 0  # Always use first batch item
            
            # Add new object to tracking state
            tracking_state["memory"] = torch.cat([tracking_state["memory"], memory_vector], dim=1)
            tracking_state["boxes"] = torch.cat([tracking_state["boxes"], box_tensor], dim=1)
            tracking_state["obj_ids"] = torch.cat(
                [tracking_state["obj_ids"], torch.tensor([obj_id], device=self.device).unsqueeze(0)], dim=1
            )
            tracking_state["active"] = torch.cat(
                [tracking_state["active"], torch.tensor([True], device=self.device).unsqueeze(0)], dim=1
            )
            tracking_state["age"] = torch.cat(
                [tracking_state["age"], torch.zeros(1, 1, dtype=torch.long, device=self.device)], dim=1
            )
            tracking_state["hits"] = torch.cat(
                [tracking_state["hits"], torch.ones(1, 1, dtype=torch.long, device=self.device)], dim=1
            )
            
            # Return results
            return frame_idx, [obj_id], masks[0], results
        else:
            # No masks were generated
            return frame_idx, [], torch.zeros(0, frame.shape[0], frame.shape[1], device=self.device), results
    
    def propagate_in_video(
        self,
        tracking_state: Dict,
        start_frame: int = 0,
        end_frame: int = None,
        step: int = 1,
    ) -> Iterator[Tuple[int, List[int], torch.Tensor, Dict]]:
        """
        Propagate tracking through video frames.
        
        Args:
            tracking_state: Current tracking state
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)
            step: Frame step size
            
        Yields:
            Tuples of (frame_idx, object_ids, masks, detections)
        """
        video_path = tracking_state["video_info"]["path"]
        frame_count = tracking_state["video_info"]["frame_count"]
        
        if end_frame is None:
            end_frame = frame_count - 1
        
        # Ensure end_frame is valid
        end_frame = min(end_frame, frame_count - 1)
        
        for frame_idx in range(start_frame, end_frame + 1, step):
            # Update current frame in tracking state
            tracking_state["current_frame"][0] = frame_idx
            
            # Load and preprocess frame
            frame = self._load_video_frame(video_path, frame_idx)
            frame_tensor = self._preprocess_frame(frame).to(self.device)
            
            # Run model inference
            with torch.no_grad():
                image_size = (tracking_state["video_info"]["height"], tracking_state["video_info"]["width"])
                results = self.model(
                    image=frame_tensor,
                    memory=tracking_state["memory"],
                    tracking_state=tracking_state,
                    image_size=image_size,
                )
            
            # Update tracking state
            updated_state = self.tracker.update(tracking_state, results)
            tracking_state.update(updated_state)
            
            # Extract active objects
            active_mask = tracking_state["active"][0]
            obj_ids = tracking_state["obj_ids"][0][active_mask].cpu().tolist()
            
            # Generate masks for active objects
            if "masks" in results:
                masks = results["masks"][0]
            else:
                # If no masks in results, create empty masks
                masks = torch.zeros(
                    len(obj_ids), frame.shape[0], frame.shape[1], device=self.device
                )
            
            # Yield results
            yield frame_idx, obj_ids, masks, results
            
            # Check if processing is complete
            if tracking_state["processing_complete"][0]:
                break
    
    def _load_video_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """
        Load a specific frame from a video file.
        
        Args:
            video_path: Path to video file
            frame_idx: Frame index to load
            
        Returns:
            Loaded frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from video: {video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame tensor
        """
        # Convert to float and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        return frame_tensor
    
    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> List[float]:
        """
        Convert a binary mask to a bounding box.
        
        Args:
            mask: Binary mask as numpy array
            
        Returns:
            Box coordinates as [x, y, w, h]
        """
        # Find non-zero elements
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]  # Empty mask
        
        # Find bounding box coordinates
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Convert to [x, y, w, h] format
        x = float(x_min)
        y = float(y_min)
        w = float(x_max - x_min + 1)
        h = float(y_max - y_min + 1)
        
        return [x, y, w, h]