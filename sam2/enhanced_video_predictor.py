# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sam2.modeling.sam2_base_enhanced import SAM2EnhancedBase
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames


class SAM2EnhancedVideoPredictor(SAM2EnhancedBase):
    """
    Enhanced video predictor for SAM2 with DETR decoder and memory loop.
    """
    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        
        # Inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        
        # Cached features
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        
        # Object tracking
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        
        # Output dictionaries
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        
        # DETR tracking state
        inference_state["object_queries_per_obj"] = {}
        inference_state["memory_vectors_per_obj"] = {}
        
        # Warm up the visual backbone
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        
        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2EnhancedVideoPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2EnhancedVideoPredictor): The loaded model.
        """
        # We need to implement this with a custom build function for the enhanced model
        raise NotImplementedError(
            "from_pretrained is not yet implemented for SAM2EnhancedVideoPredictor."
        )

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # We always allow adding new objects
        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
        
        # Set up input and output structures for this object
        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        inference_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        inference_state["frames_tracked_per_obj"][obj_idx] = {}
        
        # Initialize DETR tracking state
        inference_state["object_queries_per_obj"][obj_idx] = None
        inference_state["memory_vectors_per_obj"][obj_idx] = None
        
        return obj_idx

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points or box to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # Handle box prompts
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        
        # Check if this is an initial conditioning frame
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        
        # Determine tracking direction
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
            
        # Set up output dictionary
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        
        # Determine if frame is a conditioning frame
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get previous mask logits if available
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out.get("pred_masks") is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp to avoid numerical issues
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
            
        # Get previous object queries and memory vector
        prev_queries = inference_state["object_queries_per_obj"][obj_idx]
        prev_memory_vector = inference_state["memory_vectors_per_obj"][obj_idx]
        
        # Run inference on this frame
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            prev_queries=prev_queries,
            prev_memory_vector=prev_memory_vector,
        )
        
        # Store updated object queries and memory vector
        if "updated_queries" in current_out:
            inference_state["object_queries_per_obj"][obj_idx] = current_out["updated_queries"]
        if "memory_vector" in current_out:
            inference_state["memory_vectors_per_obj"][obj_idx] = current_out["memory_vector"]
            
        # Add output to temporary dictionary
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize output masks to original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        
        # Also prepare detection results for output
        detections = None
        if "detections" in consolidated_out:
            detections = self._prepare_detections_for_output(
                inference_state, consolidated_out["detections"]
            )
            
        return frame_idx, obj_ids, video_res_masks, detections

    def _prepare_detections_for_output(self, inference_state, detections):
        """
        Convert normalized detection boxes to pixel coordinates in original video resolution.
        """
        pred_boxes = detections.get("pred_boxes", None)
        if pred_boxes is None:
            return None
            
        # Convert normalized coordinates to pixel coordinates
        video_height = inference_state["video_height"]
        video_width = inference_state["video_width"]
        
        # Unpack boxes [batch, num_queries, 4] where 4 is [cx, cy, w, h]
        cx, cy, w, h = pred_boxes.unbind(-1)
        
        # Convert to [x1, y1, x2, y2] format
        boxes = torch.stack([
            (cx - 0.5 * w) * video_width,
            (cy - 0.5 * h) * video_height,
            (cx + 0.5 * w) * video_width,
            (cy + 0.5 * h) * video_height,
        ], dim=-1)
        
        # Convert class logits to probabilities
        pred_logits = detections.get("pred_logits", None)
        if pred_logits is not None:
            scores = F.softmax(pred_logits, dim=-1)[..., :-1]  # Remove background class
            # Get highest score and corresponding class
            max_scores, pred_classes = scores.max(-1)
        else:
            max_scores = torch.ones_like(boxes[..., 0])
            pred_classes = torch.zeros_like(boxes[..., 0], dtype=torch.long)
            
        # Filter based on confidence threshold (0.5 by default)
        confidence_threshold = 0.5
        keep = max_scores > confidence_threshold
        
        # Return dictionary of results
        return {
            "boxes": boxes, 
            "scores": max_scores,
            "labels": pred_classes,
            "keep": keep,
        }

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        normalize_mask=True,
    ):
        """Add a new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        # Convert mask to tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
            
        # Add batch and channel dimensions if needed
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
            
        # Ensure proper dtype
        mask = mask.float()
        
        if normalize_mask:
            # Resize mask to internal image size
            mask = F.interpolate(
                mask,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            
        mask = mask.to(inference_state["device"])
        mask_inputs_per_frame[frame_idx] = mask
        point_inputs_per_frame.pop(frame_idx, None)
        
        # Check if this is an initial conditioning frame
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        
        # Determine tracking direction
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
            
        # Set up output dictionary
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        
        # Determine if frame is a conditioning frame
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        
        # Get previous object queries and memory vector
        prev_queries = inference_state["object_queries_per_obj"][obj_idx]
        prev_memory_vector = inference_state["memory_vectors_per_obj"][obj_idx]
        
        # Run inference on this frame
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask,
            reverse=reverse,
            run_mem_encoder=False,
            prev_queries=prev_queries,
            prev_memory_vector=prev_memory_vector,
        )
        
        # Store updated object queries and memory vector
        if "updated_queries" in current_out:
            inference_state["object_queries_per_obj"][obj_idx] = current_out["updated_queries"]
        if "memory_vector" in current_out:
            inference_state["memory_vectors_per_obj"][obj_idx] = current_out["memory_vector"]
            
        # Add output to temporary dictionary
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize output masks to original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        
        # Also prepare detection results for output
        detections = None
        if "detections" in consolidated_out:
            detections = self._prepare_detections_for_output(
                inference_state, consolidated_out["detections"]
            )
            
        return frame_idx, obj_ids, video_res_masks, detections

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        # Perform pre-flight checks and consolidate temporary outputs
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # Determine start index, end index, and processing order
        if start_frame_idx is None:
            # Default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # Default: track all frames
            max_frame_num_to_track = num_frames
            
        # Calculate processing range
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # Skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        # Process each frame in order
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            detections_per_obj = [None] * batch_size
            
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                
                # Skip frames that already have output (from user input)
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    detections = current_out.get("detections")
                    
                    # Clear non-conditioning memory if needed
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    # Get previous object queries and memory vector for this object
                    prev_queries = inference_state["object_queries_per_obj"][obj_idx]
                    prev_memory_vector = inference_state["memory_vectors_per_obj"][obj_idx]
                    
                    # Track new frame
                    storage_key = "non_cond_frame_outputs"
                    current_out, results = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                        prev_queries=prev_queries,
                        prev_memory_vector=prev_memory_vector,
                    )
                    
                    # Store output
                    obj_output_dict[storage_key][frame_idx] = current_out
                    
                    # Extract results
                    pred_masks = results[3]  # low_res_masks
                    detections = current_out.get("detections")
                    
                    # Update object queries and memory vector
                    if "updated_queries" in current_out:
                        inference_state["object_queries_per_obj"][obj_idx] = current_out["updated_queries"]
                    if "memory_vector" in current_out:
                        inference_state["memory_vectors_per_obj"][obj_idx] = current_out["memory_vector"]

                # Record that this frame has been tracked
                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                
                # Store results
                pred_masks_per_obj[obj_idx] = pred_masks
                detections_per_obj[obj_idx] = detections

            # Prepare output masks at video resolution
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
                
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            
            # Prepare detection results
            all_detections = self._consolidate_detections(
                inference_state, detections_per_obj
            )
            
            yield frame_idx, obj_ids, video_res_masks, all_detections

    def _consolidate_detections(self, inference_state, detections_per_obj):
        """
        Combine detections from multiple objects.
        """
        # If no detections, return None
        if not detections_per_obj or all(d is None for d in detections_per_obj):
            return None
            
        # Use the first valid detection as a template
        first_valid = next((d for d in detections_per_obj if d is not None), None)
        if first_valid is None:
            return None
            
        # Convert to pixel coordinates in original video resolution
        all_boxes = []
        all_scores = []
        all_labels = []
        all_obj_ids = []
        
        for obj_idx, det in enumerate(detections_per_obj):
            if det is None:
                continue
                
            # Get object ID
            obj_id = self._obj_idx_to_id(inference_state, obj_idx)
            
            # Filter by confidence
            keep_idx = det.get("keep", torch.ones_like(det["scores"], dtype=torch.bool))
            if not keep_idx.any():
                continue
                
            # Add detections for this object
            all_boxes.append(det["boxes"][keep_idx])
            all_scores.append(det["scores"][keep_idx])
            all_labels.append(det["labels"][keep_idx])
            
            # Add object ID for each detection
            num_dets = keep_idx.sum().item()
            all_obj_ids.append(torch.full((num_dets,), obj_id, dtype=torch.long))
            
        if not all_boxes:
            return None
            
        # Combine detections
        return {
            "boxes": torch.cat(all_boxes, dim=0),
            "scores": torch.cat(all_scores, dim=0),
            "labels": torch.cat(all_labels, dim=0),
            "obj_ids": torch.cat(all_obj_ids, dim=0),
        }
        
    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Check that we have at least one object
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError(
                "No input points or masks are provided for any object; please add inputs first."
            )

        # Consolidate temporary outputs for each object
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            
            # Process both conditioning and non-conditioning frame outputs
            for is_cond in [False, True]:
                storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                
                # Process all frames with temporary outputs
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder if not done already
                    if "memory_vector" not in out or out["memory_vector"] is None:
                        # Get features and detection results
                        features = self._get_image_feature(inference_state, frame_idx, 1)
                        
                        # Create a mask for memory encoding
                        if "pred_masks" in out and out["pred_masks"] is not None:
                            high_res_masks = F.interpolate(
                                out["pred_masks"].to(inference_state["device"]),
                                size=(self.image_size, self.image_size),
                                mode="bilinear",
                                align_corners=False,
                            )
                        else:
                            # Create a blank mask if no predictions available
                            high_res_masks = torch.zeros(
                                (1, 1, self.image_size, self.image_size),
                                device=inference_state["device"]
                            )
                            
                        # Get detections
                        detections = out.get("detections", None)
                        
                        # Get previous memory vector
                        prev_memory_vector = inference_state["memory_vectors_per_obj"].get(obj_idx)
                        
                        # Encode memory
                        _, _, feat_sizes, _ = features
                        memory_dict = self._encode_new_memory(
                            current_vision_feats=[x[-1:] for x in features[2]],
                            feat_sizes=feat_sizes,
                            detections=detections,
                            prev_memory_vector=prev_memory_vector,
                        )
                        
                        out["memory_vector"] = memory_dict["memory_vector"]
                        out["memory_pos_enc"] = memory_dict["memory_pos_enc"]
                        
                        # Update memory vector in state
                        inference_state["memory_vectors_per_obj"][obj_idx] = memory_dict["memory_vector"]
                    
                    # Store in output dictionary
                    obj_output_dict[storage_key][frame_idx] = out
                    
                    # Clear non-conditioning memory if needed
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                
                # Clear temporary outputs
                obj_temp_output_dict[storage_key].clear()
            
            # Ensure each object has received input
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(
                    f"No input points or masks are provided for object id {obj_id}; please add inputs first."
                )
                
            # Remove duplicate outputs
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        prev_queries=None,
        prev_memory_vector=None,
    ):
        """Run inference on a single frame."""
        images, backbone_out, features, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        
        # Forward pass
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=features[2],
            current_vision_pos_embeds=features[3],
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            prev_sam_mask_logits=prev_sam_mask_logits,
            prev_queries=prev_queries,
            prev_memory_vector=prev_memory_vector,
        )
        
        # Offload to storage device if needed
        storage_device = inference_state["storage_device"]
        if storage_device != inference_state["device"]:
            current_out = {k: (v.to(storage_device) if isinstance(v, torch.Tensor) else v)
                          for k, v in current_out.items()}
                          
        return current_out, sam_outputs

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Get image features for a frame."""
        # Look up in cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        
        if backbone_out is None:
            # Cache miss - run inference on single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            
            # Cache for future use
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
            
        # Expand features for batch size
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos
            
        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        
        return features

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        consolidate_at_video_res=False,
    ):
        """Consolidate temporary outputs across objects for a frame."""
        # Initialize consolidated output with default values
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            return {}
            
        device = inference_state["device"]
        
        # Prepare masks field
        consolidated_mask_key = (
            "pred_masks_video_res" if consolidate_at_video_res else "pred_masks"
        )
        if consolidate_at_video_res:
            H, W = inference_state["video_height"], inference_state["video_width"]
        else:
            # Assume internal SAM2 mask resolution (model dependent)
            H = W = self.image_size // 4
            
        # Initialize with all zeros (no object)
        consolidated_pred_masks = torch.zeros(
            (batch_size, 1, H, W), dtype=torch.float32, device=device
        )
        
        # Prepare detections field
        has_detections = False
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            if out is not None and "detections" in out:
                has_detections = True
                break
                
        if has_detections:
            # Collect detections from all objects
            consolidated_detections = {
                "pred_logits": [],
                "pred_boxes": [],
                "obj_scores": [],
            }
        else:
            consolidated_detections = None
            
        # Fill in the consolidated output with each object's output
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            
            # Look up temporary output first
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            
            # If not found in temporary output, look in consolidated output
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
                
            # Skip if no output for this object
            if out is None:
                continue
                
            # Add mask to consolidated output
            if "pred_masks" in out and out["pred_masks"] is not None:
                obj_mask = out["pred_masks"]
                
                # Resize mask if needed
                if obj_mask.shape[-2:] != consolidated_pred_masks.shape[-2:]:
                    resized_obj_mask = F.interpolate(
                        obj_mask,
                        size=consolidated_pred_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    resized_obj_mask = obj_mask
                    
                consolidated_pred_masks[obj_idx:obj_idx+1] = resized_obj_mask
                
            # Add detections to consolidated output
            if consolidated_detections is not None and "detections" in out:
                det = out["detections"]
                if det is not None:
                    for key in consolidated_detections:
                        if key in det:
                            consolidated_detections[key].append(det[key])
        
        # Finalize consolidated output
        consolidated_out = {consolidated_mask_key: consolidated_pred_masks}
        
        # Add detections if available
        if consolidated_detections is not None and any(v for v in consolidated_detections.values()):
            # Convert lists to tensors for each key
            for key in consolidated_detections:
                if consolidated_detections[key]:
                    consolidated_detections[key] = torch.cat(consolidated_detections[key], dim=0)
                else:
                    # Create empty tensor with appropriate shape
                    if key == "pred_logits":
                        consolidated_detections[key] = torch.zeros((0, 2), device=device)  # [0, num_classes+1]
                    elif key == "pred_boxes":
                        consolidated_detections[key] = torch.zeros((0, 4), device=device)  # [0, 4]
                    elif key == "obj_scores":
                        consolidated_detections[key] = torch.zeros((0, 1), device=device)  # [0, 1]
                        
            consolidated_out["detections"] = consolidated_detections
            
        return consolidated_out

    def _get_orig_video_res_output(self, inference_state, mask_logits):
        """Convert mask logits to binary masks at original video resolution."""
        sigmoid_mask_scores = torch.sigmoid(mask_logits)
        
        if self.fill_hole_area > 0:
            sigmoid_mask_scores = fill_holes_in_mask_scores(
                sigmoid_mask_scores, area_threshold=self.fill_hole_area
            )
            
        if self.non_overlap_masks:
            # Apply non-overlapping constraints by keeping highest scoring mask at each location
            max_obj_inds = torch.argmax(sigmoid_mask_scores, dim=0, keepdim=True)
            batch_obj_inds = torch.arange(
                sigmoid_mask_scores.size(0), device=sigmoid_mask_scores.device
            )[:, None, None, None]
            keep = max_obj_inds == batch_obj_inds
            sigmoid_mask_scores = sigmoid_mask_scores * keep.float()
            
        # Convert to binary masks
        binary_masks = (sigmoid_mask_scores > 0.5).cpu()
        
        return sigmoid_mask_scores, binary_masks

    def _clear_obj_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
        """Clear non-conditioning memory around an input frame."""
        memory_radius = 5  # Number of frames to clear on each side
        num_frames = inference_state["num_frames"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        
        # Clear surrounding frames in non_cond_frame_outputs
        for nearby_frame_idx in range(
            max(0, frame_idx - memory_radius),
            min(num_frames, frame_idx + memory_radius + 1)
        ):
            if nearby_frame_idx != frame_idx:
                obj_output_dict["non_cond_frame_outputs"].pop(nearby_frame_idx, None)