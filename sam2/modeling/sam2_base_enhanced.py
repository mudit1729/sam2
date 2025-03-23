# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.detr_decoder import build_detr_decoder
from sam2.modeling.detection_head import ObjectDetectionHead, ObjectRefinementHead
from sam2.modeling.enhanced_memory_attention import build_enhanced_memory_attention
from sam2.modeling.enhanced_memory_encoder import EnhancedMemoryEncoder, DetectionProcessor


class SAM2EnhancedBase(SAM2Base):
    """
    Enhanced SAM2 base model with DETR decoder and memory feedback loop.
    This implementation adds:
    1. Memory attention for conditioning current features with past memory
    2. DETR decoder for object detection
    3. Object detection head
    4. Memory encoder that combines detections with visual features
    5. Memory bank with feedback to memory attention
    """
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        # Enhanced components
        detr_decoder_layers=6,
        detr_decoder_heads=8,
        detr_decoder_dim_feedforward=2048,
        detr_num_queries=100,
        detection_num_classes=1,  # Default for tracking is just 1 class
        detection_num_layers=3,
        refinement_enabled=True,
        small_object_threshold=0.05,
        refinement_factor=2.0,
        enhanced_memory_attention_layers=2,
        # Original parameters
        num_maskmem=7,
        image_size=512,
        backbone_stride=16,
        sigmoid_scale_for_mem_enc=1.0,
        sigmoid_bias_for_mem_enc=0.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr=False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores=False,
        pred_obj_scores_mlp=False,
        fixed_no_obj_ptr=False,
        soft_no_obj_ptr=False,
        use_mlp_for_obj_ptr_proj=False,
        no_obj_embed_spatial=False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder=False,
    ):
        # Initialize the base SAM2 model
        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,  # Original memory attention (will be replaced)
            memory_encoder=memory_encoder,      # Original memory encoder (will be enhanced)
            num_maskmem=num_maskmem,
            image_size=image_size,
            backbone_stride=backbone_stride,
            sigmoid_scale_for_mem_enc=sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc=sigmoid_bias_for_mem_enc,
            binarize_mask_from_pts_for_mem_enc=binarize_mask_from_pts_for_mem_enc,
            use_mask_input_as_output_without_sam=use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn=max_cond_frames_in_attn,
            directly_add_no_mem_embed=directly_add_no_mem_embed,
            use_high_res_features_in_sam=use_high_res_features_in_sam,
            multimask_output_in_sam=multimask_output_in_sam,
            multimask_min_pt_num=multimask_min_pt_num,
            multimask_max_pt_num=multimask_max_pt_num,
            multimask_output_for_tracking=multimask_output_for_tracking,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            memory_temporal_stride_for_eval=memory_temporal_stride_for_eval,
            non_overlap_masks_for_mem_enc=non_overlap_masks_for_mem_enc,
            use_obj_ptrs_in_encoder=use_obj_ptrs_in_encoder,
            max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
            add_tpos_enc_to_obj_ptrs=add_tpos_enc_to_obj_ptrs,
            proj_tpos_enc_in_obj_ptrs=proj_tpos_enc_in_obj_ptrs,
            use_signed_tpos_enc_to_obj_ptrs=use_signed_tpos_enc_to_obj_ptrs,
            only_obj_ptrs_in_the_past_for_eval=only_obj_ptrs_in_the_past_for_eval,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            fixed_no_obj_ptr=fixed_no_obj_ptr,
            soft_no_obj_ptr=soft_no_obj_ptr,
            use_mlp_for_obj_ptr_proj=use_mlp_for_obj_ptr_proj,
            no_obj_embed_spatial=no_obj_embed_spatial,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
            compile_image_encoder=compile_image_encoder,
        )
        
        # Store parameters
        self.hidden_dim = self.image_encoder.neck.d_model
        self.refinement_enabled = refinement_enabled
        
        # Part 1: Replace the memory attention with enhanced version
        self.enhanced_memory_attention = build_enhanced_memory_attention(
            d_model=self.hidden_dim,
            nhead=detr_decoder_heads,
            num_layers=enhanced_memory_attention_layers,
            dim_feedforward=detr_decoder_dim_feedforward,
        )
        
        # Part 2: Add the DETR decoder
        self.detr_decoder = build_detr_decoder(
            d_model=self.hidden_dim,
            nhead=detr_decoder_heads,
            num_decoder_layers=detr_decoder_layers,
            dim_feedforward=detr_decoder_dim_feedforward,
            num_queries=detr_num_queries,
        )
        
        # Part 3: Add the object detection head
        self.detection_head = ObjectDetectionHead(
            hidden_dim=self.hidden_dim,
            num_classes=detection_num_classes,
            num_layers=detection_num_layers,
        )
        
        # Part 4: Add the object refinement head (if enabled)
        if refinement_enabled:
            self.refinement_head = ObjectRefinementHead(
                hidden_dim=self.hidden_dim,
                small_object_threshold=small_object_threshold,
                refinement_factor=refinement_factor,
            )
        
        # Part 5: Create the detection processor for memory encoding
        self.detection_processor = DetectionProcessor(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            max_detections=detr_num_queries,
        )
        
        # Part 6: Create the enhanced memory encoder
        # We'll use the original memory encoder for patch processing
        # and wrap it with our enhanced memory encoder
        self.enhanced_memory_encoder = EnhancedMemoryEncoder(
            out_dim=self.mem_dim,
            patch_encoder=self.memory_encoder,
            position_encoding=self.memory_encoder.position_encoding,
            detection_processor=self.detection_processor,
            in_dim=self.hidden_dim,
            fusion_type='concat',
        )
        
        # Part 7: Initialize memory bank (similar to base implementation)
        self.memory_bank = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.memory_bank, std=0.02)
        
        # Learned object queries for initialization
        self.object_queries = torch.nn.Parameter(
            torch.zeros(detr_num_queries, self.hidden_dim)
        )
        trunc_normal_(self.object_queries, std=0.02)
        
    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """
        Override the base method to use our enhanced memory attention.
        Fuses the current frame's visual feature map with previous memory.
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        
        # The case of no memory (first frame or disabled memory)
        if self.num_maskmem == 0 or is_init_cond_frame:
            # Just reshape the visual features without memory conditioning
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat
        
        # For non-initial frames, retrieve memory vectors
        memory_vectors = []
        memory_pos = []
        
        # First check if we have non-conditioning memory frames
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos  # how many frames before current frame
            if not track_in_reverse:
                # the frame immediately before this frame (i.e. frame_idx - 1)
                prev_frame_idx = frame_idx - t_rel
            else:
                # the frame immediately after this frame (i.e. frame_idx + 1)
                prev_frame_idx = frame_idx + t_rel
                
            # Skip if frame is out of range
            if prev_frame_idx < 0 or (num_frames is not None and prev_frame_idx >= num_frames):
                continue
                
            # Get the memory for this frame
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
            if out is None:
                # Check conditioning frames as well
                out = output_dict["cond_frame_outputs"].get(prev_frame_idx, None)
                
            if out is not None and "memory_vector" in out:
                # Retrieve memory vector for this frame
                mem_vec = out["memory_vector"].to(device, non_blocking=True)
                mem_pos = out["memory_pos_enc"].to(device, non_blocking=True)
                
                memory_vectors.append(mem_vec)
                memory_pos.append(mem_pos)
        
        # Convert current visual features to the right shape
        # [HW, B, C] -> [B, HW, C]
        current_features = current_vision_feats[-1].permute(1, 0, 2)
        current_pos = current_vision_pos_embeds[-1].permute(1, 0, 2)
        
        # Stack memory vectors if available
        if memory_vectors:
            stacked_memory = torch.stack(memory_vectors, dim=0)  # [num_mems, B, C]
            stacked_memory_pos = torch.stack(memory_pos, dim=0)  # [num_mems, B, C]
        else:
            stacked_memory = None
            stacked_memory_pos = None
        
        # Apply enhanced memory attention
        memory_conditioned_features = self.enhanced_memory_attention(
            curr_features=current_features,
            memory_vectors=stacked_memory,
            curr_pos=current_pos,
            memory_pos=stacked_memory_pos,
        )
        
        # Reshape back to spatial format
        # [B, HW, C] -> [B, C, H, W]
        pix_feat_with_mem = memory_conditioned_features.view(B, H * W, C)
        pix_feat_with_mem = pix_feat_with_mem.permute(0, 2, 1).view(B, C, H, W)
        
        return pix_feat_with_mem
    
    def _process_with_detr_decoder(self, memory_conditioned_features, prev_queries=None):
        """
        Process memory-conditioned features with DETR decoder.
        
        Args:
            memory_conditioned_features: Features after memory attention [B, C, H, W]
            prev_queries: Optional previous object queries for temporal consistency
            
        Returns:
            Dict containing detections and updated queries
        """
        B, C, H, W = memory_conditioned_features.shape
        
        # Reshape features for decoder: [B, C, H, W] -> [HW, B, C]
        memory = memory_conditioned_features.flatten(2).permute(2, 0, 1)
        
        # Prepare positional encoding for memory
        # We reuse the position encoding from the original implementation
        if hasattr(self, 'sam_prompt_encoder') and hasattr(self.sam_prompt_encoder, 'get_dense_pe'):
            pos_embed = self.sam_prompt_encoder.get_dense_pe()
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        else:
            # Fallback if no position encoding is available
            pos_embed = torch.zeros_like(memory)
        
        # Use previous queries if provided, otherwise use learned queries
        if prev_queries is not None:
            # Ensure queries are properly shaped [num_queries, B, C]
            if prev_queries.dim() == 2:
                # [B, C] -> [1, B, C]
                queries = prev_queries.unsqueeze(0)
            elif prev_queries.dim() == 3 and prev_queries.shape[1] == B:
                # [num_queries, B, C] is the correct shape
                queries = prev_queries
            else:
                # Unexpected shape, fall back to learned queries
                queries = self.object_queries.unsqueeze(1).repeat(1, B, 1)
        else:
            # Use learned object queries
            queries = self.object_queries.unsqueeze(1).repeat(1, B, 1)
        
        # Process features with DETR decoder
        decoder_output = self.detr_decoder(
            memory=memory,
            memory_pos=pos_embed,
            query_embed=queries,
        )
        
        # Apply detection head to get class predictions and bounding boxes
        detections = self.detection_head(decoder_output)
        
        # Extract updated queries to use in the next frame
        # Use the output of the last decoder layer as updated queries
        updated_queries = decoder_output
        
        return {"detections": detections, "updated_queries": updated_queries}
    
    def _apply_refinement(self, detections, high_res_features, prev_memory_vector=None):
        """
        Apply refinement to detections, particularly for small objects.
        
        Args:
            detections: Output from detection head
            high_res_features: High-resolution features for refinement
            prev_memory_vector: Optional previous memory vector
            
        Returns:
            Refined detections
        """
        if not self.refinement_enabled:
            return detections
            
        # If using high-res features
        if self.use_high_res_features_in_sam and high_res_features is not None:
            refined_detections = self.refinement_head(
                detections=detections,
                high_res_vectors=high_res_features,
                prev_memory_vector=prev_memory_vector,
            )
            return refined_detections
        else:
            # No refinement if high-res features are not available
            return detections
    
    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        detections,
        prev_memory_vector=None,
    ):
        """
        Encode the current image and detections into a memory feature.
        
        Args:
            current_vision_feats: Visual features from the current frame
            feat_sizes: Feature sizes
            detections: Detection results from object detection head
            prev_memory_vector: Optional previous memory vector
            
        Returns:
            Dictionary containing memory vector and positional encoding
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        
        # Reshape features to spatial format: [HW, B, C] -> [B, C, H, W]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        
        # Use the enhanced memory encoder
        memory_dict = self.enhanced_memory_encoder(
            patch_embeddings=pix_feat,
            detections=detections,
            prev_memory_vector=prev_memory_vector,
        )
        
        return memory_dict
    
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
        prev_queries=None,
        prev_memory_vector=None,
    ):
        """
        Enhanced track step for processing a single frame.
        
        Args:
            frame_idx: Current frame index
            is_init_cond_frame: Whether this is an initial conditioning frame
            current_vision_feats: Visual features for current frame
            current_vision_pos_embeds: Position embeddings for current frame
            feat_sizes: Feature sizes
            point_inputs: Point inputs (clicks)
            mask_inputs: Mask inputs
            output_dict: Output dictionary for storing results
            num_frames: Total number of frames
            track_in_reverse: Whether to track in reverse time order
            prev_sam_mask_logits: Previous mask logits
            prev_queries: Previous object queries for temporal consistency
            prev_memory_vector: Previous memory vector
            
        Returns:
            Tuple of (current output, sam outputs, high-res features, conditioned features)
        """
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
            
        # Process with SAM if mask inputs are provided and use_mask_input_as_output_without_sam is True
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # Use the original method for this case
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
            
            # No DETR processing in this case
            return current_out, sam_outputs, high_res_features, pix_feat
        
        # 1. Memory Attention: Fuse the visual feature with previous memory features
        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
        )
        
        # 2. If point or mask inputs are provided, use SAM for interactivity
        # This allows the model to still work with user inputs like the original SAM2
        if point_inputs is not None or (mask_inputs is not None and prev_sam_mask_logits is not None):
            # Apply SAM-style segmentation head
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs if mask_inputs is not None else prev_sam_mask_logits,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
            
            # Store the outputs
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = sam_outputs
            
            # Store results in the current output
            current_out["pred_masks"] = low_res_masks
            current_out["pred_masks_high_res"] = high_res_masks
            current_out["obj_ptr"] = obj_ptr
            current_out["object_score_logits"] = object_score_logits
            
            # 3. Also process with DETR decoder for object detection
            detr_results = self._process_with_detr_decoder(
                memory_conditioned_features=pix_feat_with_mem,
                prev_queries=prev_queries,
            )
            
            # 4. Apply refinement to detections
            refined_detections = self._apply_refinement(
                detections=detr_results["detections"],
                high_res_features=high_res_features[0] if high_res_features else None,
                prev_memory_vector=prev_memory_vector,
            )
            
            # Store detection results
            current_out["detections"] = refined_detections
            current_out["updated_queries"] = detr_results["updated_queries"]
            
            # 5. Encode memory vector
            memory_dict = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                detections=refined_detections,
                prev_memory_vector=prev_memory_vector,
            )
            
            # Store memory information
            current_out["memory_vector"] = memory_dict["memory_vector"]
            current_out["memory_pos_enc"] = memory_dict["memory_pos_enc"]
            
            return current_out, sam_outputs, high_res_features, pix_feat_with_mem
            
        else:
            # No user inputs, use DETR for automatic tracking
            
            # 3. Process with DETR decoder
            detr_results = self._process_with_detr_decoder(
                memory_conditioned_features=pix_feat_with_mem,
                prev_queries=prev_queries,
            )
            
            # 4. Apply refinement to detections
            refined_detections = self._apply_refinement(
                detections=detr_results["detections"],
                high_res_features=high_res_features[0] if high_res_features else None,
                prev_memory_vector=prev_memory_vector,
            )
            
            # Store detection results
            current_out["detections"] = refined_detections
            current_out["updated_queries"] = detr_results["updated_queries"]
            
            # 5. Encode memory vector
            memory_dict = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                detections=refined_detections,
                prev_memory_vector=prev_memory_vector,
            )
            
            # Store memory information
            current_out["memory_vector"] = memory_dict["memory_vector"]
            current_out["memory_pos_enc"] = memory_dict["memory_pos_enc"]
            
            # Create dummy SAM outputs for compatibility with the original code
            B = pix_feat_with_mem.size(0)
            C = self.hidden_dim
            device = pix_feat_with_mem.device
            
            # Create blank masks
            h, w = high_res_features[0].shape[-2:] if high_res_features else (self.image_size // 4, self.image_size // 4)
            low_res_masks = torch.zeros(B, 1, h, w, device=device)
            high_res_masks = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
            
            # Use detection scores as object scores
            obj_scores = refined_detections["obj_scores"].mean(dim=1)
            
            # Create dummy SAM outputs
            sam_outputs = (
                low_res_masks.repeat(1, 3, 1, 1),  # low_res_multimasks
                high_res_masks.repeat(1, 3, 1, 1),  # high_res_multimasks
                torch.ones(B, 3, device=device),  # ious
                low_res_masks,  # low_res_masks
                high_res_masks,  # high_res_masks
                torch.zeros(B, C, device=device),  # obj_ptr
                obj_scores,  # object_score_logits
            )
            
            return current_out, sam_outputs, high_res_features, pix_feat_with_mem