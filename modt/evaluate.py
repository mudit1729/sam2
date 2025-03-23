# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import torch
import argparse
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union

try:
    from nuimages import NuImages
except ImportError:
    print("Warning: nuimages package not found. Install with: pip install nuimages")

from .base_model import SAM2EnhancedModel
from .tracking_predictor import MODTVideoPredictor
from .data_loader import create_nuimages_dataloader
from .tracker import ObjectTracker

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for SAM2 Enhanced Object Tracking with nuImages")
    
    # Dataset parameters
    parser.add_argument("--nuimages-root", type=str, required=True,
                        help="Path to the nuImages dataset root directory")
    parser.add_argument("--nuimages-version", type=str, default="v1.0-mini",
                        help="Version of the nuImages dataset to use")
    parser.add_argument("--clip-length", type=int, default=13,
                        help="Number of frames in each clip")
    
    # Model parameters
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--sam2-checkpoint", type=str, default=None,
                        help="Path to SAM2 model checkpoint")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize and save tracking results")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    
    return parser.parse_args()

def load_model(model_path, sam2_checkpoint, device):
    """Load the SAM2EnhancedModel from checkpoint."""
    # This is similar to the build_model function in train.py
    # Load SAM2 model components
    from .train import load_sam2_model
    
    sam2_model = load_sam2_model(sam2_checkpoint)
    image_encoder = sam2_model["image_encoder"]
    mask_decoder = sam2_model["mask_decoder"]
    
    # Create the enhanced model
    model = SAM2EnhancedModel(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        embed_dim=256,
        memory_dim=256,
        num_heads=8,
        num_decoder_layers=6,
        num_queries=100,
        num_classes=80,
        use_refinement=True,
    )
    
    # Load trained parameters
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def create_video_from_clip(clip_images, output_path, fps=10):
    """Create a video from a list of images."""
    if not clip_images:
        logger.warning("No images provided to create video")
        return
    
    # Convert PIL images or tensors to numpy arrays
    frames = []
    for img in clip_images:
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy array
            if img.dim() == 4:  # batch dimension
                img = img.squeeze(0)
            if img.dim() == 3 and img.shape[0] in [1, 3]:  # CHW format
                img = img.permute(1, 2, 0)
            img_np = img.cpu().numpy()
            
            # Normalize if needed
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            # Assume PIL image
            img_np = np.array(img)
        
        # Ensure RGB format
        if img_np.ndim == 2:  # grayscale
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        
        frames.append(img_np)
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    logger.info(f"Video saved to {output_path}")

def visualize_tracking(clip_images, tracking_results, output_dir, sample_idx):
    """Visualize tracking results on images and save as a video."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare frames with tracking visualizations
    vis_frames = []
    
    for i, img in enumerate(clip_images):
        # Convert tensor to numpy array if needed
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:  # batch dimension
                img = img.squeeze(0)
            if img.dim() == 3 and img.shape[0] in [1, 3]:  # CHW format
                img = img.permute(1, 2, 0)
            img_np = img.cpu().numpy()
            
            # Normalize if needed
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            # Assume PIL image
            img_np = np.array(img)
        
        # Create a copy for visualization
        vis_frame = img_np.copy()
        
        # Draw tracking results if available for this frame
        if i < len(tracking_results):
            frame_result = tracking_results[i]
            obj_ids = frame_result.get("obj_ids", [])
            boxes = frame_result.get("boxes", [])
            
            # Draw boxes and object IDs
            for obj_id, box in zip(obj_ids, boxes):
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Draw rectangle
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw object ID
                cv2.putText(vis_frame, f"ID: {obj_id}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        vis_frames.append(vis_frame)
    
    # Create and save video
    video_path = os.path.join(output_dir, f"tracking_sample_{sample_idx}.mp4")
    create_video_from_clip(vis_frames, video_path)
    
    # Save individual frames
    frames_dir = os.path.join(output_dir, f"frames_sample_{sample_idx}")
    os.makedirs(frames_dir, exist_ok=True)
    
    for i, frame in enumerate(vis_frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    return video_path

def evaluate_on_nuimages(args):
    """Evaluate the model on nuImages dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.model_path, args.sam2_checkpoint, device)
    model = model.to(device)
    model.eval()
    
    # Create tracker
    tracker = ObjectTracker(
        memory_dim=model.memory_dim,
        max_objects=20,
        similarity_threshold=0.7,
    ).to(device)
    
    # Create video predictor
    predictor = MODTVideoPredictor(model, tracker, device)
    
    # Initialize nuImages
    try:
        logger.info(f"Initializing nuImages {args.nuimages_version} from {args.nuimages_root}")
        nuim = NuImages(
            dataroot=args.nuimages_root,
            version=args.nuimages_version,
            verbose=True,
            lazy=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize nuImages: {e}")
        # Create a dummy nuImages instance for testing
        try:
            # Try to import from train module
            from .train import DummyNuImages
            nuim = DummyNuImages()
        except ImportError:
            # Define it here if import fails
            logger.warning("Creating a dummy nuImages instance for testing")
            class DummyNuImages:
                def __init__(self):
                    self.sample = [{"token": f"sample_{i}", "timestamp": i*1000} for i in range(10)]
                    self.sample_data = []
                    for i, sample in enumerate(self.sample):
                        for j in range(13):  # 13 frames per sample
                            self.sample_data.append({
                                "token": f"sd_{i}_{j}",
                                "sample_token": sample["token"],
                                "timestamp": sample["timestamp"] + (j-6)*50,  # centered around keyframe
                                "filename": f"dummy/image_{i}_{j}.jpg"
                            })
                
                def get(self, table_name, token):
                    if table_name == "sample":
                        return next((s for s in self.sample if s["token"] == token), None)
                    elif table_name == "sample_data":
                        return next((sd for sd in self.sample_data if sd["token"] == token), None)
                    elif table_name == "object_ann" or table_name == "surface_ann":
                        return {"category_name": "vehicle.car", "bbox": [100, 100, 50, 30]}
                    return None
                
                def list_anns(self, sample_token):
                    # Return (object_tokens, surface_tokens)
                    return [f"obj_ann_{sample_token}_{i}" for i in range(3)], []
            
            nuim = DummyNuImages()
            
        # Override the dataroot to use a local test directory
        args.nuimages_root = "./test_data"
        os.makedirs(args.nuimages_root, exist_ok=True)
        os.makedirs(os.path.join(args.nuimages_root, "dummy"), exist_ok=True)
    
    # Define image transformations
    # Use dimensions divisible by 14 for DINOv2's patch size
    transform = transforms.Compose([
        transforms.Resize((448, 448)),  # 448 is divisible by 14 (14*32=448)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataloader
    logger.info("Creating dataloader")
    dataloader = create_nuimages_dataloader(
        nuim,
        dataroot=args.nuimages_root,
        batch_size=1,  # Process one sample at a time for evaluation
        clip_length=args.clip_length,
        transform=transform,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluation loop
    logger.info("Starting evaluation")
    
    results = []
    
    num_samples = args.num_samples or len(dataloader)
    for sample_idx, batch in enumerate(tqdm(dataloader, total=num_samples)):
        if args.num_samples and sample_idx >= args.num_samples:
            break
        
        sample_token = batch["meta"][0]["sample_token"]
        
        # Initialize tracking for this clip
        clip_images = batch["clip"][0]  # Get the first clip in the batch
        
        # Create a tracking state for this clip
        tracking_state = {
            "memory": torch.zeros(1, 0, model.memory_dim, device=device),
            "boxes": torch.zeros(1, 0, 4, device=device),
            "obj_ids": torch.zeros(1, 0, dtype=torch.long, device=device),
            "active": torch.zeros(1, 0, dtype=torch.bool, device=device),
            "age": torch.zeros(1, 0, dtype=torch.long, device=device),
            "hits": torch.zeros(1, 0, dtype=torch.long, device=device),
            "next_obj_id": torch.ones(1, dtype=torch.long, device=device),
            "video_info": {
                "path": f"sample_{sample_token}",
                "width": clip_images[0].shape[2] if isinstance(clip_images[0], torch.Tensor) else clip_images[0].width,
                "height": clip_images[0].shape[1] if isinstance(clip_images[0], torch.Tensor) else clip_images[0].height,
                "fps": 10,
                "frame_count": len(clip_images),
            },
            "current_frame": torch.tensor([0], device=device),
            "processing_complete": torch.tensor([False], device=device),
        }
        
        # Run tracking on this clip
        tracking_results = []
        
        # Process first frame to initialize tracking
        first_image = clip_images[0].to(device) if isinstance(clip_images[0], torch.Tensor) else transform(clip_images[0]).unsqueeze(0).to(device)
        
        # Get annotations for first frame
        annotations = batch["annotations"][0]["object"]
        first_frame_boxes = []
        
        # Extract bounding boxes from annotations
        for obj_ann in annotations:
            if "bbox" in obj_ann:
                box = obj_ann["bbox"]  # [x, y, w, h]
                first_frame_boxes.append(box)
        
        with torch.no_grad():
            # Run model on first frame
            results_first = model(
                image=first_image,
                memory=None,
                tracking_state=tracking_state,
            )
            
            # Update tracking state with first frame results
            tracking_state = tracker.update(tracking_state, results_first)
            
            # Get object IDs and boxes for visualization
            active_mask = tracking_state["active"][0]
            obj_ids = tracking_state["obj_ids"][0][active_mask].cpu().tolist()
            boxes = tracking_state["boxes"][0][active_mask].cpu().tolist()
            
            # Add results for first frame
            tracking_results.append({
                "frame_idx": 0,
                "obj_ids": obj_ids,
                "boxes": boxes,
            })
            
            # Process remaining frames
            for frame_idx in range(1, len(clip_images)):
                # Update current frame in tracking state
                tracking_state["current_frame"][0] = frame_idx
                
                # Load and preprocess frame
                image = clip_images[frame_idx].to(device) if isinstance(clip_images[frame_idx], torch.Tensor) else transform(clip_images[frame_idx]).unsqueeze(0).to(device)
                
                # Run model inference
                frame_results = model(
                    image=image,
                    memory=tracking_state["memory"],
                    tracking_state=tracking_state,
                )
                
                # Update tracking state
                tracking_state = tracker.update(tracking_state, frame_results)
                
                # Extract active objects
                active_mask = tracking_state["active"][0]
                obj_ids = tracking_state["obj_ids"][0][active_mask].cpu().tolist()
                boxes = tracking_state["boxes"][0][active_mask].cpu().tolist()
                
                # Add results for this frame
                tracking_results.append({
                    "frame_idx": frame_idx,
                    "obj_ids": obj_ids,
                    "boxes": boxes,
                })
        
        # Save tracking results
        result_path = os.path.join(args.output_dir, f"result_sample_{sample_idx}.json")
        with open(result_path, "w") as f:
            json.dump({
                "sample_token": sample_token,
                "tracking_results": tracking_results,
            }, f, indent=2)
        
        # Visualize tracking if requested
        if args.visualize:
            video_path = visualize_tracking(
                clip_images, 
                tracking_results,
                args.output_dir,
                sample_idx
            )
            logger.info(f"Tracking visualization saved to {video_path}")
        
        # Add to overall results
        results.append({
            "sample_token": sample_token,
            "num_frames": len(clip_images),
            "num_objects": len(set(sum([frame["obj_ids"] for frame in tracking_results], []))),
        })
    
    # Save overall results summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "num_samples": len(results),
            "results": results,
        }, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
    logger.info(f"Summary saved to {summary_path}")

def main():
    args = parse_args()
    evaluate_on_nuimages(args)

if __name__ == "__main__":
    main()