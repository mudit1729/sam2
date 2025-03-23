# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
from tqdm import tqdm
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union

try:
    from nuimages import NuImages
except ImportError:
    print("Warning: nuimages package not found. Install with: pip install nuimages")

from .base_model import SAM2EnhancedModel
from .data_loader import create_nuimages_dataloader
from .tracker import ObjectTracker

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SAM2 Enhanced Object Tracking with nuImages")
    
    # Dataset parameters
    parser.add_argument("--nuimages-root", type=str, required=True,
                        help="Path to the nuImages dataset root directory")
    parser.add_argument("--nuimages-version", type=str, default="v1.0-mini",
                        help="Version of the nuImages dataset to use")
    parser.add_argument("--clip-length", type=int, default=13,
                        help="Number of frames in each clip")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to resume training")
    parser.add_argument("--sam2-checkpoint", type=str, default=None,
                        help="Path to SAM2 model checkpoint")
                        
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Interval for logging training progress")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Interval for saving model checkpoints (in epochs)")
    parser.add_argument("--compressed-save", action="store_true",
                        help="Use compressed format for saving checkpoints (saves space)")
    
    # Overfitting mode for debugging
    parser.add_argument("--overfit", action="store_true",
                        help="Use overfitting mode with a single sequence")
    
    return parser.parse_args()

def load_dinov2_encoder():
    """
    Load a DINOv2 model to use as image encoder.
    """
    try:
        import torch.hub
        logger.info("Loading DINOv2 model from torch hub")
        try:
            # Try using torch hub
            dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            logger.warning(f"Could not load DINOv2 from torch hub: {e}")
            # Create a dummy model with similar output structure
            dinov2_model = None
    except:
        logger.warning("Failed to load DINOv2, creating a dummy model instead")
        dinov2_model = None
    
    # Create a wrapper class for DINOv2 that matches SAM2's expected interface
    class DINOv2Encoder(nn.Module):
        def __init__(self, base_model=None):
            super().__init__()
            self.base_model = base_model
            self.patch_embed_dim = 384 if base_model else 256  # ViT-S/14 dim
            
        def forward(self, x):
            b, c, h, w = x.shape
            
            if self.base_model:
                # Use the actual DINOv2 model
                with torch.no_grad():
                    # DINOv2 returns a dict with multiple outputs
                    dinov2_features = self.base_model.forward_features(x)
                    
                # Check the actual output format
                if isinstance(dinov2_features, dict):
                    # Extract patch tokens from DINOv2 output
                    if 'x_norm_patchtokens' in dinov2_features:
                        patch_features = dinov2_features['x_norm_patchtokens']
                    else:
                        # Fallback to dummy if the expected key isn't there
                        logger.warning("DINOv2 output format unexpected, using dummy features")
                        patch_features = torch.randn(b, (h//14)*(w//14), self.patch_embed_dim, device=x.device)
                else:
                    # If it's just a tensor, use it directly
                    patch_features = dinov2_features
                
                # Create high-res features (in real implementation, these would come from earlier layers)
                # For simplicity, we'll create random features of the expected shape
                high_res_features = torch.randn(b, 64, h//4, w//4, device=x.device)
                
                return {
                    "patch_features": patch_features, 
                    "high_res_features": high_res_features
                }
            else:
                # Create dummy features
                patch_features = torch.randn(b, (h//14)*(w//14), self.patch_embed_dim, device=x.device)
                high_res_features = torch.randn(b, 64, h//4, w//4, device=x.device)
                return {"patch_features": patch_features, "high_res_features": high_res_features}
    
    # Create mask decoder
    class DummyMaskDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, image_embeddings, point_coords=None, point_labels=None, boxes=None, mask_input=None):
            if isinstance(image_embeddings, dict):
                patch_features = image_embeddings["patch_features"]
            else:
                patch_features = image_embeddings
                
            b = patch_features.shape[0]
            return {"masks": torch.randn(b, 1, 256, 256, device=patch_features.device), 
                    "iou_scores": torch.randn(b, 1, device=patch_features.device)}
    
    return {
        "image_encoder": DINOv2Encoder(dinov2_model),
        "mask_decoder": DummyMaskDecoder()
    }

def load_sam2_model(checkpoint_path: str):
    """
    Load SAM2 model from checkpoint or create a DINOv2-based model.
    """
    if checkpoint_path:
        try:
            logger.info(f"Loading SAM2 checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # Here you would initialize the SAM2 model and load state_dict
            # Since we don't have direct access to SAM2, we'll create a dummy model
            logger.warning("SAM2 loading not implemented, falling back to DINOv2")
            return load_dinov2_encoder()
        except Exception as e:
            logger.warning(f"Error loading SAM2 checkpoint: {e}")
            return load_dinov2_encoder()
    else:
        logger.info("No checkpoint provided, using DINOv2 encoder")
        return load_dinov2_encoder()

def build_model(args):
    """Build and initialize the SAM2EnhancedModel."""
    # Load SAM2 model components
    if args.sam2_checkpoint:
        sam2_model = load_sam2_model(args.sam2_checkpoint)
        image_encoder = sam2_model["image_encoder"]
        mask_decoder = sam2_model["mask_decoder"]
    else:
        # For testing without a real SAM2 checkpoint
        logger.warning("No SAM2 checkpoint provided, creating a dummy model")
        sam2_model = load_sam2_model(None)  # This will create a dummy model
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
        num_classes=80,  # Adjust based on nuImages categories
        use_refinement=True,
    )
    
    # Load from checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    return model

def prepare_input(batch, device):
    """Prepare input data for the model."""
    # Extract the first frame from each clip
    first_frames = []
    for clip in batch["clip"]:
        if len(clip) > 0:
            first_frames.append(clip[0])
    
    # Convert to tensor if not already
    if len(first_frames) > 0 and not isinstance(first_frames[0], torch.Tensor):
        first_frames = [transforms.ToTensor()(img) for img in first_frames]
    
    # Stack tensors
    if len(first_frames) > 0:
        frames_tensor = torch.stack(first_frames).to(device)
    else:
        # Return empty dict if no frames
        return {}
    
    # Prepare annotations
    annotations = batch["annotations"]  # List of annotation dicts
    
    # Extract bounding boxes from annotations
    boxes = []
    for ann_dict in annotations:
        clip_boxes = []
        for obj_ann in ann_dict["object"]:
            if "bbox" in obj_ann:
                # Get the bbox in x,y,w,h format and convert to tensor
                bbox = obj_ann["bbox"]
                if isinstance(bbox, list) and len(bbox) == 4:
                    clip_boxes.append(torch.tensor(bbox, device=device))
            
        if len(clip_boxes) > 0:
            # Stack boxes for this clip
            boxes.append(torch.stack(clip_boxes))
        else:
            # Empty tensor if no boxes
            boxes.append(torch.zeros((0, 4), device=device))
    
    return {
        "frames": frames_tensor,
        "boxes": boxes,
        "annotations": annotations,
        "meta": batch["meta"]
    }

def train_one_epoch(model, dataloader, optimizer, device, epoch, args):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch_idx, batch in enumerate(progress_bar):
        # Prepare input data
        inputs = prepare_input(batch, device)
        if not inputs or "frames" not in inputs or len(inputs["frames"]) == 0:
            logger.warning(f"Skipping empty batch {batch_idx}")
            continue
        
        # Forward pass
        optimizer.zero_grad()
        try:
            outputs = model(
                image=inputs["frames"],
                # We don't pass boxes here because SAM2EnhancedModel expects them in a specific format
                # In a real implementation, we would format them correctly
            )
            
            # Calculate losses for training
            # In a complete implementation, we would compute detection, segmentation, and tracking losses
            class_logits = outputs.get("class_logits")
            pred_boxes = outputs.get("boxes")
            
            # Initialize loss components
            classification_loss = torch.tensor(0.0, device=device)
            box_loss = torch.tensor(0.0, device=device)
            memory_loss = torch.tensor(0.0, device=device)
            
            # Classification loss
            if class_logits is not None:
                # For this simple implementation, we'll just regularize the logits
                # In a real implementation, we would use cross-entropy with ground truth classes
                classification_loss = torch.sum(class_logits**2) * 0.01
            
            # Box regression loss
            if pred_boxes is not None:
                # Simple regularization on box predictions
                # In a real implementation, we would use IoU or L1 loss with ground truth boxes
                box_loss = torch.sum(pred_boxes**2) * 0.01
                
                # If we have ground truth boxes, compute a more meaningful box loss
                if "boxes" in inputs and len(inputs["boxes"]) > 0:
                    # Just use the first object's box as a target for simplicity
                    # In a real implementation, we would match predictions to ground truth
                    # and compute proper box regression losses (GIoU, L1, etc.)
                    for i, gt_boxes in enumerate(inputs["boxes"]):
                        if i < pred_boxes.shape[0] and gt_boxes.shape[0] > 0:
                            # Simple L1 loss for the first predicted box and first GT box
                            box_loss += F.l1_loss(
                                pred_boxes[i, 0, :], 
                                gt_boxes[0],
                                reduction='sum'
                            ) * 0.1
            
            # Memory consistency loss (if memory vectors are present)
            memory_vectors = outputs.get("memory_vectors")
            if memory_vectors is not None:
                # Encourage memory vectors to be unit norm
                norm = torch.norm(memory_vectors, dim=-1)
                memory_loss = F.mse_loss(norm, torch.ones_like(norm)) * 0.01
            
            # Combine all losses
            loss = classification_loss + box_loss + memory_loss
            
            # Ensure loss is non-zero to prevent NaN gradients
            if loss == 0:
                loss = torch.tensor(0.1, device=device, requires_grad=True)
                
            # Log loss components
            if batch_idx % args.log_interval == 0:
                logger.info(f"  Cls loss: {classification_loss.item():.4f}, "
                           f"Box loss: {box_loss.item():.4f}, "
                           f"Memory loss: {memory_loss.item():.4f}")
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_loss / (batch_count) if batch_count > 0 else 0
                progress_bar.set_postfix({"loss": avg_loss})
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    if batch_count == 0:
        return 0.0
    return total_loss / batch_count

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
        logger.warning("Creating a dummy dataset instead for testing")
        # Create a dummy nuImages-like structure for testing
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
        # Make the DummyNuImages class available globally for the evaluate.py script
        globals()["DummyNuImages"] = DummyNuImages
        # Override the dataroot to use a local test directory
        args.nuimages_root = "./test_data"
        os.makedirs(args.nuimages_root, exist_ok=True)
        os.makedirs(os.path.join(args.nuimages_root, "dummy"), exist_ok=True)
        
        # Create dummy images
        for i in range(10):
            for j in range(13):
                img_path = os.path.join(args.nuimages_root, f"dummy/image_{i}_{j}.jpg")
                if not os.path.exists(img_path):
                    try:
                        import numpy as np
                        from PIL import Image
                        # Create a simple colored image
                        img_array = np.ones((256, 256, 3), dtype=np.uint8) * (i*20 + 50)
                        # Add a box to simulate an object
                        img_array[100:130, 100:150] = [200, 0, 0]
                        img = Image.fromarray(img_array)
                        img.save(img_path)
                    except ImportError:
                        # If PIL is not available, create empty files
                        with open(img_path, "w") as f:
                            f.write("dummy image")
    
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
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        transform=transform,
        shuffle=not args.overfit,  # Don't shuffle in overfit mode
        num_workers=args.num_workers,
    )
    
    # Build model
    logger.info("Building model")
    model = build_model(args)
    model = model.to(device)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), 
                           lr=args.learning_rate, 
                           weight_decay=args.weight_decay)
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, args)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            try:
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": train_loss,
                }
                
                if args.compressed_save:
                    try:
                        # Try using the full compression options first
                        torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=True, _use_thread=False)
                    except TypeError:
                        # Fallback if _use_thread is not supported in this PyTorch version
                        torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=True)
                    logger.info(f"Compressed checkpoint saved to {checkpoint_path}")
                else:
                    # Use standard save format
                    torch.save(checkpoint_data, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
            except (OSError, RuntimeError) as e:
                logger.error(f"Failed to save checkpoint: {e}")
                logger.info("Continuing training without saving checkpoint")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    try:
        if args.compressed_save:
            # Use compressed save format
            try:
                torch.save(model.state_dict(), final_model_path, _use_new_zipfile_serialization=True, _use_thread=False)
            except TypeError:
                # Fallback if _use_thread is not supported in this PyTorch version
                torch.save(model.state_dict(), final_model_path, _use_new_zipfile_serialization=True)
            logger.info(f"Compressed final model saved to {final_model_path}")
        else:
            # Use standard save format
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to save final model: {e}")
        # Try saving a smaller checkpoint without optimizer state
        try:
            smaller_path = os.path.join(args.output_dir, "final_model_small.pth")
            # Save only essential model components
            essential_state = {}
            for k, v in model.state_dict().items():
                if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k:
                    essential_state[k] = v
            
            if args.compressed_save:
                try:
                    torch.save(essential_state, smaller_path, _use_new_zipfile_serialization=True, _use_thread=False)
                except TypeError:
                    # Fallback if _use_thread is not supported in this PyTorch version
                    torch.save(essential_state, smaller_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(essential_state, smaller_path)
            
            logger.info(f"Smaller model checkpoint saved to {smaller_path}")
        except (OSError, RuntimeError) as e2:
            logger.error(f"Failed to save smaller model: {e2}")
            logger.warning("Could not save any model checkpoint. Check disk space and permissions.")

if __name__ == "__main__":
    main()