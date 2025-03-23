#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import torch
import logging

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modt.base_model import SAM2EnhancedModel
from modt.memory_attention import EnhancedMemoryAttention
from modt.memory_encoder import EnhancedMemoryEncoder
from modt.detr_decoder import DETRDecoder
from modt.detection_head import DetectionHead, DetectionRefinementHead

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_image_encoder():
    """Create a dummy image encoder for testing."""
    class DummyImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed_dim = 256
            
        def forward(self, x):
            b, c, h, w = x.shape
            # Create dummy patch features
            patch_features = torch.randn(b, (h//14)*(w//14), self.patch_embed_dim, device=x.device)
            # Create dummy high-res features
            high_res_features = torch.randn(b, 64, h//4, w//4, device=x.device)
            return {
                "patch_features": patch_features,
                "high_res_features": high_res_features
            }
    
    return DummyImageEncoder()

def create_dummy_mask_decoder():
    """Create a dummy mask decoder for testing."""
    class DummyMaskDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, image_embeddings, point_coords=None, point_labels=None, boxes=None, mask_input=None):
            if isinstance(image_embeddings, dict):
                patch_features = image_embeddings["patch_features"]
            else:
                patch_features = image_embeddings
                
            b = patch_features.shape[0]
            return {
                "masks": torch.randn(b, 1, 256, 256, device=patch_features.device),
                "iou_scores": torch.randn(b, 1, device=patch_features.device)
            }
    
    return DummyMaskDecoder()

def test_model_forward():
    """Test the forward pass of the SAM2EnhancedModel."""
    # Create model components
    image_encoder = create_dummy_image_encoder()
    mask_decoder = create_dummy_mask_decoder()
    
    # Create the model
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
    
    # Create a dummy input
    batch_size = 2
    image = torch.randn(batch_size, 3, 448, 448)  # 448 is divisible by 14 (DINOv2 patch size)
    
    # Test forward pass without memory
    logger.info("Testing forward pass without memory...")
    try:
        outputs = model(image=image)
        logger.info("Forward pass without memory succeeded!")
        logger.info(f"Output keys: {outputs.keys()}")
    except Exception as e:
        logger.error(f"Forward pass without memory failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Test forward pass with memory
    logger.info("Testing forward pass with memory...")
    try:
        memory = torch.randn(batch_size, 10, 256)  # 10 memory vectors
        outputs = model(image=image, memory=memory)
        logger.info("Forward pass with memory succeeded!")
        logger.info(f"Output keys: {outputs.keys()}")
    except Exception as e:
        logger.error(f"Forward pass with memory failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    return True

def test_dinov2_encoder():
    """Test with DINOv2-style output format."""
    # Create a dummy DINOv2-style encoder that just returns a tensor
    class DINOv2Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed_dim = 384
            
        def forward(self, x):
            b, c, h, w = x.shape
            # DINOv2 returns just the patch features (not a dict)
            return torch.randn(b, (h//14)*(w//14), self.patch_embed_dim, device=x.device)
    
    image_encoder = DINOv2Encoder()
    mask_decoder = create_dummy_mask_decoder()
    
    # Create the model
    model = SAM2EnhancedModel(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        embed_dim=256,
        memory_dim=256,
        num_heads=8,
        num_decoder_layers=6,
        num_queries=100,
        num_classes=80,
        use_refinement=False,  # False because we don't have high_res_features
    )
    
    # Create a dummy input
    batch_size = 2
    image = torch.randn(batch_size, 3, 448, 448)
    
    # Test forward pass with DINOv2-style encoder
    logger.info("Testing forward pass with DINOv2-style encoder...")
    try:
        outputs = model(image=image)
        logger.info("Forward pass with DINOv2-style encoder succeeded!")
        logger.info(f"Output keys: {outputs.keys()}")
        return True
    except Exception as e:
        logger.error(f"Forward pass with DINOv2-style encoder failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Testing SAM2EnhancedModel...")
    
    # Run tests
    standard_test = test_model_forward()
    dinov2_test = test_dinov2_encoder()
    
    # Show summary
    logger.info("Test results:")
    logger.info(f"  Standard test: {'PASSED' if standard_test else 'FAILED'}")
    logger.info(f"  DINOv2 test: {'PASSED' if dinov2_test else 'FAILED'}")
    
    if standard_test and dinov2_test:
        logger.info("All tests passed! The model is working correctly.")
    else:
        logger.error("Some tests failed. Please check the logs for details.")