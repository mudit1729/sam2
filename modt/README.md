# MODT: Memory-based Object Detection and Tracking

This module implements a memory-based enhancement to the SAM2 (Segment Anything 2) model, enabling object detection and tracking across video frames.

## Overview

MODT extends SAM2 with the following capabilities:
- Memory feedback mechanism for temporal consistency
- Object detection with a DETR-style decoder
- Object tracking across video frames
- Integration with DINOv2 as an alternative encoder

## Architecture

The MODT architecture includes the following components:

1. **Base Model** (`SAM2EnhancedModel`): Integrates all components and implements the forward pass
2. **Memory Attention** (`EnhancedMemoryAttention`): Cross-attention between current frame features and memory
3. **Memory Encoder** (`EnhancedMemoryEncoder`): Encodes detection features into memory representations
4. **DETR Decoder** (`DETRDecoder`): Transformer-based decoder for object detection
5. **Detection Head** (`DetectionHead`, `DetectionRefinementHead`): Predicts class probabilities and bounding boxes
6. **Tracker** (`ObjectTracker`): Tracks objects across frames using memory and IoU matching

## Usage

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd sam2

# Install requirements
pip install -e .
```

### Training

To train the model using the nuImages dataset:

```bash
python -m modt.train \
    --nuimages-root /path/to/nuimages \
    --nuimages-version v1.0-mini \
    --batch-size 2 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --output-dir ./outputs \
    --clip-length 13
```

For using DINOv2 as an encoder (no SAM2 checkpoint needed):

```bash
python -m modt.train \
    --nuimages-root /path/to/nuimages \
    --nuimages-version v1.0-mini \
    --batch-size 2 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --output-dir ./outputs
```

### Evaluation

To evaluate a trained model:

```bash
python -m modt.evaluate \
    --nuimages-root /path/to/nuimages \
    --nuimages-version v1.0-mini \
    --checkpoint /path/to/checkpoint.pth \
    --output-dir ./results
```

## Testing

To run basic model tests:

```bash
python -m modt.test_model
```

## Data Format

The model expects input images of size that is divisible by 14 for compatibility with DINOv2's patch size (448×448 recommended). The dataloader handles:

- Image sequences from nuImages
- Bounding box annotations
- Temporal information for tracking

## Model Components

### SAM2EnhancedModel

The main model class that integrates all components:

```python
from modt.base_model import SAM2EnhancedModel

model = SAM2EnhancedModel(
    image_encoder=encoder,  # SAM2 or DINOv2 encoder
    mask_decoder=mask_decoder,
    embed_dim=256,
    memory_dim=256,
    num_heads=8,
    num_decoder_layers=6,
    num_queries=100,
    num_classes=80,
    use_refinement=True,
)

# Forward pass
outputs = model(
    image=image_tensor,  # [B, 3, H, W]
    memory=memory_vectors,  # Optional [B, M, memory_dim]
    tracking_state=None,  # Optional tracking state
)
```

### Training with DINOv2

```python
import torch.hub
import torch.nn as nn
from modt.base_model import SAM2EnhancedModel

# Load DINOv2
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Create a wrapper to match SAM2's interface
class DINOv2Encoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.patch_embed_dim = 384  # ViT-S/14 dim
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Use the actual DINOv2 model
        with torch.no_grad():
            # DINOv2 returns a dict with multiple outputs
            dinov2_features = self.base_model.forward_features(x)
            
        # Extract patch tokens from DINOv2 output
        if isinstance(dinov2_features, dict):
            if 'x_norm_patchtokens' in dinov2_features:
                patch_features = dinov2_features['x_norm_patchtokens']
            else:
                # Handle other output formats if needed
                print("Warning: Unexpected DINOv2 output format")
                patch_features = torch.randn(b, (h//14)*(w//14), self.patch_embed_dim, device=x.device)
        else:
            # If it's just a tensor, use it directly
            patch_features = dinov2_features
        
        # Create high-res features if needed
        high_res_features = torch.randn(b, 64, h//4, w//4, device=x.device)
        
        # Return in the expected format for the model
        return {
            "patch_features": patch_features, 
            "high_res_features": high_res_features
        }

# Create the model
model = SAM2EnhancedModel(
    image_encoder=DINOv2Encoder(dinov2),
    mask_decoder=dummy_mask_decoder,  # Replace with actual mask decoder if available
    embed_dim=256,
    memory_dim=384,  # Match DINOv2's output dimension
    num_heads=8,
    num_decoder_layers=6,
    num_queries=100,
    num_classes=80,
)
```

## Notes

- The model is designed to work with both SAM2's encoder and DINOv2
- When using DINOv2, the patch size is 14×14, so input dimensions should be divisible by 14
- Memory conditioning enables temporal consistency in tracking
- The detection refinement head can improve small object detection when high-resolution features are available

## License

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.