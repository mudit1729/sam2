# SAM2 Enhanced Object Tracking

This extension to SAM2 implements a memory feedback loop architecture for enhanced object tracking, especially for small objects.

## Architecture Overview

The enhanced architecture builds on SAM2 by adding several key components:

1. **Enhanced Memory Attention**: Conditions current frame features with memory vectors from previous frames
2. **DETR Decoder**: Uses object queries to extract detection features from memory-conditioned features
3. **Object Detection Head**: Predicts object classes and bounding boxes
4. **Object Refinement Head**: Specifically improves small object detection using high-resolution features
5. **Enhanced Memory Encoder**: Encodes detections with patch embeddings to create memory vectors
6. **Memory Feedback Loop**: Connects memory bank back to memory attention

### Flow Diagram
```
Input Frame -> DINOv2 Encoder -> Memory Attention -> DETR Decoder -> 
Object Detection Head -> Detection Refinement -> Memory Encoder -> 
Memory Bank -> Memory Attention (loop back)
```

## Key Files

1. `sam2/modeling/detr_decoder.py` - DETR-style transformer decoder for object detection
2. `sam2/modeling/detection_head.py` - Object detection and refinement heads
3. `sam2/modeling/enhanced_memory_attention.py` - Memory attention with previous frame conditioning
4. `sam2/modeling/enhanced_memory_encoder.py` - Memory encoder that combines detections with features
5. `sam2/modeling/sam2_base_enhanced.py` - Enhanced SAM2 base model integrating all components
6. `sam2/enhanced_video_predictor.py` - Video predictor that uses the enhanced model

## How It Works

1. **Memory Conditioning**: The current frame features are conditioned with memory vectors from previous frames using cross-attention.
2. **Object Queries**: Learned object queries interact with memory-conditioned features through the DETR decoder.
3. **Object Detection**: The detection head predicts bounding boxes and classes from the decoder outputs.
4. **Small Object Refinement**: High-resolution features are used to refine small object detections.
5. **Memory Generation**: Detections and patch embeddings are combined to create memory vectors.
6. **Memory Feedback**: Memory vectors are stored in a memory bank and fed back to memory attention for future frames.

## Usage

The enhanced model maintains the same interface as the original SAM2, allowing for easy integration:

```python
# Initialize the model
model = SAM2EnhancedVideoPredictor.from_pretrained("model_id")

# Initialize tracking state
state = model.init_state("video.mp4")

# Add points or box for tracking
frame_idx, obj_ids, masks, detections = model.add_new_points_or_box(
    state, frame_idx=0, obj_id=1, points=points, labels=labels
)

# Propagate in video
for frame_idx, obj_ids, masks, detections in model.propagate_in_video(state):
    # Process results
    process_results(frame_idx, obj_ids, masks, detections)
```

## Extensions

This implementation can be extended in several ways:

1. **Multi-object Tracking**: The model is designed to handle multiple objects with separate memory banks.
2. **Long-term Memory**: The memory bank could be enhanced to store long-term memories for better tracking.
3. **Motion Classification**: Objects could be classified as static or dynamic for specialized tracking.
4. **Memory Selection**: More sophisticated memory selection strategies could be implemented.

## Model Performance

The enhanced architecture is expected to improve tracking performance, especially for:

1. Small objects
2. Occluded objects
3. Fast-moving objects
4. Objects with similar appearance

This implementation provides a foundation for further research and development in video object tracking.