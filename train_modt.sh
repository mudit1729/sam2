#!/bin/bash
# Training script for Memory-based Object Detection and Tracking (MODT)

set -e

# Default values
NUIMAGES_ROOT=""
NUIMAGES_VERSION="v1.0-mini"
BATCH_SIZE=2
EPOCHS=10
LEARNING_RATE=0.0001
OUTPUT_DIR="./outputs/modt"
CLIP_LENGTH=13
NUM_WORKERS=4
CHECKPOINT=""
SAM2_CHECKPOINT=""
OVERFIT=false
COMPRESSED_SAVE=false

# Parse command line arguments
function usage {
  echo "Usage: $0 [OPTIONS]"
  echo "Train a SAM2 Enhanced model for object detection and tracking."
  echo ""
  echo "Options:"
  echo "  --nuimages-root PATH     Path to nuImages dataset root (required)"
  echo "  --nuimages-version VER   nuImages version [default: v1.0-mini]"
  echo "  --batch-size N           Batch size for training [default: 2]"
  echo "  --epochs N               Number of training epochs [default: 10]"
  echo "  --learning-rate LR       Learning rate [default: 0.0001]"
  echo "  --output-dir DIR         Output directory [default: ./outputs/modt]"
  echo "  --clip-length N          Number of frames per clip [default: 13]"
  echo "  --num-workers N          Number of data loading workers [default: 4]"
  echo "  --checkpoint PATH        Path to checkpoint to resume training"
  echo "  --sam2-checkpoint PATH   Path to SAM2 model checkpoint"
  echo "  --overfit                Use a single sequence for overfitting (debugging)"
  echo "  --compressed-save        Use compressed format for saving checkpoints (saves disk space)"
  echo "  --help                   Display this help message"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nuimages-root)
      NUIMAGES_ROOT="$2"
      shift 2
      ;;
    --nuimages-version)
      NUIMAGES_VERSION="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --clip-length)
      CLIP_LENGTH="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --sam2-checkpoint)
      SAM2_CHECKPOINT="$2"
      shift 2
      ;;
    --overfit)
      OVERFIT=true
      shift
      ;;
    --compressed-save)
      COMPRESSED_SAVE=true
      shift
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate required arguments
if [ -z "$NUIMAGES_ROOT" ]; then
  echo "Error: --nuimages-root is required"
  usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python -m modt.train"
CMD+=" --nuimages-root $NUIMAGES_ROOT"
CMD+=" --nuimages-version $NUIMAGES_VERSION"
CMD+=" --batch-size $BATCH_SIZE"
CMD+=" --epochs $EPOCHS"
CMD+=" --learning-rate $LEARNING_RATE"
CMD+=" --output-dir $OUTPUT_DIR"
CMD+=" --clip-length $CLIP_LENGTH"
CMD+=" --num-workers $NUM_WORKERS"

# Add optional arguments
if [ ! -z "$CHECKPOINT" ]; then
  CMD+=" --checkpoint $CHECKPOINT"
fi

if [ ! -z "$SAM2_CHECKPOINT" ]; then
  CMD+=" --sam2-checkpoint $SAM2_CHECKPOINT"
fi

if [ "$OVERFIT" = true ]; then
  CMD+=" --overfit"
fi

if [ "$COMPRESSED_SAVE" = true ]; then
  CMD+=" --compressed-save"
fi

# Print and execute command
echo "Executing: $CMD"
eval $CMD