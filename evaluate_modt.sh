#!/bin/bash
# Evaluation script for Memory-based Object Detection and Tracking (MODT)

set -e

# Default values
NUIMAGES_ROOT=""
NUIMAGES_VERSION="v1.0-mini"
CHECKPOINT=""
OUTPUT_DIR="./results/modt"
CLIP_LENGTH=13
NUM_WORKERS=4
VISUALIZATION=true

# Parse command line arguments
function usage {
  echo "Usage: $0 [OPTIONS]"
  echo "Evaluate a trained SAM2 Enhanced model for object detection and tracking."
  echo ""
  echo "Options:"
  echo "  --nuimages-root PATH     Path to nuImages dataset root (required)"
  echo "  --nuimages-version VER   nuImages version [default: v1.0-mini]"
  echo "  --checkpoint PATH        Path to model checkpoint (required)"
  echo "  --output-dir DIR         Output directory [default: ./results/modt]"
  echo "  --clip-length N          Number of frames per clip [default: 13]"
  echo "  --num-workers N          Number of data loading workers [default: 4]"
  echo "  --no-visualization       Disable visualization output"
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
    --checkpoint)
      CHECKPOINT="$2"
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
    --no-visualization)
      VISUALIZATION=false
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

if [ -z "$CHECKPOINT" ]; then
  echo "Error: --checkpoint is required"
  usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python -m modt.evaluate"
CMD+=" --nuimages-root $NUIMAGES_ROOT"
CMD+=" --nuimages-version $NUIMAGES_VERSION"
CMD+=" --checkpoint $CHECKPOINT"
CMD+=" --output-dir $OUTPUT_DIR"
CMD+=" --clip-length $CLIP_LENGTH"
CMD+=" --num-workers $NUM_WORKERS"

# Add visualization flag
if [ "$VISUALIZATION" = false ]; then
  CMD+=" --no-visualization"
fi

# Print and execute command
echo "Executing: $CMD"
eval $CMD