#!/bin/bash

# Check if number of videos is provided as argument
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [number_of_videos]"
    echo "  number_of_videos: Optional. Number of videos to process. If not provided, processes all videos."
    exit 0
fi

# Directory containing the videos
VIDEO_DIR="data/output_videos"
# Directory containing the frames
FRAME_DIR="data/first_frames"
# Base config file
CONFIG_FILE="configs/IP-multicontrolnet.yaml"
# Directory for temporary configs
TMP_CONFIG_DIR="tmp_configs"

# Create temporary config directory
mkdir -p $TMP_CONFIG_DIR

# Get list of all videos
videos=($(ls $VIDEO_DIR/*.mp4 | sort))

# Determine how many videos to process
total_videos=${#videos[@]}
if [ -n "$1" ] && [ "$1" -gt 0 ]; then
    if [ "$1" -lt "$total_videos" ]; then
        videos=("${videos[@]:0:$1}")
        echo "Processing first $1 videos out of $total_videos total videos"
    else
        echo "Requested $1 videos but only $total_videos available. Processing all videos."
    fi
else
    echo "Processing all $total_videos videos"
fi

for ((i=0; i<${#videos[@]}; i++)); do
    current_video=$(basename "${videos[$i]}" .mp4)
    
    # Calculate the index for the image prompt (+2)
    prompt_idx=$((10#$current_video + 5))
    prompt_image=$(printf "%05d" $prompt_idx)
    
    # Check if prompt image exists
    if [ ! -f "$FRAME_DIR/${prompt_image}.png" ]; then
        echo "Warning: No prompt image found for ${prompt_image}.png, skipping..."
        continue
    fi
    
    echo "Processing video $current_video with prompt from $prompt_image ($((i+1))/${#videos[@]})"
    
    # Create temporary config file
    tmp_config="$TMP_CONFIG_DIR/config_${current_video}.yaml"
    cp "$CONFIG_FILE" "$tmp_config"
    
    # Update video name and image prompt in the config
    sed -i "s/video_name: .*/video_name: \"$current_video\"/" "$tmp_config"
    sed -i "s/image_prompt: .*/image_prompt: \"$prompt_image\"/" "$tmp_config"
    
    # Run the main script with the modified config
    python run_experiment.py "$tmp_config"
    
    echo "Completed processing $current_video"
done

# Clean up temporary configs
rm -rf $TMP_CONFIG_DIR
rm -rf generated/