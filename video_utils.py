#!/usr/bin/env python3
"""
Video Frame Extraction Utility for X2Sim
----------------------------------------
This utility provides functions to extract frames from YouTube videos or local video files.
"""

import os
import re
import subprocess
import logging
import argparse
import shutil
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("video_utils")

def extract_video_frames(video_url, output_dir=None, num_frames=15):
    """
    Download YouTube video and extract frames.
    
    Args:
        video_url (str): YouTube URL or local video file path
        output_dir (str, optional): Directory to save frames. Defaults to "video_frames" in current directory.
        num_frames (int, optional): Number of frames to extract. Defaults to 15.
    
    Returns:
        str: Path to the directory containing extracted frames, or None if unsuccessful
    """
    logger.info(f"Processing video: {video_url}")
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "video_frames")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Frames will be saved to: {output_dir}")
    
    # Find the path for ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.error("FFmpeg not found. Please install ffmpeg using: pip install ffmpeg-python")
        logger.error("Note: You may also need to install the ffmpeg system binaries")
        return None
    
    # Find the path for yt-dlp
    ytdlp_path = shutil.which("yt-dlp")
    if not ytdlp_path:
        logger.error("yt-dlp not found. Please install yt-dlp using: pip install yt-dlp")
        return None
    
    logger.info(f"Using ffmpeg from: {ffmpeg_path}")
    logger.info(f"Using yt-dlp from: {ytdlp_path}")
    
    # Check if the input is a YouTube URL or a local file
    is_youtube = "youtube.com" in video_url or "youtu.be" in video_url
    
    # Process based on whether it's a YouTube video or local file
    if is_youtube:
        # For YouTube URLs, download the video first
        temp_video = os.path.join(os.getcwd(), "temp_video.mp4")
        logger.info(f"Downloading video from {video_url}...")
        try:
            subprocess.run(
                [ytdlp_path, "-f", "best", "-o", temp_video, video_url],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Video downloaded successfully")
            video_path = temp_video
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download video: {e}")
            return None
    else:
        # For local files, use directly
        if not os.path.exists(video_url):
            logger.error(f"Video file not found: {video_url}")
            return None
        video_path = video_url
    
    try:
        # Get video duration to calculate frame extraction interval
        logger.info("Calculating frame extraction interval...")
        result = subprocess.run(
            [ffmpeg_path, "-i", video_path],
            capture_output=True,
            text=True
        )
        
        # Extract duration from FFmpeg output
        match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        if match:
            hours, minutes, seconds = map(float, match.groups())
            video_duration = hours * 3600 + minutes * 60 + seconds
            logger.info(f"Video duration: {video_duration:.2f} seconds")
            
            # Calculate interval between frames
            interval = video_duration / num_frames
        else:
            logger.warning("Could not determine video duration. Using default interval.")
            interval = 1.0  # Default to 1 second intervals
        
        # Extract frames using FFmpeg
        logger.info(f"Extracting {num_frames} frames at {interval:.2f} second intervals...")
        subprocess.run([
            ffmpeg_path, "-i", video_path,
            "-vf", f"fps=1/{interval}",
            os.path.join(output_dir, "frame_%04d.png")
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Verify frames were extracted
        extracted_frames = list(Path(output_dir).glob("frame_*.png"))
        if not extracted_frames:
            logger.error("No frames were extracted from the video")
            return None
        
        logger.info(f"Successfully extracted {len(extracted_frames)} frames")
        
        # Clean up temporary video file if we downloaded from YouTube
        if is_youtube and os.path.exists(temp_video):
            os.remove(temp_video)
            logger.info("Temporary video file removed")
        
        return output_dir
        
    except Exception as e:
        logger.error(f"Error during frame extraction: {e}")
        # Clean up temporary files
        if is_youtube and os.path.exists(temp_video):
            os.remove(temp_video)
        return None

def main():
    """Command line interface for the frame extraction utility"""
    parser = argparse.ArgumentParser(description="Extract frames from YouTube videos or local video files")
    parser.add_argument("video", help="YouTube URL or path to local video file")
    parser.add_argument("--output", "-o", help="Directory to save extracted frames", default=None)
    parser.add_argument("--frames", "-n", type=int, help="Number of frames to extract", default=15)
    
    args = parser.parse_args()
    
    result = extract_video_frames(args.video, args.output, args.frames)
    if result:
        print(f"Frames successfully extracted to: {result}")
        return 0
    else:
        print("Frame extraction failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())