#!/usr/bin/env python3
"""
Video Frame Extraction Utility for X2Sim
----------------------------------------
This utility provides functions to extract frames from YouTube videos or local video files
without requiring system binaries.
"""

import os
import sys
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
import yt_dlp
import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("video_utils")

def download_youtube_video(youtube_url, output_path="input_video.mp4"):
    """
    Download a YouTube video using yt-dlp Python library
    
    Args:
        youtube_url (str): YouTube URL to download
        output_path (str): Path to save the downloaded video
        
    Returns:
        str: Path to the downloaded video or None if unsuccessful
    """
    try:
        logger.info(f"Downloading video from: {youtube_url}")
        
        # Clean up any existing files with the same name
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Removed existing file: {output_path}")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Prefer MP4 format
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
            'noprogress': False,
        }
        
        # Use yt-dlp to download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading video with format: {ydl_opts['format']}")
            info = ydl.extract_info(youtube_url, download=True)
            logger.info(f"Downloaded video: {info.get('title', 'Unknown title')}")
        
        # Verify the file exists and has content
        if not os.path.exists(output_path):
            alt_path = f"{output_path}.mp4"
            if os.path.exists(alt_path):
                logger.info(f"Found video at alternate path: {alt_path}")
                # Rename to the expected path
                shutil.move(alt_path, output_path)
                logger.info(f"Renamed {alt_path} to {output_path}")
            else:
                logger.error(f"Output file not found at: {output_path} or {alt_path}")
                # Search for other possible filenames
                dir_path = os.path.dirname(output_path) or "."
                files = os.listdir(dir_path)
                logger.info(f"Files in directory: {files}")
                return None
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            logger.error(f"Downloaded file is empty: {output_path}")
            return None
        
        logger.info(f"Successfully downloaded video to {output_path} (size: {file_size} bytes)")
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {str(e)}")
        # Print the full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        return None

def extract_frames_with_cv2(video_path, output_dir, num_frames=15):
    """
    Extract frames from a video file using OpenCV
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted frames
        num_frames (int): Number of frames to extract
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        logger.info(f"Extracting frames from {video_path} using OpenCV")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {frame_count} frames, {fps} fps, {duration:.2f} seconds")
        
        if frame_count <= 0:
            logger.error("Video contains no frames")
            return False
        
        # Calculate frame intervals
        if frame_count <= num_frames:
            # If video has fewer frames than requested, use all frames
            frame_indices = list(range(frame_count))
        else:
            # Calculate evenly spaced frame indices
            frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        logger.info(f"Extracting {len(frame_indices)} frames at indices: {frame_indices}")
        
        # Extract frames
        frames_extracted = 0
        for i, frame_idx in enumerate(frame_indices):
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame at index {frame_idx}")
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save the frame as a PNG image
            frame_path = os.path.join(output_dir, f"frame_{i+1:04d}.png")
            Image.fromarray(frame_rgb).save(frame_path)
            frames_extracted += 1
            
        cap.release()
        
        logger.info(f"Successfully extracted {frames_extracted} frames to {output_dir}")
        return frames_extracted > 0
        
    except Exception as e:
        logger.error(f"Error extracting frames with OpenCV: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

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
    
    # Check if the input is a YouTube URL or a local file
    is_youtube = "youtube.com" in video_url or "youtu.be" in video_url
    
    # Process based on whether it's a YouTube video or local file
    if is_youtube:
        # For YouTube URLs, download the video first
        temp_video = os.path.join(os.getcwd(), "temp_video.mp4")
        logger.info(f"Downloading video from {video_url}...")
        
        # Use the download_youtube_video function to download the video
        video_path = download_youtube_video(youtube_url=video_url, output_path=temp_video)
        
        if not video_path:
            logger.error("Failed to download YouTube video")
            return None
    else:
        # For local files, use directly
        if not os.path.exists(video_url):
            logger.error(f"Video file not found: {video_url}")
            return None
        video_path = video_url
    
    # Extract frames using OpenCV
    success = extract_frames_with_cv2(video_path, output_dir, num_frames)
    
    if not success:
        logger.error("Failed to extract frames from video")
        return None
    
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

def main():
    """Command line interface for the frame extraction utility"""
    parser = argparse.ArgumentParser(description="Extract frames from YouTube videos or local video files")
    parser.add_argument("video", help="YouTube URL or path to local video file")
    parser.add_argument("--output", "-o", help="Directory to save extracted frames", default=None)
    parser.add_argument("--frames", "-n", type=int, help="Number of frames to extract", default=15)
    parser.add_argument("--download-only", "-d", action="store_true", 
                        help="Only download the video without extracting frames")
    
    args = parser.parse_args()
    
    if args.download_only:
        if "youtube.com" in args.video or "youtu.be" in args.video:
            result = download_youtube_video(args.video)
            if result:
                print(f"Video successfully downloaded to: {result}")
                return 0
            else:
                print("Video download failed")
                return 1
        else:
            print("The --download-only option is only applicable for YouTube URLs")
            return 1
    else:
        result = extract_video_frames(args.video, args.output, args.frames)
        if result:
            print(f"Frames successfully extracted to: {result}")
            return 0
        else:
            print("Frame extraction failed")
            return 1

if __name__ == "__main__":
    sys.exit(main())