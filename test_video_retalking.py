#!/usr/bin/env python3
"""
Test script for VideoReTalking using cog container.
This follows the same pattern as the wav2lip implementation.
"""

import subprocess
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_video_retalking(face_video: Path, audio_path: Path, output_path: Path):
    """
    Run VideoReTalking inference using the cog container.

    Args:
        face_video (Path): Path to the video file containing the speaker's face.
        audio_path (Path): Path to the audio file for lip sync.
        output_path (Path): Output path for the final lip-synced video.
    """
    
    # Ensure all paths are absolute
    face_video = face_video.resolve()
    audio_path = audio_path.resolve()
    output_path = output_path.resolve()
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "cog", "run",
        "--gpus", "all",  # Enable GPU access
        "-v", f"{face_video.parent}:/input",
        "-v", f"{audio_path.parent}:/audio", 
        "-v", f"{output_path.parent}:/output",
        "python", "inference.py",
        "--face", f"/input/{face_video.name}",
        "--audio", f"/audio/{audio_path.name}",
        "--outfile", f"/output/{output_path.name}"
    ]

    logger.info(f"[VIDEO-RETALKING] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,  # Run from the video-retalking directory
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"[VIDEO-RETALKING] Output saved to: {output_path}")
        logger.info(f"[VIDEO-RETALKING] stdout:\n{result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[VIDEO-RETALKING] Failed with return code {e.returncode}")
        logger.error(f"stdout:\n{e.stdout}")
        logger.error(f"stderr:\n{e.stderr}")
        raise RuntimeError("VideoReTalking processing failed.") from e

def main():
    """Test the video retalking functionality with sample files."""
    
    # Use the sample files you provided
    face_video = Path("segment_0000.mp4")
    audio_path = Path("segment_0000.wav")
    output_path = Path("output_retalking_test.mp4")
    
    # Check if input files exist
    if not face_video.exists():
        logger.error(f"Face video not found: {face_video}")
        return
    
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return
    
    logger.info(f"Testing VideoReTalking with:")
    logger.info(f"  Face video: {face_video}")
    logger.info(f"  Audio: {audio_path}")
    logger.info(f"  Output: {output_path}")
    
    try:
        run_video_retalking(face_video, audio_path, output_path)
        logger.info("✅ VideoReTalking test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ VideoReTalking test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
