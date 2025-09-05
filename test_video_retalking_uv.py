#!/usr/bin/env python3
"""
Test script for VideoReTalking using uv environment instead of Docker.

This script processes all video/audio pairs in the test_clips folder and saves
both the output videos and debug logs for analysis.
"""

import subprocess
import os
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_video_retalking_with_uv(face_video: Path, audio_path: Path, output_path: Path, debug_path: Path):
    """
    Run VideoReTalking inference using uv virtual environment.

    Args:
        face_video (Path): Path to the video file containing the speaker's face.
        audio_path (Path): Path to the audio file for lip sync.
        output_path (Path): Output path for the final lip-synced video.
        debug_path (Path): Path to save debug output text file.
    """
    
    # Ensure all paths are absolute
    face_video = face_video.resolve()
    audio_path = audio_path.resolve()
    output_path = output_path.resolve()
    debug_path = debug_path.resolve()
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the uv python executable
    uv_python = Path.cwd() / ".venv" / "bin" / "python"
    inference_script = Path.cwd() / "inference.py"
    
    if not uv_python.exists():
        raise RuntimeError(f"UV virtual environment not found at {uv_python}")
    
    if not inference_script.exists():
        raise RuntimeError(f"Inference script not found at {inference_script}")
    
    cmd = [
        str(uv_python),
        str(inference_script),
        "--face", str(face_video),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
        "--pose_pitch_threshold", "30",
        "--pose_yaw_threshold", "60",
        "--re_preprocess",
    ]

    logger.info(f"[VIDEO-RETALKING-UV] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Save debug output to text file
        debug_content = f"""VideoReTalking UV Test Debug Output
Generated: {datetime.now().isoformat()}
Face Video: {face_video}
Audio File: {audio_path}
Output Video: {output_path}

Command: {' '.join(cmd)}

=== STDOUT ===
{result.stdout}

=== STDERR ===
{result.stderr}

=== ANALYSIS ===
"""
        
        # Analyze output for key metrics
        if "Using original frame" in result.stdout:
            original_frame_count = result.stdout.count("Using original frame")
            debug_content += f"Original frames used: {original_frame_count}\n"
        
        if "Final pattern:" in result.stdout:
            pattern_line = [line for line in result.stdout.split('\n') if "Final pattern:" in line]
            if pattern_line:
                debug_content += f"Pose pattern: {pattern_line[0]}\n"
        
        if "Warning: Face enhancement failed" in result.stdout:
            enhancement_warnings = result.stdout.count("Warning: Face enhancement failed")
            debug_content += f"Face enhancement warnings: {enhancement_warnings}\n"
            
        # Write debug output to file
        with open(debug_path, 'w') as f:
            f.write(debug_content)
        
        logger.info(f"[VIDEO-RETALKING-UV] Output saved to: {output_path}")
        logger.info(f"[VIDEO-RETALKING-UV] Debug saved to: {debug_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_content = f"""VideoReTalking UV Test Error Output
Generated: {datetime.now().isoformat()}
Face Video: {face_video}
Audio File: {audio_path}
Output Video: {output_path}

Command: {' '.join(cmd)}
Return Code: {e.returncode}

=== STDOUT ===
{e.stdout}

=== STDERR ===
{e.stderr}
"""
        
        # Write error output to debug file
        with open(debug_path, 'w') as f:
            f.write(error_content)
            
        logger.error(f"[VIDEO-RETALKING-UV] Failed with return code {e.returncode}")
        logger.error(f"Debug output saved to: {debug_path}")
        return False

def find_video_audio_pairs(test_clips_dir: Path):
    """
    Find all matching video/audio pairs in the test_clips directory.
    
    Args:
        test_clips_dir (Path): Directory containing test video and audio files.
        
    Returns:
        list: List of tuples (video_path, audio_path, stem_name)
    """
    pairs = []
    
    # Find all MP4 files
    for video_file in test_clips_dir.glob("*.mp4"):
        stem = video_file.stem
        
        # Look for matching WAV file
        wav_file = test_clips_dir / f"{stem}.wav"
        if wav_file.exists():
            pairs.append((video_file, wav_file, stem))
            logger.info(f"Found pair: {stem}")
        else:
            logger.warning(f"No matching audio for video: {video_file.name}")
    
    return pairs

def main():
    """Test VideoReTalking on all video/audio pairs in test_clips folder using uv environment."""
    
    test_clips_dir = Path("test_clips")
    results_dir = test_clips_dir / "results_uv"
    
    # Check if test_clips directory exists
    if not test_clips_dir.exists():
        logger.error(f"Test clips directory not found: {test_clips_dir}")
        return 1
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video/audio pairs
    pairs = find_video_audio_pairs(test_clips_dir)
    
    if not pairs:
        logger.error("No video/audio pairs found in test_clips directory")
        return 1
    
    logger.info(f"Found {len(pairs)} video/audio pairs to process")
    
    # Process each pair
    success_count = 0
    total_count = len(pairs)
    
    for video_path, audio_path, stem in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {stem}")
        logger.info(f"{'='*60}")
        
        output_video = results_dir / f"{stem}_result.mp4"
        debug_file = results_dir / f"{stem}_debug.txt"
        
        # Skip if output already exists
        if output_video.exists() and not os.getenv("FORCE_REPROCESS"):
            logger.info(f"Skipping {stem} - output already exists (set FORCE_REPROCESS=1 to override)")
            continue
        
        try:
            success = run_video_retalking_with_uv(video_path, audio_path, output_video, debug_file)
            if success:
                success_count += 1
                logger.info(f"✅ {stem} completed successfully")
            else:
                logger.error(f"❌ {stem} failed")
                
        except Exception as e:
            logger.error(f"❌ {stem} failed with exception: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total pairs processed: {total_count}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {total_count - success_count}")
    logger.info(f"Results saved to: {results_dir}")
    
    if success_count == total_count:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"⚠️  {total_count - success_count} tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
