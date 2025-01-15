import argparse
import os
import re
import subprocess

from dotenv import load_dotenv

from narrate.generate_images import load_prompts


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert a timestamp (MM:SS) to seconds."""
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

def create_video(image_dir: str, prompt_file: str, output_file: str = 'book_video.mp4', fps: int = 30, audio_file: str = None):
    """Create a video from a directory of images using timestamps from filenames and prompts."""
    # Load all prompts to get timing information
    prompts = load_prompts(prompt_file)

    # Create a temporary file for ffmpeg input
    input_file = 'ffmpeg_input.txt'
    with open(input_file, 'w') as f:
        # Get all PNG files in the directory
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not image_files:
            raise ValueError(f"No PNG files found in {image_dir}")

        # Write duration for each image
        for i, filename in enumerate(image_files):
            # Extract timestamp from filename (e.g. h00_m25_00.png -> 25:00)
            match = re.search(r'h\d+_m(\d+)_(\d+)\.png', filename)
            if not match:
                raise ValueError(f"Invalid filename format: {filename}")

            minutes, seconds = match.groups()
            timestamp = f"{minutes}:{seconds}"
            start_time = timestamp_to_seconds(timestamp)

            # Find end time from prompts
            end_time = None
            for prompt in prompts:
                if prompt['start_time'] == timestamp:
                    end_time = timestamp_to_seconds(prompt['end_time'])
                    break

            if end_time is None:
                raise ValueError(f"Could not find timing for image: {filename}")

            duration = end_time - start_time
            f.write(f"file '{os.path.join(image_dir, filename)}'\n")
            f.write(f"duration {duration}\n")

    # Base ffmpeg command for video
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'concat',
        '-safe', '0',
        '-i', input_file,
    ]
    
    # Add audio input if provided
    if audio_file:
        cmd.extend(['-i', audio_file])
        # Map both video and audio, using shortest flag to prevent extending beyond images
        cmd.extend([
            '-map', '0:v',  # First input's video
            '-map', '1:a',  # Second input's audio
            '-shortest',    # End when shortest input ends
            '-vf', 'format=yuv420p'  # Video format filter (after mapping)
        ])
    else:
        # Just add video format filter for video-only output
        cmd.extend(['-vf', 'format=yuv420p'])
    
    # Add output file
    cmd.append(output_file)

    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved to {output_file}")
    finally:
        # Clean up temporary file
        os.remove(input_file)

if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description='Create video from book narration images')
    parser.add_argument('--image-dir', type=str, default='book_images',
                      help='Directory containing the generated images')
    parser.add_argument('--prompt-file', type=str, default='narrate/text/prompts.txt',
                      help='Path to the prompts file')
    parser.add_argument('--output', type=str, default='book_video.mp4',
                      help='Output video file path')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second for output video')
    parser.add_argument('--audio', type=str,
                      help='Optional audio file to add to the video')

    args = parser.parse_args()

    create_video(
        image_dir=args.image_dir,
        prompt_file=args.prompt_file,
        output_file=args.output,
        fps=args.fps,
        audio_file=args.audio
    )
