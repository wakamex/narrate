#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import signal
import sys

def signal_handler(signum, frame):
    print("\nInterrupt received, cleaning up...")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def remove_deadair(input_file, noise_threshold=-45, min_silence=0.04, volume=1.002):
    """Remove silence/dead air from an audio file.

    Output will be in the same folder with '_trimmed' added to the filename.
    
    Args:
        input_file: Path to input audio file
        noise_threshold: Threshold in dB below which is considered silence (default: -45)
        min_silence: Minimum duration of silence in seconds (default: 0.04)
        volume: Volume adjustment factor (default: 1.002)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Automatically generate output filename in same folder
    output_file = str(input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}")
    
    # First pass: get duration
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    
    try:
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        print(f"Input duration: {duration:.2f} seconds")
    except:
        print("Could not determine input duration")
        duration = None
    
    # Main processing command
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-af', f"silencedetect=noise={noise_threshold}dB:d={min_silence},"
               f"aformat=sample_fmts=fltp:sample_rates=48000,"
               f"areverse,silencedetect=noise={noise_threshold}dB:d={min_silence},"
               f"areverse,asetpts=PTS-STARTPTS,"
               f"volume={volume}",
        '-progress', 'pipe:1',  # Output progress to stdout
        output_file
    ]
    
    print(f"\nProcessing {input_file}...")
    print(f"Output will be saved as {output_file}")
    print("This may take a while for long files...")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line.startswith('out_time_ms='):
                current_time = float(line.split('=')[1]) / 1000000
                if duration:
                    progress = (current_time / duration) * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                else:
                    print(f"\rProcessed: {current_time:.1f} seconds", end='', flush=True)
        
        # Get the return code
        return_code = process.poll()
        if return_code == 0:
            print(f"\nSuccessfully created {output_file}")
        else:
            error = process.stderr.read()
            print(f"\nError during processing (code {return_code}): {error}")
            raise subprocess.CalledProcessError(return_code, cmd)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user, cleaning up...")
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Remove dead air from audio files')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('-n', '--noise', type=float, default=-45,
                      help='Noise threshold in dB (default: -45)')
    parser.add_argument('-d', '--duration', type=float, default=0.04,
                      help='Minimum silence duration in seconds (default: 0.04)')
    parser.add_argument('-v', '--volume', type=float, default=1.002,
                      help='Volume adjustment factor (default: 1.002)')
    
    args = parser.parse_args()
    
    try:
        remove_deadair(
            args.input,
            args.noise,
            args.duration,
            args.volume
        )
    except KeyboardInterrupt:
        print("\nProcess cancelled by user")
        sys.exit(1)

if __name__ == '__main__':
    main()
