#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import re

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def convert_chunks_to_mp3(wav_dir="book_audio/wav", mp3_dir="book_audio/mp3"):
    # Ensure output directory exists
    Path(mp3_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all chunk files and sort them naturally
    wav_files = [f for f in os.listdir(wav_dir) if f.startswith('chunk_') and f.endswith('.wav')]
    wav_files.sort(key=natural_sort_key)
    
    if not wav_files:
        print("No chunk files found in", wav_dir)
        return
    
    print(f"Found {len(wav_files)} chunk files")
    
    # Create concat file
    with open('concat.txt', 'w') as f:
        for wav_file in wav_files:
            f.write(f"file '{os.path.join(wav_dir, wav_file)}'\n")
    
    # Convert to MP3
    output_file = os.path.join(mp3_dir, 'combined_output.mp3')
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'concat.txt',
        '-c:a', 'libmp3lame',
        '-q:a', '2',
        output_file
    ]
    
    print("Converting to MP3...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up concat file
        try:
            os.remove('concat.txt')
        except:
            pass

if __name__ == '__main__':
    convert_chunks_to_mp3()
