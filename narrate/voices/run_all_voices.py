import torch
from models import build_model
from kokoro import generate
import soundfile as sf
import subprocess

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)

# List of all voices
voices = [
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky'
]

# Read sample text
with open('sample.txt', 'r') as f:
    text = f.read()

# Process each voice
for voice_name in voices:
    print(f'Processing {voice_name}...')
    
    # Load voice pack
    voicepack = torch.load(f'voices/{voice_name}.pt', map_location=device)
    
    # Generate audio
    audio, _ = generate(
        model=MODEL,
        text=text,
        voicepack=voicepack,
        lang=voice_name[0],  # First letter determines language (a/b)
    )
    
    # Save as WAV
    wav_file = f'samples/{voice_name}.wav'
    sf.write(wav_file, audio, 24000)
    
    # Convert to MP3
    mp3_file = f'samples/{voice_name}.mp3'
    subprocess.run(['ffmpeg', '-i', wav_file, mp3_file])
    
print('Done processing all voices!')
