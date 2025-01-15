import torch
from models import build_model
from kokoro import generate
from mix_voices import mix_voices
import soundfile as sf
import subprocess

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)

# Read sample text
with open('sample.txt', 'r') as f:
    text = f.read()

# Define the combinations to test
combinations = [
    # Bella/George combinations
    {'af_bella': 0.3, 'bm_george': 0.7, 'name': 'af_bella_bm_george_30_70'},
    {'af_bella': 0.4, 'bm_george': 0.6, 'name': 'af_bella_bm_george_40_60'},
    
    # Bella/Adam combination
    {'af_bella': 0.4, 'am_adam': 0.6, 'name': 'af_bella_am_adam_40_60'},
]

# Process each combination
for combo in combinations:
    name = combo.pop('name')
    print(f'Processing {name}...')
    
    # Create mixed voice
    voicepack = mix_voices(combo, device=device)
    
    # Generate audio
    audio, _ = generate(
        model=MODEL,
        text=text,
        voicepack=voicepack,
        lang='a',  # Use American English for consistency
    )
    
    # Save as WAV
    wav_file = f'samples/{name}.wav'
    sf.write(wav_file, audio, 24000)
    
    # Convert to MP3
    mp3_file = f'samples/{name}.mp3'
    subprocess.run(['ffmpeg', '-i', wav_file, mp3_file])
    
print('Done processing all combinations!')
