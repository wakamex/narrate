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
    # 50/50 mixes
    {'af_bella': 0.5, 'am_adam': 0.5, 'name': 'af_bella_am_adam_50_50'},
    {'af_bella': 0.5, 'bm_george': 0.5, 'name': 'af_bella_bm_george_50_50'},
    {'bf_emma': 0.5, 'am_adam': 0.5, 'name': 'bf_emma_am_adam_50_50'},
    {'bf_emma': 0.5, 'bm_george': 0.5, 'name': 'bf_emma_bm_george_50_50'},
    
    # 30/70 mixes (slightly masculine)
    {'af_bella': 0.3, 'am_adam': 0.7, 'name': 'af_bella_am_adam_30_70'},
    {'bf_emma': 0.3, 'bm_george': 0.7, 'name': 'bf_emma_bm_george_30_70'},
    
    # 70/30 mixes (slightly feminine)
    {'af_bella': 0.7, 'am_adam': 0.3, 'name': 'af_bella_am_adam_70_30'},
    {'bf_emma': 0.7, 'bm_george': 0.3, 'name': 'bf_emma_bm_george_70_30'},
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
