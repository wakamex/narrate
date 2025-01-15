import torch
from models import build_model
from kokoro import generate
from mix_voices import mix_voices
import soundfile as sf
import subprocess
from itertools import combinations

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)

# Read sample text
with open('sample.txt', 'r') as f:
    text = f.read()

# Base voices to mix
voices = ['af_bella', 'am_adam', 'bm_george', 'bf_emma']

# Generate all 3-way combinations
three_way_combos = list(combinations(voices, 3))
mix_configs = []

# Add 3-way combinations with equal weights
for combo in three_way_combos:
    weight = 1.0 / 3.0  # Equal weights for each voice
    mix_configs.append({
        'weights': {voice: weight for voice in combo},
        'name': f"{combo[0].split('_')[1]}_{combo[1].split('_')[1]}_{combo[2].split('_')[1]}_equal"
    })

# Add 4-way combination with equal weights
mix_configs.append({
    'weights': {voice: 0.25 for voice in voices},  # 25% each
    'name': 'bella_adam_george_emma_equal'
})

# Process each combination
for config in mix_configs:
    name = config['name']
    weights = config['weights']
    print(f'Processing {name}...')
    
    # Create mixed voice
    voicepack = mix_voices(weights, device=device)
    
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
