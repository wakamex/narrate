import os

import torch

from narrate.kokoro import build_model
from narrate.narrate_book import generate_with_cleanup
from narrate.voices import mix_voices


def test_simple_generation():
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Kokoro-82M', 'kokoro-v0_19.pth')
    model = build_model(model_path, device)
    
    print("\nCreating mixed voice...")
    voice_weights = {
        'af_bella': 0.4,
        'bm_george': 0.6
    }
    voicepack = mix_voices(voice_weights, device=device)
    
    # Test simple phrases
    test_phrases = [
        "had explained it",  # Simple phrase we expect to work
        "Thufir Hawat",     # Proper noun we expect to fail
        "the planet",       # Another simple phrase
        "CHOAM"            # Another proper noun
    ]
    
    for phrase in test_phrases:
        print(f"\nTesting phrase: {phrase}")
        try:
            result = generate_with_cleanup(model, phrase, voicepack, 'a')
            if result[0] is not None:
                print("✓ Success!")
            else:
                print("✗ Failed (returned None)")
        except Exception as e:
            print(f"✗ Failed with error: {str(e)}")

if __name__ == '__main__':
    test_simple_generation()
