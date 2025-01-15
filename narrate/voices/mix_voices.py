import os
import torch

def mix_voices(voices_and_weights, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Mix multiple voices according to specified weights.
    
    Args:
        voices_and_weights (dict): Dictionary mapping voice names to weights
                                 e.g. {'am_adam': 1.0, 'af_bella': 0.5}
        device (str): Device to load tensors on
        
    Returns:
        torch.Tensor: Mixed voice tensor
    """
    # Get path to voices directory
    voices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Kokoro-82M', 'voices')
    
    # Normalize weights to sum to 1
    total_weight = sum(voices_and_weights.values())
    normalized_weights = {k: v/total_weight for k, v in voices_and_weights.items()}
    
    # Initialize mixed voice with first voice
    mixed_voice = None
    
    # Load and mix voices
    for voice_name, weight in normalized_weights.items():
        voice_path = os.path.join(voices_dir, f"{voice_name}.pt")
        voice = torch.load(voice_path, map_location=device)
        if mixed_voice is None:
            mixed_voice = voice * weight
        else:
            mixed_voice += voice * weight
    
    return mixed_voice
