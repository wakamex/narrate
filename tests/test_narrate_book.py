import os
import unittest

import torch

from narrate.kokoro import build_model
from narrate.narrate_book import generate_with_cleanup
from narrate.voices import mix_voices


class TestNarrateBook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across all tests."""
        print("Loading model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Kokoro-82M', 'kokoro-v0_19.pth')
        cls.model = build_model(model_path, device)
        
        print("\nCreating mixed voice...")
        voice_weights = {
            'af_bella': 0.4,
            'bm_george': 0.6
        }
        cls.voicepack = mix_voices(voice_weights, device=device)

    def test_audio_generation_failure(self):
        """Test handling of audio generation failure for a complex text segment."""
        test_segments = [
            # Original long segment that works
            (
                "Thufir Hawat, his father's Master of Assassins, had explained it: "
                "their mortal enemies, the Harkonnens, had been on Arrakis eighty years, "
                "holding the planet in quasi-fief under a CHOAM Company contract to mine "
                "the geriatric spice, melange."
            ),
            # New segment that fails
            (
                "Arrakis would be a place so different from Caladan that Paul's mind "
                "whirled with the new knowledge."
            )
        ]
        
        for segment in test_segments:
            print(f"\nTesting segment: {segment}")
            # Try generating with the actual model
            result = generate_with_cleanup(self.model, segment, self.voicepack, 'a')
            # Either it should succeed or return (None, None)
            if result[0] is not None:
                print("✓ Success: Generated audio for the text!")
            else:
                print("✗ Failed: Could not generate audio for the text")
        
    def test_segment_analysis(self):
        """Test different segments of the text to identify problematic parts"""
        test_segments = [
            # Simple phrases
            "had explained it",
            "their mortal enemies",
            "holding the planet",
            "under a contract",
            
            # Proper nouns from Dune
            "Thufir Hawat",
            "Harkonnens",
            "Arrakis",
            "CHOAM",
            "Caladan",  
            "Paul",     
            
            # Compound words and special terms
            "quasi-fief",
            "geriatric spice",
            
            # Foreign words
            "melange",
            
            # Titles and special capitalization
            "Master of Assassins",
            "CHOAM Company",
            
            # Phrases from failing segment
            "a place so different",
            "Paul's mind whirled",
            "with the new knowledge",
            "would be a place"
        ]
        
        results = {'success': [], 'failure': []}
        
        for segment in test_segments:
            print(f"\nTesting segment: {segment}")
            result = generate_with_cleanup(self.model, segment, self.voicepack, 'a')
            
            if result[0] is not None:
                results['success'].append(segment)
                print("✓ Success!")
            else:
                results['failure'].append(segment)
                print("✗ Failed")
        
        # Print summary of results
        print("\nSegment Analysis Results:")
        print("\nSuccessful segments:")
        for segment in sorted(results['success']):
            print(f"✓ {segment}")
            
        print("\nFailed segments:")
        for segment in sorted(results['failure']):
            print(f"✗ {segment}")
        
        # Print analysis
        print("\nAnalysis:")
        if not results['failure']:
            print("All segments generated successfully!")
            print("\nThis suggests the original error might be caused by:")
            print("1. The length of the combined text")
            print("2. Interaction between multiple terms")
            print("3. Other factors not related to vocabulary")
        else:
            print("Some segments failed to generate:")
            print("1. Failed segments may need special handling")
            print("2. Consider creating a custom pronunciation dictionary")
            print("3. May need to pre-process certain terms")

if __name__ == '__main__':
    unittest.main()
