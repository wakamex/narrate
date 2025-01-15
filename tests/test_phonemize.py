import unittest

from narrate.kokoro import phonemize


class TestPhonemize(unittest.TestCase):
    def test_phonemize_with_quotes(self):
        test_cases = [
            '"CHOAM controls the spice," Paul said.',
            'By giving me Arrakis, His Majesty is forced to give us a CHOAM directorship...a subtle gain.',
            '"This is a test." He said.',
            'By giving me Arrakis, His Majesty is forced to give us a CHOAM directorship...a subtle gain." "CHOAM controls the spice," Paul said.',
        ]
        
        for text in test_cases:
            print(f"\nTesting text: [{text}]")
            try:
                result = phonemize(text, 'a', debug=True)
                print(f"Success! Result: {result}")
            except Exception as e:
                print(f"Failed with error: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                raise
                
    def test_phonemize_ellipsis(self):
        """Test various forms of ellipsis handling"""
        test_cases = [
            # Single ellipsis
            'The spice must flow...',
            # Multiple ellipsis
            'We\'re the ones who tamed Arrakis...except for the few mongrel Fremen hiding in the skirts of the desert...and some tame smugglers bound to the planet almost as tightly as the native labor pool.',
            # Mixed dots and ellipsis
            'The spice... it controls... everything.',
            # Ellipsis with quotes
            '"The spice..." he whispered.',
        ]
        
        for text in test_cases:
            print(f"\nTesting ellipsis: [{text}]")
            try:
                result = phonemize(text, 'a', debug=True)
                print(f"Success! Result: {result}")
            except Exception as e:
                print(f"Failed with error: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                raise

    def test_phonemize_contractions(self):
        """Test handling of contractions and apostrophes"""
        test_cases = [
            # Basic contractions
            "I can't do it.",
            "You mustn't let yourself hope too much.",
            # Multiple contractions
            "I won't say it's impossible, but you shouldn't try.",
            # Mixed with quotes
            '"You mustn\'t let yourself hope too much."',
        ]
        
        for text in test_cases:
            print(f"\nTesting contractions: [{text}]")
            try:
                result = phonemize(text, 'a', debug=True)
                print(f"Success! Result: {result}")
            except Exception as e:
                print(f"Failed with error: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                raise

if __name__ == '__main__':
    unittest.main()
