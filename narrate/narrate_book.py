import builtins
import re
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from narrate.kokoro import build_model, generate, phonemize
from narrate.voices import mix_voices

DEBUG = True

# Store the original print function
real_print = builtins.print

# Override the built-in print with our debug-aware version
def print(*args, **kwargs):
    if DEBUG:
        real_print(*args, **kwargs)

# always print
def print2(*args, **kwargs):
    real_print(*args, **kwargs)

builtins.print = print

# Global flag for interrupt handling
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\nInterrupt received, cleaning up...")

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def create_output_dirs():
    """Create output directories for wav and mp3 files."""
    Path('book_audio/wav').mkdir(parents=True, exist_ok=True)
    Path('book_audio/mp3').mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """Clean and normalize text for phonemization."""
    try:
        # Handle potential encoding issues
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')

        # Handle curly quotes and apostrophes first
        text = text.replace('"', '"').replace('"', '"')  # Convert curly quotes to straight quotes
        text = text.replace(''', "'").replace(''', "'")  # Convert curly apostrophes to straight ones

        # Pronunciation mappings for better TTS output
        DUNE_PRONUNCIATIONS = {
            'Bene Gesserit': 'Ben-ee Jes-er-it',
            "Muad’Dib": 'Moo-ad-dib',
            'CHOAM': 'Chom',
            'Harkonnens': 'Har-kon-en',
            'fief': 'feef',
            'Gom Jabbar': 'Gohm Ja-bar',
            'Kwisatz Haderach': 'Kwee-sats Ha-der-ak',
            'Tleilaxu': 'Tlay-lax-oo',
            'Atreides': 'A-tray-i-dees',
            'Landsraad': 'Lands-rod',
            'Arrakeen': 'A-ra-keen',
            'Shai-hulud': 'Shy Hoo-lud',
            'Jabbar': 'Ja-bar',
            'Giedi Prime': 'Gee-dee Prime',
            'Faufreluches': 'Faw-freh-lu-chez',
            'Sietch': 'See-ich',
            'Sukh': 'Suh-kh',
            'Grumman': 'Grum-mun',
            'Padishah': 'Pah-dish-ah',
            'Stilltent': 'Still-tent',
            'Bindu': 'Byndoo',
            'Fremen': 'Frehmen',
            'Ghanima': 'Ga-nee-ma',
            'Adab': 'ah-dahb',
            'Irulan': 'Ee-ru-lan',
            'Leto': 'Lee-toh',
            'Stilgar': 'Stil-gar',
            'Rabban': 'Rab-ban',
            'Fenring': 'Fen-ring',
            'Sardaukar': 'Sar-dow-kar',
            'Piter': 'Pee-ter',
            'Chaouhada': 'cha-oo-ha-da',
            'Shishakali': 'Shi-shak-ly',
            'Syach': 'Sa-itch',
            'Canley': 'Can-lee',
            'Verota': 'Ve-roh-ta',
            'Uzeel': 'Oo-zeel',
            'Sayadina': 'Say-a-deen-a',
            'Alia': 'A-lee-ah',
            'Arrakis': 'A-ra-kis',
            'Chani': 'Chan-ee',
            'Faradn': 'Far-a-din',
            'Fedaykin': 'Fe-die-kin',
            'Harq-al-ada': 'Hark-al-a-da',
            'Melange': 'Meh-lange',
            'Qanat': 'Kah-nat',
            'Sabiha': 'Sa-bee-ha',
            'Tuek': 'Tuek',
        }

        # Apply Dune-specific pronunciation mappings
        text_lower = text.lower()
        for original, replacement in DUNE_PRONUNCIATIONS.items():
            # Find all occurrences with word boundaries, preserving original case
            matches = re.finditer(r'\b' + re.escape(original) + r'\b', text_lower)
            for match in reversed(list(matches)):  # Process from end to start to maintain string indices
                start, end = match.span()
                text = text[:start] + replacement + text[end:]

        # Normalize line endings first
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Handle ellipses and other punctuation
        text = text.replace('…', '...')

        # Clean up whitespace while preserving paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces/tabs within lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Convert multiple blank lines to double newlines

        # Final cleanup - convert all newlines to spaces
        text = re.sub(r'\s*\n\s*', ' ', text)
        text = text.strip()

        return text

    except Exception as e:
        print(f"Error in clean_text: {str(e)}")
        return ""

def split_into_segments(text, model, voicepack):
    """Split text into segments that will result in phonemes that can be processed.

    Returns a list of tuples (segment, wav, ps) where wav and ps may be None if generation failed.
    """

    def recursive_split(segment, depth=0):
        """Recursively split segment until all parts are under the limit."""
        indent = "  " * depth
        print(f"\n{indent}Recursive split depth {depth} ({len(segment)} chars):")
        print(f"{indent}{segment[:100]}...")

        # Try to phonemize the whole segment first
        print(f"\n{indent}Trying to phonemize segment ({len(segment)} chars):")
        print(f"{indent}{segment[:100]}...")

        wav, ps = generate_with_cleanup(model, segment, voicepack, lang='a')
        if wav is not None:
            # If we got audio, this segment was okay
            print(f"{indent}Segment OK: generated {len(wav)} samples")
            return [(segment, wav, ps)]

        print(f"{indent}Segment too big or failed, trying to split...")

        # First try splitting on sentence boundaries and combining as many as possible
        sentences = split_on_sentences(segment)
        if len(sentences) > 1:
            print(f"Found {len(sentences)} sentences, trying optimal combinations")
            result = []
            current_segment = []
            current_text = ""

            for s in sentences:
                test_text = (current_text + " " + s).strip() if current_text else s
                wav, ps = generate_with_cleanup(model, test_text, voicepack, lang='a')

                if wav is not None:
                    # This combination works, keep building
                    current_text = test_text
                    current_segment = [(test_text, wav, ps)]
                # This combination is too big, save current and start new
                elif current_text:
                    result.extend(current_segment)
                    current_text = s
                    wav, ps = generate_with_cleanup(model, s, voicepack, lang='a')
                    if wav is not None:
                        current_segment = [(s, wav, ps)]
                    else:
                        # Single sentence is too big, need to split it
                        result.extend(recursive_split(s, depth + 1))
                        current_text = ""
                        current_segment = []
                else:
                    # First sentence is too big
                    result.extend(recursive_split(s, depth + 1))

            # Don't forget to add the last segment
            if current_text:
                result.extend(current_segment)

            return result

        # If no sentence breaks or still too big, split on words
        print(f"{indent}No sentence breaks found, splitting on words")
        words = segment.split()
        if len(words) <= 1:
            print(f"{indent}Warning: Cannot split further, returning potentially oversized segment")
            return [(segment, None, None)]

        # Try to find optimal word split point
        for i in range(len(words) - 1, 0, -1):
            left = ' '.join(words[:i])
            wav, ps = generate_with_cleanup(model, left, voicepack, lang='a')
            if wav is not None:
                # Found a valid left split, recursively handle the right
                right = ' '.join(words[i:])
                return [(left, wav, ps)] + recursive_split(right, depth + 1)

        # If we couldn't find a valid split point, split in half
        mid = len(words) // 2
        left = ' '.join(words[:mid])
        right = ' '.join(words[mid:])

        result = []
        result.extend(recursive_split(left, depth + 1))
        result.extend(recursive_split(right, depth + 1))
        return result

    def split_on_sentences(text):
        """Split text on sentence boundaries."""
        # First normalize to single line
        text = re.sub(r'\s*\n\s*', ' ', text).strip()

        # Simple split on sentence endings
        splits = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in splits if s.strip()]

    print(f"\nSplitting text into segments ({len(text)} chars)")
    return recursive_split(text.strip())

def generate_with_cleanup(model, text, voicepack, lang):
    """Generate audio with phoneme cleanup."""
    try:
        # Try to phonemize with original text
        ps = phonemize(text, lang, debug=True)
        if ps:
            # Try generation with phonemes
            result = generate(model, text, voicepack, lang=lang, ps=ps)
            if result is not None:
                wav, ps = result  # Use the processed phonemes from generate
                return wav, ps

        print2("Failed to generate audio with cleaned text")
        return None, None

    except Exception as e:
        print(f"Error in generate_with_cleanup: {e}")
        return None, None

def process_chunk(chunk_num, chunk, model, voicepack):
    """Process a single text chunk."""
    global interrupted
    if interrupted:
        return False

    print(f"\nProcessing chunk {chunk_num}...")
    print(f"Original chunk ({len(chunk)} chars):")
    print(chunk)

    try:
        # Split into segments and get their audio
        segment_pairs = split_into_segments(chunk, model, voicepack)

        # Process each segment
        audio_segments = []
        for segment, wav, ps in segment_pairs:
            if interrupted:
                return False
            # Only generate audio if we don't already have it
            if wav is None:
                wav, ps = generate_with_cleanup(model, segment, voicepack, lang='a')
                if wav is None:
                    raise RuntimeError(f"Failed to generate audio for segment: {segment}")
            audio_segments.append(wav)
            print(f"Processed with phenome length {len(ps)}")
            print(segment)

        # Combine segments
        if not audio_segments:
            raise RuntimeError("No audio segments generated")

        wav = np.concatenate(audio_segments)

        # Save wav file
        wav_path = f'book_audio/wav/chunk_{chunk_num:04d}.wav'
        sf.write(wav_path, wav, 24000)

        return True

    except Exception as e:
        print(f"Error processing chunk {chunk_num}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise

def main():
    global interrupted
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model('kokoro-v0_19.pth', device)

    print("\nCreating mixed voice...")
    # Create mixed voice (40% Bella, 60% George)
    voice_weights = {
        'af_bella': 0.4,
        'bm_george': 0.6
    }
    voicepack = mix_voices(voice_weights, device=device)

    print("\nReading book file...")
    try:
        with open('dune.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open('dune.txt', 'r', encoding='latin1') as f:
            text = f.read()

    print(f"Read {len(text)} characters")

    print("\nCleaning text...")
    text = clean_text(text)
    print(f"Text after cleaning: {len(text)} characters")
    print("Sample of cleaned text:")
    print(text[:500] + "...")

    # Create output directories
    create_output_dirs()

    print("\nSplitting into chunks...")
    # Split into chunks of roughly equal size, preserving paragraph breaks
    chunk_size = 1000  # Target size for each chunk (reduced from 2000)
    chunks = []
    current_chunk = []
    current_size = 0

    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue

        # If this paragraph alone is bigger than chunk_size,
        # split it into sentences first
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_sentence = []
            current_sentence_size = 0

            for sentence in sentences:
                if current_sentence_size + len(sentence) > chunk_size and current_sentence:
                    chunks.append(' '.join(current_sentence))
                    current_sentence = []
                    current_sentence_size = 0
                current_sentence.append(sentence)
                current_sentence_size += len(sentence)

            if current_sentence:
                chunks.append(' '.join(current_sentence))
            continue

        # If adding this paragraph would exceed chunk size and we have content,
        # start a new chunk
        if current_size + len(para) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += len(para)

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    print(f"Split into {len(chunks)} chunks")
    print("\nSample chunks:")
    for i in range(min(3, len(chunks))):
        print(f"\nChunk {i+1} ({len(chunks[i])} chars):")
        print(chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i])

    # Process chunks in parallel
    num_workers = min(16, len(chunks))
    print(f"\nGenerating audio with {num_workers} workers ({len(chunks)} chunks)...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            if interrupted:
                break
            futures.append(executor.submit(process_chunk, i, chunk, model, voicepack))

        # Wait for all chunks to complete
        for future in tqdm(futures):
            try:
                if interrupted:
                    executor.shutdown(wait=False)
                    break
                future.result()
            except Exception as e:
                print(f"Error: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                executor.shutdown(wait=False)
                raise

    if interrupted:
        print("\nGeneration interrupted by user")
        return

    print('\nConverting WAVs to MP3...')
    # Convert all WAVs to MP3s in a single command
    subprocess.run([
        'ffmpeg',
        '-i', 'concat:' + '|'.join(f'book_audio/wav/chunk_{i:04d}.wav' for i in range(len(chunks))),
        '-c:a', 'libmp3lame',
        '-q:a', '2',
        'book_audio/dune.mp3'
    ], check=True)
    
    print('Done!')

if __name__ == '__main__':
    main()
