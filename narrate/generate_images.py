import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv


class RecraftImageGenerator:
    def __init__(self, api_token: str):
        """Initialize the Recraft image generator.
        
        Args:
            api_token: Your Recraft API token from https://recraft.ai/
        """
        if not api_token:
            raise ValueError("API token is required. Get one from https://recraft.ai/")
            
        self.api_token = api_token.strip()
        self.base_url = "https://external.api.recraft.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

    def create_style(self, image_paths: List[str], base_style: str = "digital_illustration") -> str:
        """Create a custom style from reference images.
        
        Args:
            image_paths: List of paths to PNG images (max 5)
            base_style: Base style to build upon
            
        Returns:
            str: Style ID to use in image generation
        """
        if len(image_paths) > 5:
            raise ValueError("Maximum 5 reference images allowed")
            
        # Prepare multipart form data
        files = {}
        for i, path in enumerate(image_paths, 1):
            files[f'file{i}'] = open(path, 'rb')
            
        try:
            response = requests.post(
                f"{self.base_url}/styles",
                headers=self.headers,
                data={'style': base_style},
                files=files
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to create style: {response.text}")
                
            result = response.json()
            return result['id']
            
        finally:
            # Clean up file handles
            for f in files.values():
                f.close()

    def generate_image(self, prompt: str, style: str = None, style_id: str = None, artistic_level: int = 2, size: Tuple[int, int] = (1024, 1024)) -> bytes:
        """Generate an image using Recraft's API.
        
        Args:
            prompt: The image description
            style: Style to apply (e.g. 'realistic_image', 'digital_illustration')
            style_id: Custom style ID from create_style()
            artistic_level: Level of artistic detail (0-5)
            size: Tuple of (width, height)
            
        Returns:
            bytes: The generated image data
        """
        if style and style_id:
            raise ValueError("Cannot specify both style and style_id")
            
        data = {
            "prompt": prompt,
            "artistic_level": artistic_level,
            "size": f"{size[0]}x{size[1]}",
            "model": "recraftv3",
            "response_format": "url"
        }
        
        if style_id:
            data["style_id"] = style_id
        elif style:
            data["style"] = style
            
        try:
            response = requests.post(
                f"{self.base_url}/images/generations",
                headers={**self.headers, "Content-Type": "application/json"},
                json=data
            )
            
            if response.status_code != 200:
                raise Exception(f"API error ({response.status_code}): {response.text}")
            
            # Get image URL from response
            result = response.json()
            if not result or 'data' not in result or not result['data']:
                raise Exception(f"Invalid response format: {response.text}")
                
            # Extract URL from response
            image_url = result['data'][0]['url']
            if not image_url:
                raise Exception("No image URL in response")
                
            # Download the actual image
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                raise Exception("Failed to download generated image")
                
            return img_response.content
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")

def parse_prompt(prompt_line: str) -> Dict:
    """Parse a single prompt line into its components.
    
    Example lines:
    (00:00 - 00:16) Prompt: A vast desert landscape... Style: realistic_image, artistic_level: 2, size: 1707x1024
    (53:37 - 53:45) Prompt: A close up view... style: pop_art, artistic_level: 3
    (27:21 - 27:33) Prompt: A medium shot of the arid desert landscape... style: mystic_naturalism, artistic_level: 1
    """
    timestamp_match = re.match(r'\((\d{2}:\d{2}) - (\d{2}:\d{2})\)', prompt_line)
    if not timestamp_match:
        raise ValueError(f"Invalid timestamp format in line: {prompt_line}")
    
    start_time, end_time = timestamp_match.groups()
    
    prompt_match = re.search(r'Prompt:\s*(.*?)(?=(?:style|Style):|$)', prompt_line, re.IGNORECASE)
    if not prompt_match:
        raise ValueError(f"Could not find prompt text in line: {prompt_line}")
    
    prompt_text = prompt_match.group(1).strip()
    
    style_match = re.search(r'(?:style|Style):\s*([^,\s]+)', prompt_line, re.IGNORECASE)
    style_name = style_match.group(1).strip() if style_match else "realistic_image"
    
    artistic_level_match = re.search(r'artistic_level:\s*(\d+)', prompt_line)
    artistic_level = int(artistic_level_match.group(1)) if artistic_level_match else 2
    
    size_match = re.search(r'size:\s*(\d+)x(\d+)', prompt_line)
    size = (
        int(size_match.group(1)), 
        int(size_match.group(2))
    ) if size_match else (1024, 1024)
    
    return {
        'start_time': start_time,
        'end_time': end_time,
        'prompt': prompt_text,
        'style': style_name,
        'artistic_level': artistic_level,
        'size': size
    }

def load_prompts(prompt_file: str) -> List[Dict]:
    """Load and parse all prompts from the prompts file."""
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('('):
                try:
                    prompt_data = parse_prompt(line)
                    prompts.append(prompt_data)
                except ValueError as e:
                    print(f"Warning: Skipping invalid prompt line: {e}")
    return prompts

def generate_book_images(prompt_file: str, api_token: str, output_dir: str = 'book_images', start: int = 0, end: int = None):
    """Generate images for all prompts in the prompt file.
    
    Args:
        prompt_file: Path to the prompts.txt file
        api_token: Recraft API token
        output_dir: Directory to save generated images
        start: Start index (0-based) of prompts to generate
        end: End index (exclusive) of prompts to generate, None for all
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generator = RecraftImageGenerator(api_token)
    
    # Load style ID
    style_file = Path(__file__).parent / "text" / "style_id.txt"
    if not style_file.exists():
        raise ValueError("Style ID not found. Please run create_style.py first")
    style_id = style_file.read_text().strip()
    
    prompts = load_prompts(prompt_file)
    
    # Validate indices
    if start < 0:
        raise ValueError("Start index must be non-negative")
    if end is not None and end <= start:
        raise ValueError("End index must be greater than start index")
    if end is not None and end > len(prompts):
        end = len(prompts)
    if start >= len(prompts):
        raise ValueError(f"Start index {start} is beyond the number of prompts ({len(prompts)})")
    
    # Slice prompts based on start/end
    prompts = prompts[start:end]
    
    for i, prompt_data in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)} for timestamp {prompt_data['start_time']}")
        
        try:
            image_data = generator.generate_image(
                prompt=prompt_data['prompt'],
                style_id=style_id,  # Use our custom style
                artistic_level=prompt_data['artistic_level'],
                size=(1820, 1024)  # Closest to HD aspect ratio (16:9)
            )
            
            filename = f"h00_m{prompt_data['start_time'].replace(':', '_')}.png"
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
                
            print(f"Saved image to {output_path}")
            
        except Exception as e:
            print(f"Error generating image for timestamp {prompt_data['start_time']}: {e}")

if __name__ == '__main__':
    PACKAGE_DIR = Path(__file__).parent

    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Generate images for book narration')
    parser.add_argument('--prompt-file', default=str(PACKAGE_DIR / 'text' / 'prompts.txt'),
                      help='Path to the prompts file')
    parser.add_argument('--output-dir', default='book_images',
                      help='Directory to save generated images')
    parser.add_argument('--start', type=int, default=0,
                      help='Start index of prompts to generate (0-based)')
    parser.add_argument('--end', type=int, default=None,
                      help='End index of prompts to generate (exclusive), defaults to all')
    
    args = parser.parse_args()
    
    api_token = os.getenv('RECRAFT_API_KEY')
    if not api_token:
        raise ValueError("Please set RECRAFT_API_KEY environment variable")
    
    generate_book_images(
        prompt_file=args.prompt_file,
        api_token=api_token,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end
    )
