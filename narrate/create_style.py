import os
from pathlib import Path

from dotenv import load_dotenv

from narrate.generate_images import RecraftImageGenerator


def main():
    # Load API token
    load_dotenv()
    api_token = os.getenv('RECRAFT_API_KEY')
    if not api_token:
        raise ValueError("Please set RECRAFT_API_KEY environment variable")

    # Get reference images
    image_dir = Path.home() / "defi"
    image_paths = [
        str(image_dir / "jodo1.webp"),
        str(image_dir / "jodo2.jpg"),
        str(image_dir / "jodo3.jpeg"),
        str(image_dir / "jodo4.jpg"),
        str(image_dir / "jodo5.jpg")
    ]

    # Create style
    generator = RecraftImageGenerator(api_token)
    style_id = generator.create_style(image_paths)
    print(f"Created style with ID: {style_id}")

    # Save style ID in narrate/text directory
    style_file = Path(__file__).parent / "text" / "style_id.txt"
    style_file.write_text(style_id)
    print(f"Saved style ID to {style_file}")

if __name__ == "__main__":
    main()
