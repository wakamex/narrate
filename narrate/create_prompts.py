import os
import time

import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
    """Upload the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Wait for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
        print("...all files ready")
    print()

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

files = [
    upload_to_gemini("recraft.txt", mime_type="text/plain"),
    upload_to_gemini("dune_first_half.mp3", mime_type="audio/mpeg"),
]

# Some files have a processing delay. Wait for them to be ready.
wait_for_files_active(files)

PROMPT = """
I'm creating a video of stitched together AI generated images to go alongside my narration of Dune. Review my audio and provide timestamps and evocative prompts that I can feed to Recraft v3 to generate these images.
I'm attaching documentation for Recraft in recraft.txt, and the full audio in dune_first_half.mp3. provide only the timestamp and prompt. continue after this prompt:
(25:47 - 26:03) Prompt: A close up on a face that is wrinkled and hard, with a set of eyes that are both piercing and sharp. Style: digital_engraving, artistic_level: 3
make your new prompt begin at 26:03 and continue from there.
"""

chat_session = model.start_chat(files=files)

response = chat_session.send_message(PROMPT)

print(response.text)
