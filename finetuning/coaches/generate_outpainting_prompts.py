# /Users/colinsmall/github/ehm/finetuning/coaches/generate_outpainting_prompts.py

import os
import argparse
import base64
import json
import csv
from PIL import Image
from tqdm import tqdm
import sys
import time

# Add the parent directory (finetuning) to the Python path
# to import chat_operations
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    # Assumes chat_operations.py is in the parent directory (finetuning)
    from chat_operations import sessionless_vision_call_function
except ImportError:
    print("Error: Could not import 'sessionless_vision_call_function' from chat_operations.py.")
    print(f"Ensure chat_operations.py is in the directory: {parent_dir}")
    sys.exit(1)

# --- Configuration ---
# Choose a capable vision model like gpt-4o or gpt-4-vision-preview
DEFAULT_VISION_MODEL = "gpt-4o" #"gpt-4-vision-preview"

SYSTEM_MESSAGE = """
You are an expert image analyst. Your task is to describe the central subject of the provided image \
clearly and concisely. This description will be used as a prompt for an AI image outpainting \
model (like Stable Diffusion Inpainting) to extend the image naturally. Focus on the existing background. \
Do not mention the cropping or that it's a partial view. \
Output the description as a JSON object following the provided schema.
"""

TEXT_PROMPT_FOR_LLM = """
Describe the central subject of this image for an outpainting prompt. Keep it concise. The subject is an ice hockey coach.
"""

# Schema for the expected JSON output from the LLM
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "A concise description of the central subject for use in an outpainting prompt."
        }
    },
    "required": ["prompt"],
    "additionalProperties": False
}

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with Image.open(image_path) as img:
            # Determine format and media type
            img_format = img.format.lower() if img.format else 'png' # Default to png if format is None
            if img_format == 'jpeg':
                media_type = 'image/jpeg'
            elif img_format == 'png':
                 media_type = 'image/png'
            # Add more types if needed (gif, bmp, etc.)
            else:
                 print(f"Warning: Unsupported image format '{img_format}' for {os.path.basename(image_path)}. Defaulting to PNG.")
                 media_type = 'image/png'
                 # Convert to RGB before saving to buffer if needed, PNG supports transparency
                 img = img.convert('RGB') if img.mode == 'RGBA' and media_type != 'image/png' else img


            from io import BytesIO
            buffered = BytesIO()
            # Save image to buffer in its original/detected format
            save_format = 'JPEG' if media_type == 'image/jpeg' else 'PNG'
            img.save(buffered, format=save_format)
            img_byte = buffered.getvalue()
            base64_string = base64.b64encode(img_byte).decode('utf-8')
            return base64_string, media_type
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None, None

def generate_prompts(input_dir, output_csv, model):
    """
    Generates prompts for images in input_dir using a vision LLM and saves them to output_csv.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    print(f"Found {len(image_files)} images in {input_dir}.")
    print(f"Using model: {model}")
    print(f"Writing prompts to: {output_csv}")

    processed_count = 0
    error_count = 0
    rate_limit_pauses = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in tqdm(image_files[:5], desc="Generating prompts"):
            image_path = os.path.join(input_dir, filename)
            base64_image, media_type = encode_image_to_base64(image_path)

            if not base64_image:
                error_count += 1
                continue

            # --- LLM Call ---
            attempts = 0
            max_attempts = 3
            wait_time = 5 # seconds
            success = False
            while attempts < max_attempts:
                json_reply_str = sessionless_vision_call_function(
                    system_message=SYSTEM_MESSAGE,
                    text_message=TEXT_PROMPT_FOR_LLM,
                    base64_image=base64_image,
                    image_media_type=media_type,
                    schema=OUTPUT_SCHEMA,
                    model=model
                )

                if json_reply_str:
                    try:
                        reply_data = json.loads(json_reply_str)
                        prompt_text = reply_data.get("prompt", "")
                        if prompt_text:
                            writer.writerow({'file_name': filename, 'prompt': prompt_text})
                            processed_count += 1
                            success = True
                            break # Success, exit retry loop
                        else:
                            print(f"Warning: LLM response for {filename} lacked 'prompt' key or value was empty. Response: {json_reply_str}")
                            # Decide if this counts as an error or just needs retry
                            attempts += 1 # Count as attempt, maybe retry
                            time.sleep(wait_time) # Wait before retrying

                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON response for {filename}. Response: {json_reply_str}")
                        # Decide if this counts as an error or just needs retry
                        attempts += 1 # Count as attempt, maybe retry
                        time.sleep(wait_time) # Wait before retrying
                    except Exception as e:
                        print(f"Error processing LLM response for {filename}: {e}. Response: {json_reply_str}")
                        # Decide if this counts as an error or just needs retry
                        attempts += 1 # Count as attempt, maybe retry
                        time.sleep(wait_time) # Wait before retrying
                else:
                    # sessionless_vision_call_function returned None, likely API error (e.g., rate limit)
                    print(f"Warning: API call failed for {filename} (attempt {attempts+1}/{max_attempts}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    attempts += 1
                    rate_limit_pauses +=1
                    wait_time *= 2 # Exponential backoff

            if not success:
                 print(f"Error: Failed to get a valid prompt for {filename} after {max_attempts} attempts.")
                 error_count += 1
            # --- End LLM Call ---

            # Optional: Short delay to avoid hitting rate limits aggressively
            # time.sleep(0.5)

    print(f"\nPrompt generation complete.")
    print(f"  Successfully generated prompts for: {processed_count} images.")
    print(f"  Errors/Skipped: {error_count}")
    if rate_limit_pauses > 0:
        print(f"  Paused {rate_limit_pauses} times due to potential rate limits.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Default input is the output of the previous script
    default_input = os.path.join(script_dir, "resized_images") # <<< Use the output of resize_coaches.py
    default_output_csv = os.path.join(script_dir, "outpainting_prompts.csv")

    parser = argparse.ArgumentParser(description="Generate outpainting prompts for images using a vision LLM.")
    parser.add_argument("--input_dir", type=str, default=default_input,
                        help="Directory containing the resized/cropped images (output of resize_coaches.py).")
    parser.add_argument("--output_csv", type=str, default=default_output_csv,
                        help="Path to save the output CSV file with prompts.")
    parser.add_argument("--model", type=str, default=DEFAULT_VISION_MODEL,
                        help=f"The OpenAI vision model to use (e.g., gpt-4o, gpt-4-vision-preview). Default: {DEFAULT_VISION_MODEL}")

    args = parser.parse_args()

    # Make sure you have OPENAI_API_KEY set in your environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
    else:
        generate_prompts(args.input_dir, args.output_csv, args.model)
