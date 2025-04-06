# /Users/colinsmall/github/ehm/finetuning/coaches/generate_simple_captions.py

import os
import argparse
from tqdm import tqdm

def generate_caption(filename):
    """
    Generates a caption based on the filename format 'FirstName_LastName_....ext'.
    Returns the caption string or None if the filename format is invalid.
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')

    if len(parts) >= 2:
        first_name = parts[0]
        last_name = parts[1]
        # Simple check if names seem plausible (basic - adjust if needed)
        if first_name.isalpha() and last_name.isalpha():
            return f"A high-quality photo of a hockey coach, GM, or staff member named {first_name} {last_name}"
        else:
            print(f"Warning: Could not reliably extract names from '{filename}'. Skipping.")
            return None
    else:
        print(f"Warning: Filename '{filename}' does not match expected 'FirstName_LastName_...' format. Skipping.")
        return None

def process_directory(input_dir):
    """
    Processes all images in the input directory to generate caption files.
    """
    print(f"Processing images in directory: {input_dir}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

    if not image_files:
        print("No image files found in the directory.")
        return

    print(f"Found {len(image_files)} image files. Generating captions...")

    generated_count = 0
    skipped_count = 0

    for filename in tqdm(image_files, desc="Generating Captions"):
        caption = generate_caption(filename)

        if caption:
            base_name = os.path.splitext(filename)[0]
            caption_filename = base_name + ".txt"
            caption_filepath = os.path.join(input_dir, caption_filename)

            try:
                with open(caption_filepath, 'w', encoding='utf-8') as f:
                    f.write(caption)
                generated_count += 1
            except IOError as e:
                print(f"\nError writing caption file {caption_filename}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1 # Filename format issue

    print("\nCaption generation complete.")
    print(f"  Successfully generated: {generated_count}")
    print(f"  Skipped (format/error): {skipped_count}")
    print(f"  Caption files saved in: {input_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Default to the output of the previous step
    default_input = os.path.join(script_dir, "step_2_upscaled_cropped")

    parser = argparse.ArgumentParser(description="Generate simple caption .txt files for images based on 'FirstName_LastName_...' filename format.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=default_input,
        help="Directory containing the image files (e.g., step_2_upscaled_cropped)."
    )

    args = parser.parse_args()
    process_directory(args.input_dir)
