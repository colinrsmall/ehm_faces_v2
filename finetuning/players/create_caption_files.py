import os
import csv
from tqdm.auto import tqdm

# --- Configuration ---
CAPTIONS_CSV = "captions.csv"
IMAGE_DIR = "final_processed_dataset" # Directory containing the selected images
# Text files will be saved in the IMAGE_DIR alongside the images

def create_caption_text_files():
    """
    Reads captions from a CSV and creates individual .txt files for images
    found in a specified directory.
    """

    if not os.path.exists(CAPTIONS_CSV):
        print(f"Error: Captions file '{CAPTIONS_CSV}' not found.")
        return
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory '{IMAGE_DIR}' not found.")
        return

    # 1. Load captions from CSV into a dictionary
    captions_map = {}
    print(f"Loading captions from '{CAPTIONS_CSV}'...")
    try:
        with open(CAPTIONS_CSV, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if "Image File Name" not in reader.fieldnames or "Caption" not in reader.fieldnames:
                 print(f"Error: CSV file must contain 'Image File Name' and 'Caption' columns.")
                 return
            for row in reader:
                image_filename = row.get("Image File Name")
                caption = row.get("Caption")
                if image_filename and caption:
                    captions_map[image_filename] = caption
        print(f"Loaded {len(captions_map)} captions.")
    except FileNotFoundError:
        print(f"Error: Could not open CSV file '{CAPTIONS_CSV}'.") # Should be caught earlier, but belt-and-suspenders
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading CSV: {e}")
        return


    # 2. Iterate through images in the target directory and create .txt files
    print(f"Processing images in '{IMAGE_DIR}' and creating caption files...")
    files_processed = 0
    files_created = 0
    files_skipped_no_caption = 0
    files_skipped_not_in_dir = 0 # Count captions for images not actually present

    # Check which captions correspond to files actually in the directory
    image_files_in_dir = set(os.listdir(IMAGE_DIR))

    for image_filename in tqdm(captions_map.keys(), desc="Creating Caption Files"):
         if image_filename in image_files_in_dir:
            caption = captions_map[image_filename]
            base_filename, _ = os.path.splitext(image_filename)
            txt_filename = f"{base_filename}.txt"
            txt_filepath = os.path.join(IMAGE_DIR, txt_filename)

            try:
                with open(txt_filepath, 'w', encoding='utf-8') as outfile:
                    outfile.write(caption)
                files_created += 1
            except IOError as e:
                print(f"Error writing file '{txt_filepath}': {e}")
            files_processed +=1 # Count as processed even if write fails

         # Although we iterate through caption keys, let's double-check presence
         # This loop structure prioritizes using the known captions efficiently
         # We don't explicitly need a separate count for files_skipped_not_in_dir now.


    # Check for image files in the directory that didn't have a caption in the CSV
    for filename in image_files_in_dir:
        # Simple check if it looks like an image file based on extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and filename not in captions_map:
             print(f"Warning: Image file '{filename}' found in '{IMAGE_DIR}' but has no corresponding caption in '{CAPTIONS_CSV}'.")
             files_skipped_no_caption += 1


    print("\nProcessing complete.")
    print(f"  Text files created: {files_created}")
    # print(f"  Images processed (had caption): {files_processed}") # files_created is more intuitive
    print(f"  Images skipped (no caption found in CSV): {files_skipped_no_caption}")

if __name__ == "__main__":
    create_caption_text_files()
