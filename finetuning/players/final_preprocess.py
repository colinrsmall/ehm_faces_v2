import os
import shutil
from PIL import Image, ImageOps
from tqdm.auto import tqdm

# --- Configuration ---
SOURCE_DIR = "hq_filtered"       # Directory with selected images and .txt files
DEST_DIR = "final_processed_dataset" # Directory for final output
TARGET_SIZE = (256, 256)        # Target canvas size
BACKGROUND_COLOR = (19, 26, 35, 255) # RGBA for #131A23
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'} # Process these image types

def preprocess_images():
    """
    Processes images: resizes to fit TARGET_SIZE, adds green background
    for transparent areas, and copies images and captions to DEST_DIR.
    """

    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Ensured destination directory exists: '{DEST_DIR}'")

    processed_count = 0
    copied_caption_count = 0
    skipped_non_image = 0
    error_count = 0

    filenames = os.listdir(SOURCE_DIR)
    print(f"Processing files in '{SOURCE_DIR}'...")

    for filename in tqdm(filenames, desc="Preprocessing Images"):
        source_path = os.path.join(SOURCE_DIR, filename)
        name_part, extension = os.path.splitext(filename)

        # --- Process Images ---
        if extension.lower() in IMAGE_EXTENSIONS:
            try:
                # Open image and ensure RGBA
                with Image.open(source_path) as img:
                    img = img.convert("RGBA")

                    # Calculate new size maintaining aspect ratio
                    img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS) # Resizes in-place to fit

                    # Create green background canvas
                    background = Image.new('RGBA', TARGET_SIZE, BACKGROUND_COLOR)

                    # Calculate position to paste image: centered horizontally, bottom-aligned vertically
                    paste_x = (TARGET_SIZE[0] - img.width) // 2
                    paste_y = TARGET_SIZE[1] - img.height # Align bottom edge
                    paste_position = (paste_x, paste_y)

                    # Paste the image onto the background using its alpha channel
                    # This ensures only originally transparent areas become green
                    background.paste(img, paste_position, img)

                    # Save the final image to the destination directory
                    dest_image_path = os.path.join(DEST_DIR, filename)
                    background.save(dest_image_path, "PNG") # Save as PNG
                    processed_count += 1

                # --- Copy Corresponding Caption File ---
                caption_filename = f"{name_part}.txt"
                source_caption_path = os.path.join(SOURCE_DIR, caption_filename)
                dest_caption_path = os.path.join(DEST_DIR, caption_filename)

                if os.path.exists(source_caption_path):
                    try:
                        shutil.copy2(source_caption_path, dest_caption_path)
                        copied_caption_count += 1
                    except Exception as copy_err:
                        print(f"Error copying caption file '{source_caption_path}': {copy_err}")
                        error_count += 1
                # else: # Don't necessarily warn if caption is missing, might be expected
                #    print(f"Warning: Caption file not found for image '{filename}': {source_caption_path}")


            except Exception as img_err:
                print(f"Error processing image '{filename}': {img_err}")
                error_count += 1
        elif extension.lower() != '.txt': # Skip .txt files silently, warn about others
             skipped_non_image += 1
            #  print(f"Skipping non-image file: {filename}")


    print("\nPreprocessing complete.")
    print(f"  Images processed and saved: {processed_count}")
    print(f"  Caption files copied: {copied_caption_count}")
    print(f"  Files skipped (non-image/non-txt): {skipped_non_image}")
    print(f"  Errors encountered: {error_count}")
    print(f"Processed files are located in: '{DEST_DIR}'")


if __name__ == "__main__":
    preprocess_images()
