# finetuning/coaches/resize_coaches.py

import os
import argparse
from PIL import Image
from tqdm import tqdm

def calculate_crop_box(img_width, img_height, target_width, target_height):
    """Calculates the box coordinates for center cropping."""
    width_diff = img_width - target_width
    height_diff = img_height - target_height

    if width_diff < 0 or height_diff < 0:
        return None # Image is smaller than target size

    left = width_diff // 2
    top = height_diff // 2
    right = left + target_width
    bottom = top + target_height

    return (left, top, right, bottom)

def process_images(input_dir, output_dir, target_width, target_height):
    """
    Resizes images in the input directory by center cropping to the target size
    and saves them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    print(f"Found {len(image_files)} images in {input_dir}.")
    print(f"Target crop size: {target_width}x{target_height}")

    skipped_count = 0
    processed_count = 0

    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename) # Keep original filename

        try:
            with Image.open(input_path) as img:
                img_width, img_height = img.size

                # Calculate crop box
                crop_box = calculate_crop_box(img_width, img_height, target_width, target_height)

                if crop_box is None:
                    # print(f"Skipping {filename}: Image size ({img_width}x{img_height}) is smaller than target ({target_width}x{target_height}).")
                    skipped_count += 1
                    continue

                # Crop the image
                cropped_img = img.crop(crop_box)

                # Ensure the cropped image is exactly the target size (should be, but good check)
                if cropped_img.size != (target_width, target_height):
                     print(f"Warning: Cropped image {filename} has unexpected size {cropped_img.size}. Expected {target_width}x{target_height}.")
                     # Optionally skip saving or try to resize explicitly
                     # continue

                # Save the cropped image
                cropped_img.save(output_path)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1

    print(f"\nProcessing complete.")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (too small or error): {skipped_count}")

if __name__ == "__main__":
    # Defaults relative to the script's location might be brittle.
    # It's often better to use absolute paths or paths relative to a project root.
    # Assuming script is in finetuning/coaches/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "raw_images")
    default_output = os.path.join(script_dir, "resized_images")

    parser = argparse.ArgumentParser(description="Resize images by center cropping.")
    parser.add_argument("--input_dir", type=str, default=default_input,
                        help="Directory containing raw images.")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Directory to save resized images.")
    parser.add_argument("--width", type=int, default=176,
                        help="Target width for cropping.")
    parser.add_argument("--height", type=int, default=192,
                        help="Target height for cropping.")

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.width, args.height)
