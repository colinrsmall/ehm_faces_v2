# /Users/colinsmall/github/ehm/finetuning/coaches/upscale_crop_coaches.py

import os
import argparse
import subprocess
import shutil
from PIL import Image, ImageOps, ImageFilter
from PIL.Image import Resampling
from tqdm import tqdm
import sys

# --- Configuration ---
TARGET_FINAL_SIZE = 512
REALESRGAN_MODEL = "Real-ESRGAN_x4plus" # Using the x4 model
REALESRGAN_SCALE = 4
# If 'realesrgan-ncnn-vulkan' is not in PATH, specify the full path here
# Example: REALESRGAN_EXECUTABLE = "/path/to/realesrgan-ncnn-vulkan"
REALESRGAN_EXECUTABLE = "realesrgan-ncnn-vulkan" # Assumes it's in PATH

def run_realesrgan(input_dir, output_dir, executable, model, face_enhance):
    """Runs Real-ESRGAN on all images in the input directory."""
    print(f"Running Real-ESRGAN (Model: {model}, Face Enhance: {face_enhance})...")
    print(f"Input: {input_dir}, Output: {output_dir}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images to upscale.")

    command = [
        executable,
        "-i", input_dir,
        "-o", output_dir,
        "-n", model,
        "-s", str(REALESRGAN_SCALE),
        "-f", "png" # Specify output format
    ]
    if face_enhance:
        command.append("--face-enhance")

    print(f"Executing command: {' '.join(command)}")

    try:
        # Use shell=False for security unless absolutely necessary
        # Capture output for better error reporting
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Real-ESRGAN Output:\n" + result.stdout)
        if result.stderr:
             print("Real-ESRGAN Errors/Warnings:\n" + result.stderr, file=sys.stderr)
        print("Real-ESRGAN execution completed.")
        return True
    except FileNotFoundError:
        print(f"Error: Real-ESRGAN executable not found at '{executable}'.")
        print("Please ensure it's installed and in your PATH, or provide the full path via --realesrgan_path.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during Real-ESRGAN execution (Return Code: {e.returncode}):")
        print("Command:", ' '.join(e.cmd))
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred running Real-ESRGAN: {e}")
        return False

def resize_and_crop(image_path, target_size):
    """Resizes the smallest dimension to target_size using Lanczos, then center crops."""
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        # Determine smallest dimension and calculate resize ratio
        if width < height:
            scale_ratio = target_size / width
            new_width = target_size
            new_height = int(height * scale_ratio)
        else:
            scale_ratio = target_size / height
            new_height = target_size
            new_width = int(width * scale_ratio)

        # Resize using Lanczos
        resized_img = img.resize((new_width, new_height), Resampling.LANCZOS)

        # Calculate center crop box
        left = (new_width - target_size) / 2
        top = (new_height - target_size) / 2
        right = (new_width + target_size) / 2
        bottom = (new_height + target_size) / 2

        # Crop
        cropped_img = resized_img.crop((left, top, right, bottom))

        # Final check if size is exactly target_size x target_size
        if cropped_img.size != (target_size, target_size):
             print(f"Warning: Final cropped image size is {cropped_img.size} for {os.path.basename(image_path)}, expected ({target_size},{target_size}). Adjusting...")
             cropped_img = cropped_img.resize((target_size, target_size), Resampling.LANCZOS)


        return cropped_img

    except Exception as e:
        print(f"Error resizing/cropping {os.path.basename(image_path)}: {e}")
        return None

def process_images(input_dir, output_dir, target_size, realesrgan_executable, face_enhance):
    """Main processing function."""

    temp_upscale_dir = os.path.join(os.path.dirname(output_dir), "temp_realesrgan_output")

    # --- Stage 1: Upscale using Real-ESRGAN ---
    if not run_realesrgan(input_dir, temp_upscale_dir, realesrgan_executable, REALESRGAN_MODEL, face_enhance):
        print("Aborting due to Real-ESRGAN error.")
        # Clean up temp dir if it exists but Real-ESRGAN failed partially
        if os.path.exists(temp_upscale_dir):
             shutil.rmtree(temp_upscale_dir)
             print(f"Removed temporary directory: {temp_upscale_dir}")
        return

    # --- Stage 2: Resize and Crop ---
    print(f"\nResizing and cropping images from {temp_upscale_dir} to {target_size}x{target_size}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    upscaled_files = [f for f in os.listdir(temp_upscale_dir) if os.path.isfile(os.path.join(temp_upscale_dir, f)) and f.lower().endswith('.png')]
    print(f"Found {len(upscaled_files)} upscaled images to process.")

    processed_count = 0
    error_count = 0

    for filename in tqdm(upscaled_files, desc="Resizing/Cropping"):
        temp_image_path = os.path.join(temp_upscale_dir, filename)
        final_image_path = os.path.join(output_dir, filename)

        final_image = resize_and_crop(temp_image_path, target_size)

        if final_image:
            try:
                final_image.save(final_image_path, "PNG")
                processed_count += 1
            except Exception as e:
                print(f"Error saving final image {filename}: {e}")
                error_count += 1
        else:
            error_count += 1

    # --- Stage 3: Cleanup ---
    print("\nCleaning up temporary directory...")
    try:
        shutil.rmtree(temp_upscale_dir)
        print(f"Removed temporary directory: {temp_upscale_dir}")
    except Exception as e:
        print(f"Error removing temporary directory {temp_upscale_dir}: {e}")

    print("\nProcessing complete.")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Final images saved to: {output_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "step_1_cropped_images") # Use the original cropped images
    default_output = os.path.join(script_dir, "step_2_upscaled_cropped")

    parser = argparse.ArgumentParser(description="Upscale images with Real-ESRGAN, resize smallest dimension, and center crop.")
    parser.add_argument("--input_dir", type=str, default=default_input,
                        help="Directory containing the initial cropped images (e.g., 176x192).")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Directory to save the final processed images (e.g., 512x512).")
    parser.add_argument("--target_size", type=int, default=TARGET_FINAL_SIZE,
                        help="Final square size for the output images.")
    parser.add_argument("--realesrgan_path", type=str, default=REALESRGAN_EXECUTABLE,
                        help="Full path to the 'realesrgan-ncnn-vulkan' executable if not in PATH.")
    parser.add_argument("--noface_enhance", action="store_true",
                        help="Disable the --face-enhance flag for Real-ESRGAN.")

    args = parser.parse_args()

    process_images(
        args.input_dir,
        args.output_dir,
        args.target_size,
        args.realesrgan_path,
        not args.noface_enhance # Pass True to face_enhance unless --noface_enhance is specified
    )
