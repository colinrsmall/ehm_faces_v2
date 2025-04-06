# /Users/colinsmall/github/ehm/finetuning/coaches/upscale_crop_coaches.py

import os
import argparse
import subprocess
import shutil
from PIL import Image, ImageOps
from PIL.Image import Resampling
from tqdm import tqdm
import sys

# --- Configuration ---
TARGET_FINAL_SIZE = 512
REALESRGAN_MODEL = "RealESRGAN_x4plus" # Model name for inference_realesrgan.py
REALESRGAN_SCALE = 4 # Corresponds to --outscale in inference_realesrgan.py

def run_inference_script(inference_script_path, input_dir, output_dir, model_name, scale, face_enhance):
    """Runs the provided Real-ESRGAN inference Python script."""
    print(f"Running Real-ESRGAN via script: {inference_script_path}")
    print(f"  Input: {input_dir}, Output: {output_dir}")
    print(f"  Model: {model_name}, Scale: {scale}, Face Enhance: {face_enhance}")

    if not os.path.isfile(inference_script_path):
        print(f"Error: Inference script not found at '{inference_script_path}'")
        return False
    if not inference_script_path.endswith(".py"):
         print(f"Warning: Inference script path '{inference_script_path}' doesn't end with .py")


    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the command
    command = [
        sys.executable, # Use the same python interpreter that's running this script
        inference_script_path,
        "-i", input_dir,
        "-o", output_dir,
        "-n", model_name,
        "-s", str(scale),
        "--ext", "png" # Force PNG output for consistency
        # Note: inference_realesrgan.py uses --suffix, but we'll ignore it
        # as we save to a temp dir and rename later if needed.
    ]

    if face_enhance:
        command.append("--face_enhance")

    # Optionally add --fp32 if needed, but default seems to be fp16 which is usually faster
    # command.append("--fp32")

    print(f"Executing command: {' '.join(command)}")

    try:
        # Use shell=False for security
        # Capture output for better error reporting
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8') # Added encoding
        # Print stdout/stderr only if they contain something
        if result.stdout:
            print("Inference Script Output:\n" + result.stdout)
        if result.stderr:
             print("Inference Script Errors/Warnings:\n" + result.stderr, file=sys.stderr)
        print("Inference script execution completed.")
        return True
    except FileNotFoundError:
        # This might happen if sys.executable is not found, though unlikely
        print(f"Error: Python executable not found at '{sys.executable}'.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during inference script execution (Return Code: {e.returncode}):")
        print("Command:", ' '.join(e.cmd))
        # Decode stdout/stderr if they are bytes (might happen depending on system/python version)
        stdout = e.stdout.decode('utf-8', errors='ignore') if isinstance(e.stdout, bytes) else e.stdout
        stderr = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else e.stderr
        if stdout:
             print("Stdout:\n", stdout)
        if stderr:
             print("Stderr:\n", stderr, file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred running the inference script: {e}")
        return False


def resize_and_crop(image_path, target_size):
    """Resizes the smallest dimension to target_size using Lanczos, then center crops."""
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        if width == 0 or height == 0:
             print(f"Warning: Invalid image dimensions ({width}x{height}) for {os.path.basename(image_path)}. Skipping.")
             return None

        # Determine smallest dimension and calculate resize ratio
        if width < height:
            scale_ratio = target_size / width
            new_width = target_size
            new_height = int(round(height * scale_ratio)) # Use round for potentially better accuracy
        else:
            scale_ratio = target_size / height
            new_height = target_size
            new_width = int(round(width * scale_ratio))

        # Ensure new dimensions are at least 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Resize using Lanczos
        resized_img = img.resize((new_width, new_height), Resampling.LANCZOS)

        # Calculate center crop box (ensure coordinates are integers and within bounds)
        left = max(0, int(round((new_width - target_size) / 2)))
        top = max(0, int(round((new_height - target_size) / 2)))
        # Calculate right/bottom based on left/top and target_size to ensure exact dimensions
        right = min(new_width, left + target_size)
        bottom = min(new_height, top + target_size)

        # Adjust left/top if calculated crop is smaller than target size (edge case)
        if right - left < target_size:
            left = max(0, right - target_size)
        if bottom - top < target_size:
             top = max(0, bottom - target_size)

        # Crop
        cropped_img = resized_img.crop((left, top, right, bottom))

        # Final check and potential resize if cropping didn't yield exact dimensions
        if cropped_img.size != (target_size, target_size):
             print(f"Warning: Cropped image size is {cropped_img.size} for {os.path.basename(image_path)}, expected ({target_size},{target_size}). Resizing forcefully...")
             cropped_img = cropped_img.resize((target_size, target_size), Resampling.LANCZOS)

        return cropped_img

    except FileNotFoundError:
         print(f"Error: File not found during resize/crop: {image_path}")
         return None
    except Exception as e:
        print(f"Error resizing/cropping {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return None

def process_images(input_dir, output_dir, target_size, inference_script_path, model_name, scale, face_enhance):
    """Main processing function using the provided inference script."""

    temp_upscale_dir = os.path.join(os.path.dirname(output_dir), "temp_realesrgan_output")

    # --- Stage 1: Upscale using the provided Python script ---
    if not run_inference_script(inference_script_path, input_dir, temp_upscale_dir, model_name, scale, face_enhance):
        print("Aborting due to inference script error.")
        # Clean up temp dir if it exists but script failed partially
        if os.path.exists(temp_upscale_dir):
             try:
                 shutil.rmtree(temp_upscale_dir)
                 print(f"Removed temporary directory: {temp_upscale_dir}")
             except Exception as e:
                 print(f"Error removing temporary directory {temp_upscale_dir}: {e}")
        return

    # --- Stage 2: Resize and Crop ---
    print(f"\nResizing and cropping images from {temp_upscale_dir} to {target_size}x{target_size}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List files in the temp directory - inference_realesrgan.py might add a suffix
    # We will handle renaming during the copy/save process
    upscaled_files = [f for f in os.listdir(temp_upscale_dir) if os.path.isfile(os.path.join(temp_upscale_dir, f)) and f.lower().endswith('.png')]
    print(f"Found {len(upscaled_files)} upscaled images to process.")

    processed_count = 0
    error_count = 0

    for filename_with_suffix in tqdm(upscaled_files, desc="Resizing/Cropping"):
        temp_image_path = os.path.join(temp_upscale_dir, filename_with_suffix)

        # Try to determine the original filename (assuming suffix is like '_out.png')
        # This part might need adjustment based on the actual suffix used by inference_realesrgan.py
        base, ext = os.path.splitext(filename_with_suffix)
        original_filename = filename_with_suffix # Default
        possible_suffixes = ['_out', '_SwinIR', '_realesrgan'] # Common suffixes
        for suffix in possible_suffixes:
             if base.endswith(suffix):
                 original_filename = base[:-len(suffix)] + ext
                 break

        final_image_path = os.path.join(output_dir, os.path.basename(original_filename)) # Use original base name

        final_image = resize_and_crop(temp_image_path, target_size)

        if final_image:
            try:
                final_image.save(final_image_path, "PNG")
                processed_count += 1
            except Exception as e:
                print(f"\nError saving final image {os.path.basename(final_image_path)}: {e}")
                error_count += 1
        else:
            error_count += 1 # Error occurred in resize_and_crop

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
    # Default path - user MUST override this if it's incorrect
    default_inference_script = os.path.join(script_dir, "..", "image_generation", "inference_realesrgan.py") # GUESSING relative path

    parser = argparse.ArgumentParser(description="Upscale images via inference_realesrgan.py, resize smallest dimension, and center crop.")
    parser.add_argument("--input_dir", type=str, default=default_input,
                        help="Directory containing the initial cropped images.")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Directory to save the final processed images.")
    parser.add_argument("--target_size", type=int, default=TARGET_FINAL_SIZE,
                        help="Final square size for the output images.")
    parser.add_argument("--inference_script_path", type=str, default=default_inference_script,
                        help="Path to the 'inference_realesrgan.py' script.")
    parser.add_argument("--model", type=str, default=REALESRGAN_MODEL,
                        help=f"Real-ESRGAN model name for inference script (e.g., RealESRGAN_x4plus). Default: {REALESRGAN_MODEL}")
    parser.add_argument("--scale", type=int, default=REALESRGAN_SCALE,
                        help="Target upscale ratio for inference script (-s/--outscale). Default: 4")
    parser.add_argument("--face_enhance", action="store_true",
                        help="Enable face enhancement flag (--face_enhance) for inference script.")
    # Removed tile, gpu_id etc. as those are handled by the inference script itself

    args = parser.parse_args()

    # Basic check if the provided script path exists
    if not os.path.isfile(args.inference_script_path):
        print(f"Error: Provided inference script path not found: {args.inference_script_path}")
        print("Please provide the correct path using --inference_script_path")
        sys.exit(1)

    process_images(
        args.input_dir,
        args.output_dir,
        args.target_size,
        args.inference_script_path,
        args.model,
        args.scale,
        args.face_enhance
    )
