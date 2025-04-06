# /Users/colinsmall/github/ehm/finetuning/coaches/outpaint_images.py

import os
import argparse
import csv
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm

# --- Configuration ---
DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-inpainting"
TARGET_SIZE = 200 # Target square size (width and height)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Add any specific pipeline arguments if needed (e.g., steps, guidance_scale)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

def prepare_image_and_mask(original_image_path, target_size):
    """
    Loads the original image, creates a centered canvas, and generates the mask.

    Args:
        original_image_path (str): Path to the input image (e.g., 176x192).
        target_size (int): The desired output size (e.g., 200).

    Returns:
        tuple: (canvas_image, mask_image) PIL Images or (None, None) on error.
    """
    try:
        original_img = Image.open(original_image_path).convert("RGB")
        orig_width, orig_height = original_img.size

        # Calculate padding
        pad_width = (target_size - orig_width) // 2
        pad_height = (target_size - orig_height) // 2

        # Ensure padding is non-negative
        if pad_width < 0 or pad_height < 0:
            print(f"Error: Original image {os.path.basename(original_image_path)} ({orig_width}x{orig_height}) is larger than target size ({target_size}x{target_size}). Cannot outpaint.")
            # Or potentially resize down first? For now, skip.
            return None, None

        # Create canvas and paste original image
        # Using RGB for the canvas as the pipeline expects RGB
        canvas = Image.new("RGB", (target_size, target_size), (127, 127, 127)) # Grey background
        paste_position = (pad_width, pad_height)
        canvas.paste(original_img, paste_position)

        # Create mask: white where to inpaint (outside), black where to keep (inside)
        mask = Image.new("L", (target_size, target_size), 255) # White background
        draw = ImageDraw.Draw(mask)
        # Draw black rectangle over the area of the original pasted image
        mask_inner_box = (
            pad_width,
            pad_height,
            pad_width + orig_width,
            pad_height + orig_height
        )
        draw.rectangle(mask_inner_box, fill=0) # Black fill

        return canvas, mask

    except FileNotFoundError:
        print(f"Error: Original image not found at {original_image_path}")
        return None, None
    except Exception as e:
        print(f"Error preparing image/mask for {os.path.basename(original_image_path)}: {e}")
        return None, None


def outpaint_images(input_dir, prompts_csv, output_dir, model_name, device):
    """
    Performs outpainting on images based on prompts from a CSV file.
    """
    # --- Setup ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    if not os.path.isfile(prompts_csv):
        print(f"Error: Prompts CSV file not found: {prompts_csv}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- Load Prompts ---
    prompts_data = []
    try:
        with open(prompts_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'file_name' not in reader.fieldnames or 'prompt' not in reader.fieldnames:
                 print(f"Error: CSV file {prompts_csv} must contain 'file_name' and 'prompt' columns.")
                 return
            for row in reader:
                prompts_data.append(row)
        print(f"Loaded {len(prompts_data)} prompts from {prompts_csv}")
    except Exception as e:
        print(f"Error reading prompts CSV {prompts_csv}: {e}")
        return

    # --- Load Pipeline ---
    print(f"Loading inpainting model '{model_name}' on device '{device}'...")
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, # Use float16 for GPU
            # revision="fp16" # Uncomment if using fp16 weights explicitly
        ).to(device)
        # Optional: Add schedulers or other optimizations if needed
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_xformers_memory_efficient_attention() # If xformers is installed
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Process Images ---
    processed_count = 0
    error_count = 0
    print(f"Starting outpainting process for {len(prompts_data)} images...")

    for item in tqdm(prompts_data, desc="Outpainting images"):
        filename = item.get('file_name')
        prompt = item.get('prompt')

        if not filename or not prompt:
            print(f"Warning: Skipping row with missing filename or prompt: {item}")
            error_count += 1
            continue

        original_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename) # Save with the same name

        # --- Prepare ---
        canvas_image, mask_image = prepare_image_and_mask(original_image_path, TARGET_SIZE)

        if canvas_image is None or mask_image is None:
            error_count += 1
            continue

        # --- Inpaint ---
        try:
            # Generator for reproducibility if needed
            # generator = torch.Generator(device=device).manual_seed(0)

            result_image = pipe(
                prompt=prompt,
                image=canvas_image,
                mask_image=mask_image,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                # generator=generator,
                height=TARGET_SIZE,
                width=TARGET_SIZE
            ).images[0]

            # --- Save ---
            result_image.save(output_image_path)
            processed_count += 1

        except Exception as e:
            print(f"Error during inpainting for {filename}: {e}")
            error_count += 1
            # Optional: Add retry logic here if needed

    print("\nOutpainting complete.")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors/Skipped: {error_count}")
    print(f"  Output saved to: {output_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Default input is the output of the resize script
    default_input_dir = os.path.join(script_dir, "resized_images")
    # Default prompts CSV is the output of the generate prompts script
    default_prompts_csv = os.path.join(script_dir, "outpainting_prompts.csv")
    # Default output directory
    default_output_dir = os.path.join(script_dir, "outpainted_images")

    parser = argparse.ArgumentParser(description="Outpaint images using Stable Diffusion Inpainting.")
    parser.add_argument("--input_dir", type=str, default=default_input_dir,
                        help="Directory containing the original (cropped) images.")
    parser.add_argument("--prompts_csv", type=str, default=default_prompts_csv,
                        help="CSV file containing 'file_name' and 'prompt' columns.")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help="Directory to save the outpainted images.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Name of the Stable Diffusion Inpainting model.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Device to run inference on ('cuda' or 'cpu').")
    # Add more arguments if you want to control steps, guidance etc. via command line
    # parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="Number of inference steps.")
    # parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE, help="Guidance scale.")

    args = parser.parse_args()

    outpaint_images(args.input_dir, args.prompts_csv, args.output_dir, args.model_name, args.device)
