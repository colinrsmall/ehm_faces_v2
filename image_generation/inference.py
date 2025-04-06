import os
import pandas as pd
from datetime import datetime
from diffusers import DiffusionPipeline
import torch
from tqdm.auto import tqdm
import argparse
import zipfile
from PIL import Image
import torchvision.transforms as transforms

# --- Configuration ---
DEFAULT_MODEL_NAME = "ostris/Flex.1-alpha"
DEFAULT_LORA_PATH = "/home/colin/ai-toolkit/output/ehm_facegen_v1" # Path to your trained LoRA weights
DEFAULT_LORA_NAME = "ehm_facegen_v1_000003000.safetensors"
DEFAULT_INPUT_CSV = "input_players.csv" # CSV with player data
DEFAULT_OUTPUT_DIR = "generated_images" # Directory to save generated images
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TEMPLATE = "h0ck3y A studio headshot of an 18-year-old hockey player against a dark blue (#131A23) background. The player is wearing shoulder pads under their jersey, is sitting square to the camera, and is smiling. The player is from {nationality} and is named {name}."
FILENAME_DATE_FORMAT = "%d_%m_%Y" # Date format for output filename
NUM_INFERENCE_STEPS = 40 # Number of diffusion steps
GUIDANCE_SCALE = 4 # Guidance scale for inference
DEFAULT_NUM_IMAGES = 25 # Default number of images to generate
DEFAULT_GENERATE_SIZE = 512 # Size for the initial generation
DEFAULT_OUTPUT_SIZE = 0    # Final output size after potential resize (0 = no resize)
DEFAULT_BATCH_SIZE = 0     # Number of images per zip file (0 = no zipping)
DEFAULT_MODNET_MODEL_PATH = "./modnet_photographic_portrait_matting.pt" # << --- IMPORTANT: Update this path

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate hockey player images using a fine-tuned model.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Base diffusion model name.")
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH, help="Path to the trained LoRA weights directory or file.")
    parser.add_argument("--lora_name", type=str, default=DEFAULT_LORA_NAME, help="Name of the LoRA weights file.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save generated images.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to run inference on ('cuda' or 'cpu').")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE, help="Guidance scale.")
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES, help="Number of images to generate.")
    parser.add_argument("--use_torch_compile", action='store_true', help="Enable torch.compile (PyTorch 2.0+) for potential speedup (adds compile time).")
    parser.add_argument("--fuse_lora", action='store_true', help="Fuse LoRA weights into the base model before generation.")
    parser.add_argument("--generate_size", type=int, default=DEFAULT_GENERATE_SIZE, help="Size (width and height) for initial image generation.")
    parser.add_argument("--output_size", type=int, default=DEFAULT_OUTPUT_SIZE, help="Final square size after optional Lanczos downsampling (0 = no resize).")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of images per zip file (0 = no zipping).")
    parser.add_argument("--modnet_model_path", type=str, default=DEFAULT_MODNET_MODEL_PATH, help="Path to the MODNet TorchScript model (.pt file).")
    return parser.parse_args()

def run_inference(args):
    """Loads the model and generates images based on the input CSV."""

    print(f"Using device: {args.device}")

    # --- Load Model and LoRA ---
    print(f"Loading base model: {args.model_name}")
    try:
        dtype = torch.float32 # Default for CPU
        if args.device == "cuda":
            if torch.cuda.is_bf16_supported():
                print("  (Using bfloat16 precision as hardware supports it)")
                dtype = torch.bfloat16
            else:
                print("  (Using float16 precision - bf16 not supported by hardware)")
                dtype = torch.float16

        pipe = DiffusionPipeline.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
        ).to(args.device)
        pipe.set_progress_bar_config(leave=False)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # --- Load MODNet Model --- #
    modnet_model = None
    if not os.path.exists(args.modnet_model_path):
        print(f"Warning: MODNet model not found at {args.modnet_model_path}. Background removal will be skipped.")
    else:
        try:
            print(f"Loading MODNet model from: {args.modnet_model_path}")
            modnet_model = torch.jit.load(args.modnet_model_path).to(args.device).eval()
            print("MODNet model loaded successfully.")
        except Exception as modnet_e:
            print(f"Error loading MODNet model: {modnet_e}. Background removal will be skipped.")
            modnet_model = None

    # --- Define MODNet Preprocessing --- #
    # Typical MODNet preprocessing: Resize, ToTensor, Normalize
    modnet_input_size = 512 # MODNet typically expects 512x512 input
    modnet_preprocess = transforms.Compose([
        transforms.Resize((modnet_input_size, modnet_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Common normalization for portrait matting
    ])

    # --- Load LoRA Weights ---
    print(f"Loading LoRA weights from: {args.lora_path}")
    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA path not found: {args.lora_path}")
        return
    try:
        # Check if it's a directory (newer format) or a single file (older format)
        if os.path.isdir(args.lora_path):
             pipe.load_lora_weights(args.lora_path, weight_name=args.lora_name)
        elif os.path.isfile(args.lora_path):
             # Older diffusers might expect the state dict path directly
             pipe.load_lora_weights(os.path.dirname(args.lora_path), weight_name=args.lora_name)
             print(f"  (Loaded single LoRA file: {args.lora_name}")
        else:
             print(f"Error: LoRA path is neither a valid directory nor file: {args.lora_path}")
             return
        print("LoRA weights loaded successfully.")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        # Optionally continue without LoRA if desired, but safer to exit
        return

    # --- Optional: Fuse LoRA ---
    if args.fuse_lora:
        print("Attempting to fuse LoRA weights into the base model...")
        try:
            pipe.fuse_lora()
            print("LoRA weights fused successfully.")
        except Exception as e:
            print(f"Could not fuse LoRA weights: {e} (Ensure LoRA was loaded correctly)")
            # Decide if you want to continue without fusing or exit
            # return

    # --- Optional: Enable torch.compile ---
    # Note: Targeting pipe.transformer for Flux models instead of pipe.unet
    if args.use_torch_compile:
         # Requires PyTorch 2.0+
         if hasattr(torch, 'compile') and args.device == "cuda": # Only compile for CUDA
             print("Attempting to compile model's transformer block with torch.compile (this may take a moment)...")
             # Options: "default", "reduce-overhead", "max-autotune"
             if hasattr(pipe, 'transformer'):
                 try:
                     pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
                     print("Transformer compiled successfully.")
                 except Exception as e:
                     print(f"Torch compile failed on transformer: {e}")
             else:
                 print("Torch compile skipped: pipe.transformer attribute not found.")
         elif not hasattr(torch, 'compile'):
             print("torch.compile not available (requires PyTorch 2.0+). Cannot enable.")
         elif args.device != "cuda":
             print("torch.compile currently only enabled for CUDA device in this script.")

    # --- Prepare Input/Output ---
    print(f"Reading player data from: {args.input_csv}")
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found: {args.input_csv}")
        return

    try:
        df = pd.read_csv(args.input_csv)
        # Check for required columns
        required_cols = ["first name", "last name", "date of birth", "nationality"]
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Input CSV must contain columns: {', '.join(required_cols)}")
            return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Limit the number of images if requested
    if args.num_images > 0:
        print(f"Limiting generation to the first {args.num_images} players from the CSV.")
        df = df.head(args.num_images)
    elif args.num_images == 0:
        print("Number of images set to 0. No images will be generated.")
        return
    # else: num_images is -1 (default), generate all

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving generated images to: {args.output_dir}")

    # --- Batch Zipping Setup ---
    batch_images = []
    batch_num = 1
    zip_created = False

    # --- Generation Loop ---
    print(f"Starting image generation for {len(df)} players...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Images"):
        try:
            first_name = str(row["first name"]).strip()
            last_name = str(row["last name"]).strip()
            dob_str = str(row["date of birth"]).strip()
            nationality = str(row["nationality"]).strip()
            name = f"{first_name} {last_name}"

            if not all([first_name, last_name, dob_str, nationality]):
                 print(f"Warning: Skipping row {index+1} due to missing data.")
                 continue

            # Format date of birth for filename
            try:
                 # IMPORTANT: Adjust the format string '%d.%m.%Y' if your CSV uses a different date format!
                 dob_obj = datetime.strptime(dob_str, '%d.%m.%Y')
                 dob_formatted = dob_obj.strftime(FILENAME_DATE_FORMAT)
            except ValueError: # Handle potential date parsing errors
                 print(f"Warning: Could not parse date '{dob_str}' for {name} using format '%d.%m.%Y'. Using original string as fallback.")
                 dob_formatted = dob_str.replace(".", "_").replace("/", "_").replace("-", "_").replace(" ", "_") # Basic fallback

            # Construct prompt
            prompt = PROMPT_TEMPLATE.format(nationality=nationality, name=name)

            # Construct filename
            filename = f"{first_name}_{last_name}_{dob_formatted}.png"
            output_path = os.path.join(args.output_dir, filename)

            # Check if file already exists to avoid re-generating (optional)
            if os.path.exists(output_path):
                 # print(f"Skipping {filename}, already exists.")
                 continue

            # Generate image
            # Use torch.inference_mode() for potentially lower memory and faster execution
            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    width=args.generate_size, # Use generate_size
                    height=args.generate_size, # Use generate_size
                ).images[0]

            # --- Background Removal using MODNet (if loaded) --- #
            image_rgba = None
            if modnet_model is not None:
                try:
                    # print(f"Removing background for {name}...")
                    image_rgb = image.convert("RGB") # Ensure image is RGB
                    original_size = image_rgb.size

                    # Preprocess
                    input_tensor = modnet_preprocess(image_rgb).unsqueeze(0).to(args.device)

                    # Inference
                    with torch.no_grad():
                        # MODNet TorchScript might return tuple, matte is often first element
                        matte_tensor = modnet_model(input_tensor)
                        if isinstance(matte_tensor, (tuple, list)):
                             matte_tensor = matte_tensor[0]

                    # Postprocess matte
                    matte_tensor = matte_tensor.squeeze().cpu()
                    # Resize matte back to original image size
                    matte_resized = transforms.ToPILImage()(matte_tensor.unsqueeze(0))
                    matte_final = matte_resized.resize(original_size, Image.BILINEAR) # Use BILINEAR for smooth matte

                    # Combine image and matte
                    image_rgba = image_rgb.copy()
                    image_rgba.putalpha(matte_final)

                    # print(f"Background removed for {name}.")
                except Exception as bg_err:
                    print(f"Warning: MODNet background removal failed for {name}: {bg_err}")
                    image_rgba = image # Fallback to original if removal fails
            else:
                 # If MODNet wasn't loaded, just use the original image
                 image_rgba = image

            # --- Optional Resizing (Applied AFTER background removal if performed) ---
            # Use image_rgba if available, otherwise original image
            image_to_resize = image_rgba if image_rgba is not None else image
            if args.output_size > 0 and args.output_size < image_to_resize.size[0]: # Compare with current size
                try:
                    # print(f"Resizing image from {args.generate_size}x{args.generate_size} to {args.output_size}x{args.output_size} using Lanczos...")
                    image_final = image_to_resize.resize((args.output_size, args.output_size), Image.LANCZOS)
                except Exception as resize_e:
                    print(f"Warning: Could not resize image for {name}: {resize_e}")
                    image_final = image_to_resize # Fallback on error
            elif args.output_size > 0 and args.output_size >= image_to_resize.size[0]:
                print(f"Warning: output_size ({args.output_size}) >= current size ({image_to_resize.size[0]}). Skipping resize for {name}.")
                image_final = image_to_resize
            else:
                image_final = image_to_resize # No resize needed or specified

            # Save image
            image_final.save(output_path) # Save the potentially resized RGBA image

            # --- Batch Zipping Logic ---
            if args.batch_size > 0:
                batch_images.append(output_path)
                zip_created = False # Reset flag for this batch

                # Check if batch is full or if it's the last image
                is_last_image = (index == len(df) - 1)
                if len(batch_images) == args.batch_size or (is_last_image and batch_images):
                    zip_filename = os.path.join(args.output_dir, f"batch_{batch_num}.zip")
                    print(f"\nCreating zip archive: {zip_filename} with {len(batch_images)} images...")
                    try:
                        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for img_path in tqdm(batch_images, desc=f"Zipping Batch {batch_num}", leave=False):
                                if os.path.exists(img_path):
                                    zipf.write(img_path, os.path.basename(img_path))
                                else:
                                     print(f"Warning: Image file not found for zipping: {img_path}")
                        zip_created = True
                    except Exception as zip_e:
                        print(f"Error creating zip file {zip_filename}: {zip_e}")

                    # Reset for next batch
                    batch_images = []
                    batch_num += 1

        except Exception as e:
            print(f"Error generating image for row {index+1} ({name}): {e}")
            # Continue to the next player

    # Final check for any remaining images if zipping was interrupted or loop ended abruptly
    if args.batch_size > 0 and batch_images and not zip_created:
         zip_filename = os.path.join(args.output_dir, f"batch_{batch_num}.zip")
         print(f"\nCreating final zip archive: {zip_filename} with {len(batch_images)} images...")
         try:
             with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                 for img_path in tqdm(batch_images, desc=f"Zipping Final Batch", leave=False):
                      if os.path.exists(img_path):
                          zipf.write(img_path, os.path.basename(img_path))
                      else:
                           print(f"Warning: Image file not found for zipping: {img_path}")
         except Exception as zip_e:
             print(f"Error creating final zip file {zip_filename}: {zip_e}")

    print("Image generation complete.")

if __name__ == "__main__":
    args = parse_arguments()
    run_inference(args)