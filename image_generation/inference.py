import os
import pandas as pd
from datetime import datetime
from diffusers import DiffusionPipeline
import torch
from tqdm.auto import tqdm
import argparse

# --- Configuration ---
DEFAULT_MODEL_NAME = "ostris/Flex.1-alpha"
DEFAULT_LORA_PATH = "/home/colin/ai-toolkit/output/ehm_facegen_v1" # Path to your trained LoRA weights
DEFAULT_LORA_NAME = "ehm_facegen_v1_000003000.safetensors"
DEFAULT_INPUT_CSV = "input_players.csv" # CSV with player data
DEFAULT_OUTPUT_DIR = "generated_images" # Directory to save generated images
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TEMPLATE = "h0ck3y A studio headshot of an 18-year-old hockey player. The player is wearing shoulder pads under their jersey, is sitting square to the camera, and is smiling. The player is from {nationality} and is named {name}."
FILENAME_DATE_FORMAT = "%d_%m_%Y" # Date format for output filename
NUM_INFERENCE_STEPS = 40 # Number of diffusion steps
GUIDANCE_SCALE = 4 # Guidance scale for inference
DEFAULT_NUM_IMAGES = 25 # Default number of images to generate
DEFAULT_WIDTH = 512 # Default output image width
DEFAULT_HEIGHT = 512 # Default output image height

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
    parser.add_argument("--use_xformers", action='store_true', help="Enable xFormers memory efficient attention for potential speedup.")
    parser.add_argument("--use_torch_compile", action='store_true', help="Enable torch.compile (PyTorch 2.0+) for potential speedup (adds compile time).")
    parser.add_argument("--fuse_lora", action='store_true', help="Fuse LoRA weights into the base model before generation.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Width of the generated images.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Height of the generated images.")
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
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # --- Optional: Enable xFormers ---
    if args.use_xformers:
        print("Attempting to enable xFormers memory efficient attention...")
        try:
            # Check if xformers is installed before attempting to enable
            import xformers
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled successfully.")
        except ImportError:
            print("xFormers not installed. Cannot enable. Run: pip install xformers")
        except Exception as e:
             print(f"Could not enable xFormers: {e}")

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
    if args.use_torch_compile:
        # Requires PyTorch 2.0+
        if hasattr(torch, 'compile') and args.device == "cuda": # Only compile for CUDA
            print("Attempting to compile UNet with torch.compile (this may take a moment)...")
            # Options: "default", "reduce-overhead", "max-autotune"
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                print("UNet compiled successfully.")
            except Exception as e:
                print(f"Torch compile failed: {e}")
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
                    width=args.width,
                    height=args.height
                ).images[0]

            # Save image
            image.save(output_path)

        except Exception as e:
            print(f"Error generating image for row {index+1} ({name}): {e}")
            # Continue to the next player

    print("Image generation complete.")

if __name__ == "__main__":
    args = parse_arguments()
    run_inference(args)