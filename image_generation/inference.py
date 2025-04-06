
import os
import pandas as pd
from datetime import datetime
from diffusers import DiffusionPipeline
import torch
from tqdm.auto import tqdm
import argparse

# --- Configuration ---
DEFAULT_MODEL_NAME = "ostris/Flex.1-alpha"
DEFAULT_LORA_PATH = "output/my_first_flex_lora_v1/step_2000" # Path to your trained LoRA weights
DEFAULT_INPUT_CSV = "input_players.csv" # CSV with player data
DEFAULT_OUTPUT_DIR = "generated_images" # Directory to save generated images
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TEMPLATE = "h0ck3y A studio headshot of a young hockey player. The player is wearing shoulder pads under their jersey, is sitting square to the camera, and is smiling. The player is from {nationality} and is named {name}."
FILENAME_DATE_FORMAT = "%d_%m_%Y" # Date format for output filename
NUM_INFERENCE_STEPS = 30 # Number of diffusion steps
GUIDANCE_SCALE = 5 # Guidance scale for inference

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate hockey player images using a fine-tuned model.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Base diffusion model name.")
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH, help="Path to the trained LoRA weights directory or file.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save generated images.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to run inference on ('cuda' or 'cpu').")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE, help="Guidance scale.")
    return parser.parse_args()

def run_inference(args):
    """Loads the model and generates images based on the input CSV."""

    print(f"Using device: {args.device}")

    # --- Load Model and LoRA ---
    print(f"Loading base model: {args.model_name}")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32, # float16 only on CUDA
            variant="fp16" if args.device == "cuda" else None, # fp16 only on CUDA
        ).to(args.device)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print(f"Loading LoRA weights from: {args.lora_path}")
    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA path not found: {args.lora_path}")
        return
    try:
        # Check if it's a directory (newer format) or a single file (older format)
        if os.path.isdir(args.lora_path):
             pipe.load_lora_weights(args.lora_path)
        elif os.path.isfile(args.lora_path):
             # Older diffusers might expect the state dict path directly
             pipe.load_lora_weights(os.path.dirname(args.lora_path), weight_name=os.path.basename(args.lora_path))
             print(f"  (Loaded single LoRA file: {os.path.basename(args.lora_path)})")
        else:
             print(f"Error: LoRA path is neither a valid directory nor file: {args.lora_path}")
             return
        print("LoRA weights loaded successfully.")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        # Optionally continue without LoRA if desired, but safer to exit
        return

    # pipe.fuse_lora() # Optional: Fuse LoRA for potentially faster inference after loading

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
                 # Attempt to parse common date formats if needed, but assume it's parsable for now
                 # IMPORTANT: Adjust the format string '%Y-%m-%d' if your CSV uses a different date format!
                 dob_obj = datetime.strptime(dob_str, '%Y-%m-%d')
                 dob_formatted = dob_obj.strftime(FILENAME_DATE_FORMAT)
            except ValueError:
                 print(f"Warning: Could not parse date '{dob_str}' for {name} using format '%Y-%m-%d'. Using original string as fallback.")
                 dob_formatted = dob_str.replace("/", "_").replace("-", "_").replace(" ", "_") # Basic fallback

            # Construct prompt
            prompt = PROMPT_TEMPLATE.format(nationality=nationality, name=name)

            # Construct filename
            filename = f"{first_name}_{last_name}_{dob_formatted}.png"
            output_path = os.path.join(args.output_dir, filename)

            # Check if file already exists to avoid re-generating (optional)
            if os.path.exists(output_path):
                 # print(f"Skipping {filename}, already exists.")
                 continue

            # Run inference
            with torch.no_grad(): # Conserve memory
                image = pipe(
                    prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance
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