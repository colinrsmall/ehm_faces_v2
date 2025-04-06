import os
import csv
import shutil
from tqdm.auto import tqdm
import re # For cleaning directory names

# --- Configuration ---
CSV_FILE = "captions.csv"
SOURCE_DIR = "training_set" # Directory containing the original images
DEST_BASE_DIR = "sorted_by_nationality" # Base directory to store sorted images

def sanitize_foldername(name):
    """Removes or replaces characters invalid for directory names."""
    # Replace slashes and parentheses with hyphens
    name = re.sub(r'[\\/\(\)]', '-', name)
    # Remove other potentially problematic characters (allow letters, numbers, underscore, hyphen, space)
    name = re.sub(r'[^\w\s-]', '', name).strip()
    # Replace multiple spaces/hyphens with a single hyphen
    name = re.sub(r'[-\s]+', '-', name)
    return name if name else "Unknown" # Return "Unknown" if name becomes empty

def sort_images():
    """Reads the CSV and copies images into nationality-based folders."""

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    # Create the base destination directory if it doesn't exist
    os.makedirs(DEST_BASE_DIR, exist_ok=True)
    print(f"Ensured base destination directory exists: '{DEST_BASE_DIR}'")

    copied_count = 0
    skipped_count = 0
    error_count = 0

    try:
        with open(CSV_FILE, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            # Convert reader to list to use tqdm
            rows = list(reader)
            print(f"Reading {len(rows)} entries from '{CSV_FILE}'...")

            for row in tqdm(rows, desc="Sorting Images"):
                try:
                    image_filename = row.get("Image File Name")
                    nationality = row.get("Nationality")

                    if not image_filename or not nationality:
                        print(f"Warning: Skipping row due to missing data: {row}")
                        skipped_count += 1
                        continue

                    # Create a safe directory name for the nationality
                    nationality_folder_name = sanitize_foldername(nationality)
                    dest_nationality_dir = os.path.join(DEST_BASE_DIR, nationality_folder_name)

                    # Create the specific nationality directory
                    os.makedirs(dest_nationality_dir, exist_ok=True)

                    # Define source and destination paths for the image
                    source_image_path = os.path.join(SOURCE_DIR, image_filename)
                    dest_image_path = os.path.join(dest_nationality_dir, image_filename)

                    # Check if source image exists before copying
                    if os.path.exists(source_image_path):
                        # Copy the image (copy2 preserves metadata like modification time)
                        shutil.copy2(source_image_path, dest_image_path)
                        copied_count += 1
                    else:
                        print(f"Warning: Source image not found, skipping: {source_image_path}")
                        skipped_count += 1

                except Exception as e:
                    print(f"Error processing row {row}: {e}")
                    error_count += 1

    except FileNotFoundError:
        print(f"Error: Could not open CSV file '{CSV_FILE}'.")
        return
    except KeyError as e:
         print(f"Error: CSV file might be missing expected column header: {e}. Headers found: {reader.fieldnames}")
         return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print(f"\nSorting complete.")
    print(f"  Images copied: {copied_count}")
    print(f"  Rows skipped (missing data/image): {skipped_count}")
    print(f"  Errors encountered: {error_count}")
    print(f"Sorted images are located in: '{DEST_BASE_DIR}'")


if __name__ == "__main__":
    sort_images()
