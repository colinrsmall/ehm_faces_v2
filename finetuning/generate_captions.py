import os
import csv
import json
from tqdm.auto import tqdm
from chat_operations import sessionless_call_function, SYSTEM_MESSAGE

SCHEMA = {
    "type": "object",
    "properties": {
        "nationality": {
            "type": "string",
            "description": "The nationality of the hockey player.",
            "enum": ["Slavic", "Swedish/Danish/Norwegian", "North American", "Central European (Swiss, German, French, Austrian, Polish, Italian etc.)", "Finnish/Estonian", "Russian/Belorussian/Kazakh", "Latvian", "Mexican", "Asian (Japanese, Korean, Chinese)", "Southern Europe (Hungarian, Serbian, etc.)", "Spanish", "Lithuanian"],
        },
    },
    "required": ["nationality"],
    "additionalProperties": False,
}

TRAINING_SET_DIR = "final_processed_dataset"
OUTPUT_CSV = "captions.csv"
MODEL = "gpt-4o-mini"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def generate_captions():
    if not os.path.exists(TRAINING_SET_DIR):
        print(f"Error: Training set directory '{TRAINING_SET_DIR}' not found.")
        return
    if "MV_OPENAI_KEY" not in os.environ:
         print("Error: MV_OPENAI_KEY environment variable not set. Please set it before running.")
         return

    results = []
    print(f"Processing images in '{TRAINING_SET_DIR}'...")

    for filename in tqdm(os.listdir(TRAINING_SET_DIR)):
        name_part, extension = os.path.splitext(filename)
        if extension.lower() not in IMAGE_EXTENSIONS:
            continue

        # print(f"Processing {filename}...")

        try:
            name_parts = name_part.split('_')
            if len(name_parts) >= 2:
                player_name = " ".join(name_parts[:2])
            else:
                 player_name = name_part
            # print(f"  Extracted Player Name: {player_name}")

            prompt = f"Determine the likely nationality of a hockey player named '{player_name}'. Some French-sounding names may be of Canadian nationality. Respond only with the JSON according to the provided schema."

            # print(f"  Querying LLM for nationality...")
            try:
                llm_response_str = sessionless_call_function(
                    system_message=SYSTEM_MESSAGE,
                    message=prompt,
                    schema=SCHEMA,
                    model=MODEL
                )
                llm_response = json.loads(llm_response_str)
                nationality = llm_response.get("nationality", "Unknown")
                # print(f"  Received Nationality: {nationality}")

            except json.JSONDecodeError:
                # print(f"  Error: Could not parse LLM JSON response: {llm_response_str}")
                nationality = "ErrorParsingLLMResponse"
            except Exception as llm_err: 
                #  print(f"  Error calling LLM: {llm_err}")
                 nationality = "ErrorCallingLLM"


            caption = f"A studio headshot of a young hockey player without their helmet. The player is wearing should pads under their jersey, is sitting upright and square to the camera. The background is a solid blue color. The player is {nationality}."
            # print(f"  Generated Caption: {caption}")

            results.append([filename, player_name, nationality, caption])

        except Exception as e:
            # print(f"  Error processing file {filename}: {e}")
            results.append([filename, "Error", "Error", "Error"]) 

    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image File Name", "Player Name", "Nationality", "Caption"]) 
            writer.writerows(results)
        print(f"\nSuccessfully generated captions and saved to '{OUTPUT_CSV}'.")
    except IOError as e:
         print(f"\nError writing to CSV file '{OUTPUT_CSV}': {e}")

if __name__ == "__main__":
    generate_captions()