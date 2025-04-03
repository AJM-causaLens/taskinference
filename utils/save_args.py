from datetime import datetime
import json
import os

def save_args_as_json_or_markdown(args, file_path_base):
    # Convert args to a dictionary if it's an object
    os.makedirs(os.path.dirname(file_path_base), exist_ok=True)
    args_dict = vars(args) if not isinstance(args, dict) else args
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"{file_path_base}/{timestamp}"
    # Save as JSON
    with open(file_path + ".json", "w") as json_file:
        json.dump(args_dict, json_file, indent=4)