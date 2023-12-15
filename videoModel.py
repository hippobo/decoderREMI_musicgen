import json

# Load the existing encode JSON file
with open('tokenizer_js_encode.json', 'r') as file:
    encode_data = json.load(file)

# Modify each key in the dictionary
# Assuming the current keys are like "['Pitch_21']", "['Velocity_119']", etc.
modified_encode_data = {key.strip("[]'"): value for key, value in encode_data.items()}

# Save the modified data back to the file
with open('tokenizer_js_encode_updated.json', 'w') as file:
    json.dump(modified_encode_data, file, indent=4)
