import json
import os

from notebooks.constants import RAW_FOLLOWING_PATH


# Define the input and output file names
def transform_data(input_filename, output_filename):
    # Check if the output file already exists
    if os.path.isfile(output_filename):
        print(f"The file '{output_filename}' already exists. Please remove it before running this script again.")
        return  # Exit the function early

    # Open and read the input JSON file
    with open(input_filename, 'r') as file:
        data = json.load(file)

    # Process the data to match the desired structure
    transformed_data = []
    for relationship in data['relationships_following']:
        for string_data in relationship['string_list_data']:
            transformed_data.append({
                "username": string_data['value'],
                "timestamp": string_data['timestamp']
            })

    # Write the transformed data to the output JSON file
    with open(output_filename, 'w') as file:
        json.dump(transformed_data, file, indent=4)
        print(f"Data has been written to '{output_filename}' successfully.")


# Execute the function
if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    home_directory = os.path.join(current_directory, '..')
    input_filename = os.path.join(home_directory, RAW_FOLLOWING_PATH)
    output_filename = os.path.join(home_directory, 'data/raw/output.json')
    transform_data(input_filename, 'output.json')
