import json
import re

def clean_test_data(input_file, output_file):
    """
    Cleans a JSON file by ensuring `s.gold.index` is a list of integers within the valid range [0, 12].
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file where cleaned data will be saved.
    """
    try:
        # Load the data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        cleaned_data = []
        for item in data:
            # Ensure `s.gold.index` exists
            if "s.gold.index" in item:
                try:
                    # Extract integers from `s.gold.index` regardless of formatting
                    gold_indices = []
                    if isinstance(item["s.gold.index"], str):
                        gold_indices = [int(i) for i in re.findall(r'-?\d+', item["s.gold.index"])]
                    elif isinstance(item["s.gold.index"], list):
                        gold_indices = item["s.gold.index"]
                    
                    # Filter indices to include only values in the range [0, 12)
                    valid_indices = [i for i in gold_indices if 0 <= i < 12]
                    
                    # Update the item with the cleaned indices
                    item["s.gold.index"] = valid_indices
                    cleaned_data.append(item)
                except (ValueError, TypeError) as e:
                    print(f"Skipping item with invalid `s.gold.index`: {item}. Error: {e}")
            else:
                print(f"Skipping item missing `s.gold.index`: {item}")
        
        # Save the cleaned data
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Cleaned data saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing file: {e}")

# Example usage
input_file = "HW3_dataset/test_with_s_gold_index.json"
output_file = "HW3_dataset/test_with_s_gold_index_cleaned.json"
clean_test_data(input_file, output_file)