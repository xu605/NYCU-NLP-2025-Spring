import json
import subprocess
import os
import time
import tqdm

def generate_s_gold_indices(data, delay=4):
    for item in tqdm.tqdm(data):
        utterance = item["u"]
        response = item["r"]
        statements = item["s"]

        prompt = (
            f"utterance: '{utterance}', situation statements: '{statements}', and the response: '{response}'.\n"
            "Hint: The response is a statement that is generated based on the utterance and the situation statements.\n"
            "The situation statements are a list of statements that provide context for the utterance.\n"
            "There are 12 statements in the situation statements.\n"
            "Possible actions:\n"
            "Step 1: Identify the indices of the statements (0-indexed) that are most relevant for evaluating the quality of the response.\n"
            "Step 2: Return only a list of 5~9 integers representing the indices of the most relevant statements, sorted in descending order of importance.\n"
        )

        try:
            curl_command = [
                "curl",
                "https://api.groq.com/openai/v1/chat/completions",
                "-s",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: key",
                "-d", json.dumps({
                    "model": "gemma-7b-it",
                    "messages": [
                        {"role": "system", "content": "You are an excelent assistant to identify relevant information to evaluate responses. Given the utterance, and situations, please identify the most relevant statements to evaluate the quality of the response. Do not provide any explanations, only the indices of the statements. For example, if the most relevant statements are the first and third statements, you should provide '[0, 2]'."},
                        {"role": "user", "content": prompt}
                    ]
                })
            ]

            # Parse the indices from the response
            response = subprocess.check_output(curl_command, stderr=subprocess.STDOUT)
            response_str = response.decode('utf-8')
            response_json = json.loads(response_str)
            # print(response_json)
            if "choices" not in response_json or len(response_json["choices"]) == 0:
                print(f"Error: No response choices found. Response: {response_json}")
                item["s.gold.index"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            else:
                indices = response_json["choices"][0]["message"]["content"]
                import re
                gold_indices = [int(i) for i in re.findall(r'-?\d+',indices)]
                item["s.gold.index"] = indices
            # print(f"Generated s.gold.index: {indices}")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.output.decode('utf-8')}")
        
        time.sleep(delay)

    return data

def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Load the test.json data
test_data_path = "./HW3_dataset/test.json"
with open(test_data_path, 'r') as f:
    test_data = json.load(f)

# Generate s.gold.index values
test_data_with_indices = generate_s_gold_indices(test_data)

# Save the modified data
output_path = "./HW3_dataset/test_with_s_gold_index_1.json"
save_data(output_path, test_data_with_indices)